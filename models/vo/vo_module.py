import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections

import os
import glob

from models.networks.pose_network import PoseEncoder, PoseDecoder, ClassPoseDecoder
# from models.networks.depth_network import DepthEncoder, DepthDecoder
from models.vo.vo_utils import *


# from models.vo.lie_algebra import SO3_CUDA
move_forward = torch.tensor([0., 0., -0.25])
turn_left = 10 * np.pi / 180. * torch.tensor([0, 1, 0])
turn_right = -10 * np.pi / 180. * torch.tensor([0, 1, 0])

ACTION_POSE_MAP = {
    1: move_forward,
    2: turn_left,
    3: turn_right,
}

class VO_Module(nn.Module):
    def __init__(self, cfgs, rl_cfgs=None):
        super().__init__()
        self.cfgs = cfgs

        # 
        self.index_ids = {}
        self.index_ids["rgb"] = torch.arange(3 * self.cfgs["num_frame"]).view(self.cfgs["num_frame"], -1)
        self.index_ids["depth"] = torch.arange(1 * self.cfgs["num_frame"]).view(self.cfgs["num_frame"], -1)
        self.index_ids["depth_d"] = torch.arange(self.cfgs["depth_d_planes"] * self.cfgs["num_frame"]).view(self.cfgs["num_frame"], -1)

        self.rec_iters = int((self.cfgs["num_frame"] - 1) / 2)
        self.frame_ids = [[0]] + [[-(i+1), i+1] for i in range(self.rec_iters)]

        self.loop_ids = [i-self.rec_iters for i in range(self.cfgs["num_frame"])]

        self.rgb_width = cfgs["rgb_width"]
        self.rgb_height = cfgs["rgb_height"]

        # pose net
        self.pose_width = cfgs["pose_width"]
        self.pose_height = cfgs["pose_height"]
        self.pose_size = [cfgs["pose_height"], cfgs["pose_width"]]
        if self.cfgs["act_spec"]:
            self.pose_encoder = PoseEncoder(cfgs, num_layers=18)
            self.pose_decoder_mf = PoseDecoder(cfgs, self.pose_encoder.num_ch_enc)
            self.pose_decoder_tl = PoseDecoder(cfgs, self.pose_encoder.num_ch_enc)
            self.pose_decoder_tr = PoseDecoder(cfgs, self.pose_encoder.num_ch_enc)
            
        elif self.cfgs["classification_model"]:
            self.pose_encoder = PoseEncoder(cfgs, num_layers=18)
            self.pose_decoder = ClassPoseDecoder(cfgs, self.pose_encoder.num_ch_enc)
        
        else:
            self.pose_encoder = PoseEncoder(cfgs, num_layers=18)
            self.pose_decoder = PoseDecoder(cfgs, self.pose_encoder.num_ch_enc)

        # compute losses
        self.train_scale = cfgs['train_scale']

        # intrinsic
        hfov = float(cfgs["hfov"]) * np.pi / 180.
        self.K = torch.from_numpy(np.array([
            [1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., 1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]
        ]))[:3, :3].float()
        self.K_inv = self.K.inverse()

        # load param for rl learning 
        if rl_cfgs is not None:
            self.rl_cfgs = rl_cfgs
            log_path = os.path.join(cfgs["log_path"], cfgs["log_name"])
            self.reuse_from(log_path, rl_cfgs.VO.vo_reuse_id)
        
        num_params = sum(p.numel() for p in self.parameters())
        num_params_M = num_params / 1e6
        print(f"total params is {num_params_M: .2f}")
    
    def forward(self, inputs, training_state):
        self.training_state = training_state
        outputs = {}
        losses_pack = collections.OrderedDict()

        preprocessed_inputs = self.preprocess_inputs(inputs)

        for f_ids in self.frame_ids[1:]:

            for f_id in f_ids:
                # predict poses
                if f_id < 0:
                    pose_inputs, indexs = self.warped_inputs(preprocessed_inputs, [f_id, 0])
                    if self.cfgs["act_spec"]:
                        warped_actions = inputs["action"][:, indexs["depth"][1]]
                    else:
                        warped_actions = None
                    outputs['rotv', f_id, 0], outputs['trans', f_id, 0] = self.infer_pose(pose_inputs, warped_actions)
                else:
                    pose_inputs, indexs = self.warped_inputs(preprocessed_inputs, [0, f_id])
                    if self.cfgs["act_spec"]:
                        warped_actions = inputs["action"][:, indexs["depth"][1]]
                    else:
                        warped_actions = None
                    outputs['rotv', 0, f_id], outputs['trans', 0, f_id] = self.infer_pose(pose_inputs, warped_actions)

            # compute losses
            rec_target, tgt_index, tgt_depth = self.warped_reconstruction_target(inputs, f_ids)
            losses = self.reprojection_loss(rec_target, tgt_index, tgt_depth, outputs, f_ids)

            # collection
            for k, v in losses.items():
                key, s = k
                losses_pack[key] = v if key not in losses_pack else losses_pack[key] + v
        
        if self.cfgs['loop_loss']:
            loop_losses = self.loop_loss(inputs, preprocessed_inputs, outputs)

            # collection
            for k, v in loop_losses.items():
                key, s = k
                losses_pack[key] = v if key not in losses_pack else losses_pack[key] + v

        total_loss = sum(_value for _key, _value in losses_pack.items())
        
        # compute error for eval (not for training)
        with torch.no_grad():
            errors = self.compute_error(inputs, preprocessed_inputs, outputs)
            losses_pack.update(errors)

        return losses_pack, total_loss

    def eval_infer(self, inputs, ids=[0, 1]):
        outputs = {}

        preprocessed_inputs = self.preprocess_inputs(inputs)

        posenet_inputs, indexs = self.warped_inputs(preprocessed_inputs, ids)
        if self.cfgs["act_spec"]:
            warped_actions = inputs["action"][:, indexs["depth"][1]]
        else:
            warped_actions = None
            
        rotv, trans = self.infer_pose(posenet_inputs, warped_actions)
        outputs["pose", -1] = to_matrix(rotv, trans, True)
        return outputs
        
    def infer_pose(self, cnn_inputs, action=None):
        if not self.cfgs["act_spec"]:
            rotv, trans = self.pose_decoder(self.pose_encoder(cnn_inputs))
        else:
            rotv, trans = [], []
            for i, a_id in enumerate(action):
                feature = self.pose_encoder(cnn_inputs[i:i+1])
                if a_id.item() == 0:
                    irotv, itrans = torch.zeros([1, 3]).to(feature.device), torch.zeros([1, 3]).to(feature.device)
                elif a_id.item() == 1:
                    irotv, itrans = self.pose_decoder_mf(feature)
                elif a_id.item() == 2:
                    irotv, itrans = self.pose_decoder_tl(feature)
                elif a_id.item() == 3:
                    irotv, itrans = self.pose_decoder_tr(feature)
                rotv.append(irotv)
                trans.append(itrans)
            rotv = torch.cat(rotv, 0)
            trans = torch.cat(trans, 0)
        return rotv, trans

    def preprocess_inputs(self, inputs):
        preprocessed_inputs = {}
        preprocessed_inputs["rgb"] = (bilinear_interp(inputs["rgb"], self.pose_size) - 0.45) / 0.225
        if self.cfgs["pose_inputs"] == "rgbd":
            preprocessed_inputs["depth"] = bilinear_interp(inputs["depth"], self.pose_size)
        elif self.cfgs["pose_inputs"] == "rgbd_d":
            preprocessed_inputs["depth"] = bilinear_interp(inputs["depth"], self.pose_size)
            preprocessed_inputs["depth_d"] = depth_discretization(preprocessed_inputs["depth"], 1, self.cfgs["depth_d_planes"])

        return preprocessed_inputs
    
    def warped_inputs(self, preprocessed_inputs, f_ids):
        pair_inds = {}
        for key, value in self.index_ids.items():
            pair_inds[key] = torch.cat([value[f_ids[0] + self.rec_iters], value[f_ids[1] + self.rec_iters]], 0)

        if self.cfgs["pose_inputs"] == "rgb":
            posenet_inputs = preprocessed_inputs["rgb"][:, pair_inds["rgb"], ...]

        elif self.cfgs["pose_inputs"] == "rgbd":
            posenet_inputs = torch.cat([preprocessed_inputs["rgb"][:, pair_inds["rgb"], ...],
                                        preprocessed_inputs["depth"][:, pair_inds["depth"], ...]],
                                        dim=1)
        elif self.cfgs["pose_inputs"] == "rgbd_d":
            posenet_inputs = torch.cat([preprocessed_inputs["rgb"][:, pair_inds["rgb"], ...],
                                        preprocessed_inputs["depth_d"][:, pair_inds["depth_d"], ...]],
                                        dim=1)
        else:
            raise NotImplementedError("pose_inputs must be (rgb, rgbd, rgbd_d)")
        return posenet_inputs, pair_inds
    
    def warped_reconstruction_target(self, inputs, f_ids):
        pair_inds = {}
        for key, value in self.index_ids.items():
            pair_inds[key] = torch.cat([value[f_ids[0] + self.rec_iters], 
                                        value[self.rec_iters], 
                                        value[f_ids[1] + self.rec_iters]], 0)
        
        if self.cfgs["rec_target"] == "rgb":
            rec_target = inputs["rgb"][:, pair_inds["rgb"], ...]

        elif self.cfgs["rec_target"] == "rgbd":
            rec_target = torch.cat([inputs["rgb"][:, pair_inds["rgb"], ...],
                                    inputs["depth"][:, pair_inds["depth"], ...]],
                                    dim=1)
            rearrange_id = torch.cat([torch.arange(9).view(3, -1), 9+torch.arange(3).view(3, -1)], 1).reshape(-1)
            rec_target = rec_target[:, rearrange_id, ...]

        elif self.cfgs["rec_target"] == "texture":
            texture = gabor_filter(inputs["rgb"][:, pair_inds["rgb"], ...], sigma=1.0)

            rec_target = torch.cat([inputs["rgb"][:, pair_inds["rgb"], ...],
                                    inputs["depth"][:, pair_inds["depth"], ...],
                                    texture],
                                    dim=1)
            rearrange_id = torch.cat([torch.arange(9).view(3, -1), 
                                        9+torch.arange(3).view(3, -1),
                                        12+torch.arange(9).view(3, -1)], 1).reshape(-1)
            rec_target = rec_target[:, rearrange_id, ...]

        else:
            raise NotImplementedError("pose_inputs must be (rgb, rgbd, texture)")

        rec_index = torch.arange(rec_target.shape[1]).view(3, -1)
        rec_depth = inputs["depth"][:, pair_inds["depth"], ...] * self.cfgs["max_depth"]

        return rec_target, rec_index, rec_depth

    def loop_loss(self, inputs, preprocessed_inputs, outputs):
        for l_id in self.loop_ids[1:]:
            key_rotv = ("rotv", l_id-1, l_id)
            key_trans = ("trans", l_id-1, l_id)
            if key_rotv not in outputs.keys():
                pose_inputs, indexs = self.warped_inputs(preprocessed_inputs, [l_id-1, l_id])
                if self.cfgs["act_spec"]:
                    warped_actions = inputs["action"][:, indexs["depth"][1]]
                else:
                    warped_actions = None
                outputs[key_rotv], outputs[key_trans] = self.infer_pose(pose_inputs, warped_actions)
            
        for l_id in self.loop_ids[2:]:
            key_rotv = ("rotv", self.loop_ids[0], l_id)
            key_trans = ("trans", self.loop_ids[0], l_id)
            if key_rotv not in outputs.keys():
                pose_inputs, indexs = self.warped_inputs(preprocessed_inputs, [self.loop_ids[0], l_id])
                if self.cfgs["act_spec"]:
                    warped_actions = inputs["action"][:, indexs["depth"][1]]
                else:
                    warped_actions = None
                outputs[key_rotv], outputs[key_trans] = self.infer_pose(pose_inputs, warped_actions)
        
        T_inc = {self.loop_ids[0]: torch.eye(4)[np.newaxis, ...].to(preprocessed_inputs["rgb"].device)}
        end_inc = min(self.cfgs["loop_loss_max_inc"], len(self.loop_ids))
        loop_ids = self.loop_ids[1:end_inc]
        for i_id in loop_ids:
            T_inc[i_id] = (to_matrix(outputs['rotv', i_id-1, i_id], outputs['trans', i_id-1, i_id], False)).matmul(T_inc[i_id-1])

        T_loop = {}
        for l_id in loop_ids[1:]:
            T_loop[l_id] = to_matrix(outputs['rotv', self.loop_ids[0], l_id], outputs['trans', self.loop_ids[0], l_id], False)

        losses = {}
        for l_id in loop_ids[1:]:
            losses['loop_loss', l_id] = self.cfgs["loop_loss_coef"] / len(loop_ids[1:]) * \
                                        (T_loop[l_id] - T_inc[l_id]).abs().mean() 
        return losses

    def reprojection_loss(self, rec_target, tgt_index, tgt_depth, pred_poses, f_ids):
        target = rec_target[:, tgt_index[1]]

        losses = {}
        for scale in range(self.train_scale):
            # reconstruction
            warped_images = self.warp_image(rec_target, tgt_index, tgt_depth, pred_poses, f_ids)

            # automask
            identity_reprojection_losses = []
            for f_id in f_ids:
                islice = tgt_index[0] if f_id < 0 else tgt_index[2]
                identity_reprojection_loss = self.compute_reprojection_loss(rec_target[:, islice[:3]], target[:, :3])
                identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).to(target.device) * 1e-5
                identity_reprojection_losses.append(identity_reprojection_loss)
            identity_reprojection_loss, _ = torch.min(torch.cat(identity_reprojection_losses, 1), dim=1, keepdim=True)

            # reconstruction loss 
            reprojection_losses = []
            for f_id in f_ids:
                reprojection_losses.append(self.compute_reprojection_loss(warped_images['rgb', f_id], target))
            reprojection_loss = torch.cat(reprojection_losses, 1)

            masks = []
            for f_id in f_ids:
                masks.append(warped_images['pixel_valid_mask', f_id])
            mask = torch.cat(masks, 1).prod(1,True)
            
            min_reprojection_loss, _ = torch.min(reprojection_loss, dim=1, keepdim=True)

            stacked_losses = torch.cat([min_reprojection_loss, identity_reprojection_loss], dim=1)
            auto_mask = (torch.argmin(stacked_losses, dim=1, keepdim=True) == 0).float().detach() * mask.detach()
            losses['reconstruct_loss', scale] = (min_reprojection_loss * auto_mask).sum() / (auto_mask.sum() + 1e-7) * \
                                                self.cfgs["rec_loss_coef"] / \
                                                self.train_scale

        return losses
    
    def warp_image(self, rec_target, rec_index, rec_depth, outputs, f_ids):
        device = rec_target.device
        B, _, H, W = rec_depth.shape

        depth = rec_depth[:, 1:2]
        coords = torch.cat([coords_grid(B, H, W), torch.ones([B, 1, H, W])], 1).to(device)
        warped_rgb = {}
        for f_id in f_ids:
            if f_id < 0:
                rgb, t_depth = rec_target[:, rec_index[0]], rec_depth[:, 0:1]
                rot_matrix = to_matrix(outputs['rotv', f_id, 0], outputs['trans', f_id, 0], True)
            else:
                rgb, t_depth = rec_target[:, rec_index[2]], rec_depth[:, 2:3]
                rot_matrix = to_matrix(outputs['rotv', 0, f_id], outputs['trans', 0, f_id], False)

            coords_proj = reproject(
                coords,
                depth, 
                rot_matrix,
                self.K.view(1, 3, 3).expand([B, 3, 3]).to(device), 
                self.K_inv.view(1, 3, 3).expand([B, 3, 3]).to(device),
            )
            warped_rgb['rgb', f_id] = grid_sampler(rgb, coords_proj)
            warped_rgb['out_boundary_mask', f_id] = out_boundary_mask(coords_proj)
            warped_rgb['pixel_valid_mask', f_id] = (depth > self.cfgs["min_depth"]).float() * (t_depth > self.cfgs["min_depth"]).float()
        return warped_rgb
    
    def compute_error(self, inputs, preprocessed_inputs, outputs):
        prev_gps = inputs['gps'][:, 0:3].reshape(-1, 3, 1).to("cpu")
        curr_gps = inputs['gps'][:, 3:6].reshape(-1, 3, 1).to("cpu")
        prev_state = inputs["state"][:, 0:4, 0:4].to("cpu")
        curr_state = inputs["state"][:, 4:8, 0:4].to("cpu")
        goal = inputs["goal"][:, 3:6].to("cpu")

        # gt_mat[:, 2, 3] *= -1
        # gt_mat[:, :3, :3] = gt_mat[:, :3, :3].inverse()
        # gt_mat[:, 2, 3] *= -1
        # gt_mat[:, :3, :3] = gt_mat[:, :3, :3].inverse()

        s_id = self.loop_ids[0]
        key_rotv = ("rotv", s_id, s_id+1)
        key_trans = ("trans", s_id, s_id+1)
        if key_rotv not in outputs.keys():
            pose_inputs, indexs = self.warped_inputs(preprocessed_inputs, [s_id, s_id+1])
            if self.cfgs["act_spec"]:
                warped_actions = inputs["action"][:, indexs["depth"][1]]
            else:
                warped_actions = None
            outputs[key_rotv], outputs[key_trans] = self.infer_pose(pose_inputs, warped_actions)

        pred_pose = to_matrix(outputs[key_rotv], outputs[key_trans], invert=False).to("cpu")

        pred_state = torch.matmul(prev_state, pred_pose)
        pred_gps = pred_state[:, :3, :3].inverse().matmul((goal - pred_state[:, :3, 3])[..., np.newaxis])

        state_error = pred_state.inverse().matmul(curr_state)
        gps_error = pred_gps - curr_gps

        errors = {}
        errors['gps_error'] = translation_error(gps_error.view(-1, 3)).mean()
        # errors['rot_error'] = rotation_error(state_error).mean()
        # errors['trans_error'] = translation_error(state_error[:, :3, 3]).mean()
        # errors['depth_error'] = depth_error.mean()
        return errors

    def compute_reprojection_loss(self, pred, target):
        pred_rgb = pred[:, :3, ...]
        target_rgb = target[:, :3, ...]
        l1_loss_rgb = robust_l1(pred_rgb, target_rgb).mean(1, True)
        ssim_loss_rgb = ssim(pred_rgb, target_rgb).mean(1, True)
        reprojection_loss = (0.15 * l1_loss_rgb + 0.85 * ssim_loss_rgb)

        if pred.shape[1] > 3: #and self.training_state["epoch"] > 2:
            reprojection_loss = reprojection_loss + 1.0 * robust_l1(pred[:, 3:, ...], target[:, 3:, ...]).mean(1, True)

        # elif pred.shape[1] > 4:
        #     reprojection_loss = reprojection_loss + depth_l1(pred[:, 3:4, ...], target[:, 3:4, ...]).mean(1, True)
        #     reprojection_loss = reprojection_loss + robust_l1(pred[:, 4:, ...], target[:, 4:, ...]).mean(1, True)

        return reprojection_loss

    '''
    ######################################################################################################
    RL infer
    ######################################################################################################
    '''
    def infer_rl_batch(self, observations, prev_observations, prev_action, return_pose=False):
        rgb_inputs = torch.cat([prev_observations["rgb"],
                                observations["rgb"]],
                                dim=3).permute(0,3,1,2).float() / 255.0
        rgb_inputs = (bilinear_interp(rgb_inputs, self.pose_size) - 0.45) / 0.225
        if self.cfgs["pose_inputs"] == "rgb":
            posenet_inputs = rgb_inputs

        elif self.cfgs["pose_inputs"] == "rgbd":
            depth_inputs = torch.cat([prev_observations["depth"],
                                    observations["depth"]],
                                    dim=3).permute(0,3,1,2).float()
            depth_inputs = bilinear_interp(depth_inputs, self.pose_size)
            posenet_inputs = torch.cat([rgb_inputs, depth_inputs], 1)

        elif self.cfgs["pose_inputs"] == "rgbd_d":
            depth_inputs = torch.cat([prev_observations["depth"],
                                    observations["depth"]],
                                    dim=3).permute(0,3,1,2).float()
            depth_inputs = bilinear_interp(depth_inputs, self.pose_size)
            depth_d_inputs = depth_discretization(depth_inputs, 1, self.cfgs["depth_d_planes"])
            posenet_inputs = torch.cat([rgb_inputs, depth_d_inputs], 1)
        else:
            raise NotImplementedError("pose_inputs must be (rgb, rgbd, rgbd_d)")
        
        if self.cfgs["act_spec"]:
            rotv, trans = self.infer_pose(posenet_inputs, prev_action)
        else:
            rotv, trans = self.infer_pose(posenet_inputs, None)
        
        # to gps
        pose = to_matrix(rotv, trans, True)
        pose[:, 2, 3] *= -1
        pose[:,:3, :3] = pose[:, :3, :3].inverse()
        prev_gps = prev_observations["pointgoal_with_gps_compass"].float()
        pred_gps = pose[:,:3, :3].inverse().matmul((prev_gps - pose[:,:3, 3])[..., np.newaxis]).view(-1, 3)

        if return_pose:
            return pred_gps, to_matrix(rotv, trans, True)
        else:
            return pred_gps

    def infer_rl_ones(self, observations, prev_observations, env_id, prev_action):
        rgb_inputs = torch.cat([prev_observations["rgb"][env_id],
                                observations["rgb"][env_id]],
                                dim=2).permute(2,0,1)[np.newaxis, ...].float()
        rgb_inputs = (bilinear_interp(rgb_inputs, self.pose_size) - 0.45) / 0.225
        if self.cfgs["pose_inputs"] == "rgb":
            posenet_inputs = rgb_inputs

        elif self.cfgs["pose_inputs"] == "rgbd":
            depth_inputs = torch.cat([prev_observations["depth"][env_id],
                                    observations["depth"][env_id]],
                                    dim=2).permute(2,0,1)[np.newaxis, ...].float()
            depth_inputs = bilinear_interp(depth_inputs, self.pose_size)
            posenet_inputs = torch.cat([rgb_inputs, depth_inputs], 1)

        elif self.cfgs["pose_inputs"] == "rgbd_d":
            depth_inputs = torch.cat([prev_observations["depth"][env_id],
                                    observations["depth"][env_id]],
                                    dim=2).permute(2,0,1)[np.newaxis, ...].float()
            depth_inputs = bilinear_interp(depth_inputs, self.pose_size)
            depth_d_inputs = depth_discretization(depth_inputs, 1, self.cfgs["depth_d_planes"])
            posenet_inputs = torch.cat([rgb_inputs, depth_d_inputs], 1)
        else:
            raise NotImplementedError("pose_inputs must be (rgb, rgbd, rgbd_d)")
        
        if self.cfgs["act_spec"]:
            rotv, trans = self.infer_pose(posenet_inputs, prev_action)
        else:
            rotv, trans = self.infer_pose(posenet_inputs, None)
        
        # to gps
        pose = to_matrix(rotv, trans, True)[0]
        pose[2, 3] *= -1
        pose[:3, :3] = pose[:3, :3].inverse()
        prev_gps = prev_observations["pointgoal_with_gps_compass"][env_id].float()
        pred_gps = pose[:3, :3].inverse().matmul((prev_gps - pose[:3, 3]).view(3, 1)).view(3)
        return pred_gps

    def rl_infer(self, observations, not_done_masks, prev_observations):
        outputs = {}
        if self.training:
            outputs.update(self.rl_compute_loss(observations, not_done_masks))
            
        else:
            outputs.update(self.infer_gpsCompass(
                observations,
                prev_observations, 
                not_done_masks
            ))
            rgb = observations["rgb"].permute(0, 3, 1, 2).contiguous() / 255.
            outputs.update( self.infer_disp(rgb, 0) )

        return outputs
    
    def rl_compute_loss(self, observations, masks):
        batchs_rgb = observations["rgb"].permute(0, 3, 1, 2).contiguous() / 255. # NE*NS 3 H W

        if batchs_rgb.shape[2:4] != self.rgb_size:
            batchs_rgb = bilinear_interp(batchs_rgb, self.rgb_size)

        out_infos = {}
        out_infos['polar2d'] = cartesian2polar(observations["pointgoal_with_gps_compass"])
        out_infos.update( self.infer_disp(batchs_rgb, frame_id=0) )

        if self.rl_cfgs.VO.training_with_rl:
            batchs_rgb_3fold = torch.cat([batchs_rgb[0:-2], batchs_rgb[1:-1], batchs_rgb[2:]], 1) # B-2 9 H W
            batchs_mask_3fold = torch.cat([masks[0:-2], masks[1:-1], masks[2:]], 1) # B-2 1
            batchs_mask = batchs_mask_3fold.float().prod(1, True).bool()

            batchs_masked_rgb = batchs_rgb_3fold[batchs_mask.view(-1)]
            B, _, H, W = batchs_masked_rgb.shape

            outputs = {}
            # predict depth
            outputs.update( self.infer_disp(batchs_masked_rgb[:, 3:6], frame_id=0) )

            # predict poses
            if batchs_rgb.shape[2:4] != self.pose_size:
                pose_inputs = bilinear_interp(batchs_masked_rgb, self.pose_size)
            else:
                pose_inputs = batchs_masked_rgb
            outputs.update( self.infer_pose(pose_inputs[:, 0:6], frame_id=-1) )
            outputs.update( self.infer_pose(pose_inputs[:, 3:9], frame_id=1) )

            # compute losses
            losses = self.compute_loss(batchs_masked_rgb, outputs)
            for k, v in losses.items():
                key, s = k
                out_infos[key] = v if key not in out_infos else out_infos[key] + v
            out_infos["total_loss"] = sum(_value for _key, _value in losses.items())
        
        # if self.rl_cfgs.VO.extra_error_print:
        # errors = self.extra_info(outputs, observations["pointgoal_with_gps_compass"], batchs_mask)
        # losses_pack.update(errors)

        return out_infos
    
    def extra_info(self, outputs, gps, batchs_mask):
        # compute error for eval (not for training)
        batchs_gps_3fold = torch.cat([gps[0:-2], gps[1:-1], gps[2:]], 1) # B-2 6
        batchs_masked_gps = batchs_gps_3fold[batchs_mask.view(-1)]

        # max_depth = self.rl_cfgs.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        # min_depth = self.rl_cfgs.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
        # batchs_depth = (depth.permute(0, 3, 1, 2).contiguous() * max_depth).clamp(min=min_depth) # NE*NS 3 H W
        # batchs_depth_3fold = torch.cat([batchs_depth[0:-2], batchs_depth[1:-1], batchs_depth[2:]], 1) # B-2 6 H W
        # batchs_masked_depth = batchs_depth_3fold[batchs_mask.view(-1)]

        prev_gps = batchs_masked_gps[:, 0:3].to("cpu")
        curr_gps = batchs_masked_gps[:, 3:6].to("cpu")

        pred_pose = to_matrix(outputs['rotv', -1], outputs['trans', -1], invert=False).to("cpu")

        pred_gps = pred_pose[:, :3, :3].inverse().matmul((prev_gps - pred_pose[:, :3, 3]).view(-1, 3, 1))
        # gps_error = pred_gps.view(-1, 3) - curr_gps

        # disp = bilinear_interp(outputs['disp', 0, 0], [self.rgb_height, self.rgb_width])
        # _, depth = disp_to_depth(disp, self.max_depth, self.min_depth)
        # depth_error = (depth - batchs_masked_depth[:, 1:2]).abs().to("cpu")

        info = {}
        info['cartesian3d'] = pred_gps.view(-1, 3)
        return info
    
    def infer_gpsCompass(self, observations, prev_observations, not_done_masks):
        outputs = {}
        curr_gps = observations["pointgoal_with_gps_compass"]
        if prev_observations is None:
            pred_gps = curr_gps.clone()
        else:
            pose_inputs = torch.cat([prev_observations["rgb"], observations["rgb"]], 3)
            pose_inputs = pose_inputs.permute(0, 3, 1, 2).contiguous() / 255.
            pred_pose = self.infer_pose(pose_inputs, frame_id=-1, rot_matrix=True)["pose", -1]

            prev_gps = prev_observations["pointgoal_with_gps_compass"]
            pred_gps = pred_pose[:, :3, :3].inverse().matmul((prev_gps - pred_pose[:, :3, 3]).view(-1, 3, 1))

            pred_gps = torch.where(not_done_masks.expand(-1, 3).view(-1, 3, 1), pred_gps, curr_gps.view(-1, 3, 1))

        outputs['cartesian3d'] = pred_gps.view(-1, 3)
        outputs['polar2d'] = cartesian2polar(outputs['cartesian3d'])
        outputs['trans_error'] = translation_error((outputs['cartesian3d'] - curr_gps)).view(-1, 1)
        return outputs
    
    def reuse_from(self, log_path, reuse_id):
        reuse_path = os.path.join(log_path, "checkpoints")
        if reuse_id < 0:
            reuse_files = glob.glob(os.path.join(reuse_path, '*.pth'))
            reuse_ids = sorted([int(filename.split('.')[-2]) for filename in reuse_files])
            reuse_id = reuse_ids[-1]

        reuse_file = os.path.join(reuse_path, 'ckpt.%d.pth' % (reuse_id))
        if not os.path.exists(reuse_file):
            raise Exception("not exit %s !" % (reuse_file))

        params = torch.load(reuse_file, map_location="cpu")
        self.load_state_dict(params["model_params"])
        print('Load params from %s.' % (reuse_file))