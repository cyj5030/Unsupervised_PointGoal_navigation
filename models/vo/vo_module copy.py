import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections

import os
import glob

from models.networks.pose_network import PoseEncoder, PoseDecoder
from models.networks.depth_network import DepthEncoder, DepthDecoder
from models.vo.vo_utils import *


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

        # pose net
        self.pose_width = cfgs["pose_width"]
        self.pose_height = cfgs["pose_height"]
        self.pose_size = [cfgs["pose_height"], cfgs["pose_width"]]
        dof = cfgs["dof"]
        rot_coef = cfgs["rot_coef"]
        trans_coef = cfgs["trans_coef"]
        self.pose_encoder = PoseEncoder(num_layers=18)
        self.pose_decoder = PoseDecoder(self.pose_encoder.num_ch_enc, dof, rot_coef, trans_coef)

        # depth net
        self.rgb_width = cfgs["rgb_width"]
        self.rgb_height = cfgs["rgb_height"]
        self.min_depth = cfgs["min_depth"]
        self.max_depth = cfgs["max_depth"]
        self.rgb_size = [cfgs["rgb_height"], cfgs["rgb_width"]]
        self.depth_encoder = DepthEncoder(18, num_inputs=1)
        self.depth_decoder = DepthDecoder(self.depth_encoder.num_ch_enc)

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
    
    def forward(self, inputs):
        outputs = {}

        # predict depth
        # outputs.update( self.infer_disp(inputs["rgb"][:, 3:6], frame_id=0) )

        # predict poses
        pose_inputs = bilinear_interp(inputs['rgb'], [self.pose_height, self.pose_width])
        outputs['rotv', -1], outputs['trans', -1] = self.infer_pose(pose_inputs[:, 0:6])
        outputs['rotv', 1], outputs['trans', 1] = self.infer_pose(pose_inputs[:, 3:9])

        # compute losses
        losses_pack = collections.OrderedDict()
        losses = self.compute_loss(inputs, outputs)

        # collection
        for k, v in losses.items():
            key, s = k
            losses_pack[key] = v if key not in losses_pack else losses_pack[key] + v
        total_loss = sum(_value for _key, _value in losses.items())
        
        # compute error for eval (not for training)
        errors = self.compute_error(inputs, outputs)
        losses_pack.update(errors)

        return losses_pack, total_loss

    def eval_infer(self, inputs):
        outputs = {}
        outputs.update( self.infer_disp(inputs["rgb"][:, 3:6], frame_id=0, as_depth=True) )
        outputs.update(self.infer_pose(inputs["rgb"], frame_id=-1, rot_matrix=True))
        outputs.update(self.compute_error(inputs, outputs))
        return outputs
        
    def infer_pose(self, cnn_inputs):
        rotv, trans = self.pose_decoder(self.pose_encoder(cnn_inputs))
        return rotv, trans
    
    def infer_disp(self, cnn_inputs, frame_id):
        outputs = self.depth_decoder(self.depth_encoder(cnn_inputs), frame_id)
        return outputs

    def action2pose(self, action):
        poses = torch.stack([ACTION_POSE_MAP[aid.item()] for aid in action], 0).to(action.device)
        return poses

    def compute_loss(self, inputs, outputs):
        B, _, H, W = inputs["rgb"].shape
        target = inputs["rgb"][:, 3:6]

        losses = {}
        for scale in range(self.train_scale):
            # reconstruction
            warped_images = self.warp_image(inputs, outputs, scale)

            # automask
            identity_reprojection_losses = []
            for f_id in [-1, 1]:
                islice = slice(0, 3) if f_id < 0 else slice(6, 9)
                identity_reprojection_loss = self.compute_reprojection_loss(inputs["rgb"][:, islice], target)
                identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).to(target.device) * 1e-5
                identity_reprojection_losses.append(identity_reprojection_loss)
            identity_reprojection_loss, _ = torch.min(torch.cat(identity_reprojection_losses, 1), dim=1, keepdim=True)

            # reconstruction loss 
            reprojection_losses = []
            for f_id in [-1, 1]:
                reprojection_losses.append(self.compute_reprojection_loss(warped_images['rgb', f_id], target))

            reprojection_loss = torch.cat(reprojection_losses, 1)
            min_reprojection_loss, _ = torch.min(reprojection_loss, dim=1, keepdim=True)
            # uncert = bilinear_interp(outputs['uncert', 0, scale], [self.rgb_height, self.rgb_width])
            # uncert_reprojection_loss = min_reprojection_loss / uncert + torch.log(uncert)

            stacked_losses = torch.cat([min_reprojection_loss, identity_reprojection_loss], dim=1)
            auto_mask = (torch.argmin(stacked_losses, dim=1, keepdim=True) == 0).float().detach()
            losses['reconstruct_loss', scale] = self.cfgs["rec_loss_coef"] * \
                                        (min_reprojection_loss * auto_mask).sum() / (auto_mask.sum() + 1e-7) / self.train_scale

            # smooth loss
            # disp = outputs['disp', 0, scale]
            # mean_disp = disp.mean(2, True).mean(3, True)
            # disp = disp / (mean_disp + 1e-7)
            # smooth_loss = self.smooth_loss(disp, target, 1.0)
            # losses['smooth_loss', scale] = 1e-3 * smooth_loss / (2 ** scale) / self.train_scale

        # pose action loss
        # for f_id in [-1, 1]:
        #     a_id = 1 if f_id < 0 else 2
        #     action_pose = self.action2pose(action[:, a_id])
        #     move_mask = (action[:, a_id] == 1).float().detach().view(-1, 1)
        #     ts_loss = (((outputs['trans', f_id] - action_pose) * move_mask).sum() / (move_mask.sum() + 1e-7)) + \
        #                 (((outputs['trans', f_id] - torch.zeros_like(action_pose)) * (1-move_mask)).sum() / ((1-move_mask).sum() + 1e-7))
        #     losses["ts_loss", f_id] = 0.1 * ts_loss

        #     rs_loss = (((outputs['trans', f_id] - action_pose) * move_mask).sum() / (move_mask.sum() + 1e-7)) + \
        #                 (((outputs['trans', f_id] - torch.zeros_like(action_pose)) * (1-move_mask)).sum() / ((1-move_mask).sum() + 1e-7))
        #     losses["rs_loss", f_id] = 0.1 * ts_loss

        return losses
    
    def warp_image(self, inputs, outputs, scale):
        B, _, H, W = inputs["rgb"].shape
        device = inputs["rgb"].device

        coords = torch.cat([coords_grid(B, H, W), torch.ones([B, 1, H, W])], 1).to(device)

        # disp = bilinear_interp(outputs['disp', 0, scale], [self.rgb_height, self.rgb_width])
        # _, depth = disp_to_depth(disp, self.max_depth, self.min_depth)

        warped_rgb = {}
        for f_id in [-1, 1]:
            if f_id < 0:
                rgb, invert = inputs["rgb"][:, 0:3], True
            else:
                rgb, invert = inputs["rgb"][:, 6:9], False

            rot_matrix = to_matrix(outputs['rotv', f_id], outputs['trans', f_id], invert)

            coords_proj = reproject(
                coords,
                inputs["depth"][:, 1:2], 
                rot_matrix,
                self.K.view(1, 3, 3).expand([B, 3, 3]).to(device), 
                self.K_inv.view(1, 3, 3).expand([B, 3, 3]).to(device),
            )
            warped_rgb['rgb', f_id] = grid_sampler(rgb, coords_proj)
            warped_rgb['out_boundary_mask', f_id] = out_boundary_mask(coords_proj)
            # warped_rgb['pixel_valid_mask', f_id] = (rgb > 0.0).float().prod()
        return warped_rgb
    
    def compute_error(self, inputs, outputs):
        prev_gps = inputs['gps'][:, 0:3].reshape(-1, 3, 1).to("cpu")
        curr_gps = inputs['gps'][:, 3:6].reshape(-1, 3, 1).to("cpu")
        prev_state = inputs["state"][:, 0:4, 0:4].to("cpu")
        curr_state = inputs["state"][:, 4:8, 0:4].to("cpu")
        goal = inputs["goal"][:, 3:6].to("cpu")

        pred_pose = to_matrix(outputs['rotv', -1], outputs['trans', -1], invert=False).to("cpu")

        pred_state = torch.matmul(prev_state, pred_pose)
        pred_gps = pred_state[:, :3, :3].inverse().matmul((goal - pred_state[:, :3, 3])[..., np.newaxis])

        state_error = pred_state.inverse().matmul(curr_state)
        gps_error = pred_gps - curr_gps

        # disp = bilinear_interp(outputs['disp', 0, 0], [self.rgb_height, self.rgb_width])
        # _, depth = disp_to_depth(disp, self.max_depth, self.min_depth)
        # depth_error = (depth - inputs['depth'][:, 1:2] * 10.).abs().to("cpu")

        errors = {}
        # errors['gps_error'] = translation_error(gps_error.view(-1, 3)).mean()
        errors['rot_error'] = rotation_error(state_error).mean()
        errors['trans_error'] = translation_error(state_error[:, :3, 3]).mean()
        # errors['depth_error'] = depth_error.mean()
        return errors

    def compute_reprojection_loss(self, pred, target, ssim_coef=0.85, l1_coef=0.15):
        l1_loss = robust_l1(pred, target).mean(1, True)
        ssim_loss = ssim(pred, target).mean(1, True)
        reprojection_loss = (ssim_coef * ssim_loss + l1_coef * l1_loss)
        return reprojection_loss
    
    def smooth_loss(self, target, img, alpha):
        def gradient(target):
            dy = target[:,:,1:,:] - target[:,:,:-1,:]
            dx = target[:,:,:,1:] - target[:,:,:,:-1]
            return dx, dy

        b, _, h, w = target.size()
        img = F.interpolate(img, (h, w), mode='area')
        dx, dy = gradient(target)
        img_dx, img_dy = gradient(img)

        smooth = torch.mean(dx.abs() * torch.exp(-alpha * img_dx.abs().mean(1, True))) + \
                torch.mean(dy.abs() * torch.exp(-alpha * img_dy.abs().mean(1, True)))
        return smooth

    '''
    ######################################################################################################
    RL infer
    ######################################################################################################
    '''
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
        if reuse_id < 0:
            reuse_path = os.path.join(log_path, "checkpoints")
            reuse_files = glob.glob(os.path.join(reuse_path, '*.pth'))
            reuse_ids = sorted([int(filename.split('.')[-2]) for filename in reuse_files])
            reuse_id = reuse_ids[-1]

        reuse_file = os.path.join(reuse_path, 'ckpt.%d.pth' % (reuse_id))
        if not os.path.exists(reuse_file):
            raise Exception("not exit %s !" % (reuse_file))

        params = torch.load(reuse_file, map_location="cpu")
        self.load_state_dict(params["model_params"])
        print('Load params from %s.' % (reuse_file))