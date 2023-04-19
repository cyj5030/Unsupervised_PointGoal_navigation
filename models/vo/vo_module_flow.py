import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections

import os
import glob

from models.networks.pose_network import PoseEncoder, PoseDecoder
from models.networks.depth_network import DepthEncoder, DepthDecoder
from models.networks.flow_network import FlowEncoder, FlowDecoder, coords_grid_v2, grid_sampler_v2
from models.vo.vo_utils import *


move_forward = torch.tensor([0., 0., -0.25])
turn_left = 10 * np.pi / 180. * torch.tensor([0, 1, 0])
turn_right = -10 * np.pi / 180. * torch.tensor([0, 1, 0])

ACTION_POSE_MAP = {
    1: move_forward,
    2: turn_left,
    3: turn_right,
}

class FlowModel(nn.Module):
    def __init__(self, cfgs):
        super(FlowModel, self).__init__()
        self.cfgs = cfgs

        self.H, self.W = cfgs['rgb_height'], cfgs['rgb_width']
        self.train_scale = cfgs['train_scale']

        self.encoder = FlowEncoder()
        self.decoder = FlowDecoder()

        # intrinsic
        hfov = float(cfgs["hfov"]) * np.pi / 180.
        self.K = torch.from_numpy(np.array([
            [1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., 1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]
        ]))[:3, :3].float()
        self.K_inv = self.K.inverse()

    def forward(self, inputs):
        outputs = {}

        # feature extract
        src_features = self.encoder(inputs['rgb'][:, 0:3])
        tgt_features = self.encoder(inputs['rgb'][:, 3:6])

        outputs.update(self.decoder(src_features, tgt_features, 0))
        outputs.update(self.decoder(tgt_features, src_features, 1))

        losses = {}
        losses.update(self.compute_losses(inputs, outputs))

        losses_pack = collections.OrderedDict()
        for k, v in losses.items():
            key, s = k
            losses_pack[key] = v if key not in losses_pack else losses_pack[key] + v
        total_loss = sum(_value for _key, _value in losses.items())

        losses_pack.update(self.compute_runtime_error(inputs, outputs))
        return losses_pack, total_loss
    
    def infer_flow(self, inputs):
        src_features = self.encoder(inputs['rgb'][:, 0:3])
        tgt_features = self.encoder(inputs['rgb'][:, 0:3])
        flows = self.decoder(src_features, tgt_features, 0)

        _, _, src_H, src_W = flows['flow', 0, 0].shape
        _, _, tgt_H, tgt_W = inputs['rgb'].shape

        flow = bilinear_interp(flows['flow', 0, 0], [tgt_H, tgt_W]) * (tgt_H/src_H)
        return flow

    def compute_losses(self, inputs, outputs):
        losses = {}
        for scale in range(self.train_scale):
            tgt_H = self.H // (2**scale)
            tgt_W = self.W // (2**scale)

            image0 = F.interpolate(inputs['rgb'][:, 0:3], [tgt_H, tgt_W], mode='area')
            image1 = F.interpolate(inputs['rgb'][:, 3:6], [tgt_H, tgt_W], mode='area')
            coords = coords_grid_v2(image0.shape[0], tgt_H, tgt_W).to(image0.device)

            _, _, src_H, src_W = outputs['flow', 0, scale].shape
            flow = bilinear_interp(outputs['flow', 0, scale], [tgt_H, tgt_W]) * (tgt_H/src_H)
            flow_inv = bilinear_interp(outputs['flow', 1, scale], [tgt_H, tgt_W]) * (tgt_H/src_H)

            consistent_mask_fw, consistent_mask_bw = self.consistent(flow, flow_inv, coords, 0.01, 0.5)
            outgoing_mask_fw = self.outgoing_mask((coords + flow), tgt_H, tgt_W)
            outgoing_mask_bw = self.outgoing_mask((coords + flow_inv), tgt_H, tgt_W)
            masks_fw = self.object_mask(consistent_mask_fw, outgoing_mask_fw)
            masks_bw = self.object_mask(consistent_mask_bw, outgoing_mask_bw)
            
            warped_image0 = grid_sampler_v2(image1, (coords + flow), padding='border') # border
            warped_image1 = grid_sampler_v2(image0, (coords + flow_inv), padding='border')
            photometric_loss0 = self.compute_photometric_loss(image0, warped_image0, masks_fw)
            photometric_loss1 = self.compute_photometric_loss(image1, warped_image1, masks_bw)
            losses['photometric_loss', scale]  = 1.0 * (photometric_loss0 + photometric_loss1) / (2**scale)
            
        smooth1, smooth2 = self.smooth_loss(outputs['flow', 0, 0], inputs['rgb'][:, 0:3], 1.0, 10.0)
        smooth_inv1, smooth_inv2 = self.smooth_loss(outputs['flow', 1, 0], inputs['rgb'][:, 3:6], 1.0, 10.0)
        losses['smooth_loss', 0] = 0.05 * (smooth1 + smooth_inv1)

        return losses

    def compute_runtime_error(self, inputs, outputs):
        prev_state = inputs["state"][:, 0:4, 0:4].to("cpu")
        curr_state = inputs["state"][:, 4:8, 0:4].to("cpu")

        depth = inputs["depth"][:, 0:1].to("cpu") * 10.0
        rgb = inputs["rgb"][:, 3:6].to("cpu")
        B,_,H,W = rgb.shape
        coords = torch.cat([coords_grid(B, H, W), torch.ones([B, 1, H, W])], 1).to("cpu")

        rot_matrix = curr_state.inverse().matmul(prev_state)

        coords_proj = reproject(
            coords,
            depth, 
            rot_matrix,
            self.K.view(1, 3, 3).expand([B, 3, 3]).to("cpu"), 
            self.K_inv.view(1, 3, 3).expand([B, 3, 3]).to("cpu"),
        )
        flow_gt = (coords[:, :2, :, :] - coords_proj)
        flow_gt[:, 0:1] = (flow_gt[:, 0:1] + 1) / 2. * (W-1)
        flow_gt[:, 1:2] = (flow_gt[:, 1:2] + 1) / 2. * (H-1)

        _, _, src_H, src_W = outputs['flow', 0, 0].shape
        flow = bilinear_interp(outputs['flow', 0, 0].to("cpu"), [H, W]) * (H/src_H)
        
        errors = {}
        errors["flow_error"] = (flow-flow_gt).abs().mean()
        return errors

    def consistent(self, flow_fw, flow_bw, coords, a1=0.1, a2=0.5):
        def sum_(x):
            return torch.sum(x.pow(2.0), dim=1, keepdim=True)
            # return torch.sum(torch.pow(x ** 2, 0.5), dim=1, keepdim=True)

        flow_fw_warped = grid_sampler(flow_bw, (coords + flow_fw))
        flow_bw_warped = grid_sampler(flow_fw, (coords + flow_bw))
        flow_diff_fw = flow_fw + flow_fw_warped
        flow_diff_bw = flow_bw + flow_bw_warped

        # thresh = a1 * (sum_(flow_fw) + sum_(flow_bw)) + a2

        thresh_fw = torch.max( torch.tensor([3.0], device=flow_bw.get_device()), 
                            a1 * (sum_(flow_fw) + sum_(flow_fw_warped)) + a2)
        thresh_bw = torch.max( torch.tensor([3.0], device=flow_bw.get_device()), 
                            a1 * (sum_(flow_bw) + sum_(flow_bw_warped)) + a2)

        # thresh_fw = a1 * (sum_(flow_fw) + sum_(flow_fw_warped)) + a2
        # thresh_bw = a1 * (sum_(flow_bw) + sum_(flow_bw_warped)) + a2

        # if epoch < 10:
        #     thresh_fw = torch.max( torch.tensor([3.0], device=flow_bw.get_device()), 
        #                         a1 * (sum_(flow_fw) + sum_(flow_fw_warped)) + a2)
        #     thresh_bw = torch.max( torch.tensor([3.0], device=flow_bw.get_device()), 
        #                         a1 * (sum_(flow_bw) + sum_(flow_bw_warped)) + a2)
        # else:
        #     thresh_fw = a1 * (sum_(flow_fw) + sum_(flow_fw_warped)) + a2
        #     thresh_bw = a1 * (sum_(flow_bw) + sum_(flow_bw_warped)) + a2

        with torch.no_grad():
            mask_fw = (sum_(flow_diff_fw) < thresh_fw).float()
            mask_bw = (sum_(flow_diff_bw) < thresh_bw).float()
        return mask_fw, mask_bw
    
    def outgoing_mask(self, coords, H, W):
        pos_x, pos_y = coords.split([1, 1], dim=1)

        with torch.no_grad():
            outgoing_mask = torch.ones_like(pos_x)
            outgoing_mask[pos_x > W - 1] = 0
            outgoing_mask[pos_x < 0] = 0
            outgoing_mask[pos_y > H - 1] = 0
            outgoing_mask[pos_y < 0] = 0
        return outgoing_mask.float()
    
    def object_mask(self, occ_mask, out_mask):
        obj_mask = torch.zeros_like(occ_mask)
        with torch.no_grad():
            obj_mask[occ_mask == 1] = 1
            obj_mask[out_mask == 0] = 1
        return obj_mask

    def compute_photometric_loss(self, pred, target, mask):
        l1_loss = robust_l1(pred, target)
        l1_loss = (l1_loss * mask).sum() / (mask.sum() + 1e-7)

        # ssim_loss = ssim(pred, target)
        # ssim_loss = (ssim_loss * mask).sum() / (mask.sum() + 1e-7)

        census_loss, trans_mask = census(pred, target, 0.4)
        census_loss = (census_loss * mask * trans_mask).sum() / ((mask * trans_mask).sum() + 1e-7)
        
        photometric_loss = (0.5*census_loss + l1_loss).sum()
        return photometric_loss

    def smooth_loss(self, tgt_tensor, img, a1, a2):

        def gradient(ten):
            dy = ten[:,:,1:,:] - ten[:,:,:-1,:]
            dx = ten[:,:,:,1:] - ten[:,:,:,:-1]
            return dx, dy

        b, _, h, w = tgt_tensor.size()

        img = F.interpolate(img, (h, w), mode='area')

        dx, dy = gradient(tgt_tensor)
        img_dx, img_dy = gradient(img)

        dxx = dx[:,:,:,1:] - dx[:,:,:,:-1]
        dyy = dy[:,:,1:,:] - dy[:,:,:-1,:]

        img_dxx, img_dyy = img_dx[:,:,:,1:], img_dy[:,:,1:,:]

        smooth1 = torch.mean(dx.abs() * torch.exp(-a1 * img_dx.abs().mean(1, True))) + \
                torch.mean(dy.abs() * torch.exp(-a1 * img_dy.abs().mean(1, True)))

        smooth2 = torch.mean(dxx.abs() * torch.exp(-a2 * img_dxx.abs().mean(1, True))) + \
                torch.mean(dyy.abs() * torch.exp(-a2 * img_dyy.abs().mean(1, True)))
        return smooth1, smooth2