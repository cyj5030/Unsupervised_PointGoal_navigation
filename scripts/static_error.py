import torch
import numpy as np
import copy
import matplotlib.pyplot as plt
import cv2
import os
import glob
from tqdm import tqdm
import torchvision

from scipy.stats import norm
from matplotlib.pyplot import Axes

import sys
sys.path.append(os.getcwd())
from models.vo.dataset import VODataset, VODataset_small_chunk
from models.vo.vo_utils import *
from models.vo.lie_algebra import SO3_CPU
# from models.vo.compared_methods.icp import icp

import quaternion as q
move_forward = torch.eye(4)
move_forward[2, 3] = -0.25

turn_left = torch.eye(4)
turn_left[:3, :3] = torch.from_numpy(q.as_rotation_matrix(q.from_rotation_vector(10 * np.pi / 180. * np.array([0, 1, 0])))).float()

turn_right= torch.eye(4)
turn_right[:3, :3] = torch.from_numpy(q.as_rotation_matrix(q.from_rotation_vector(-10 * np.pi / 180. * np.array([0, 1, 0])))).float()

action_piror = {
    1: move_forward,
    2: turn_left,
    3: turn_right,
}

def to_rMat(T_vec):
    vs = []
    for v in T_vec:
        rmat = np.eye(4)
        rmat[:3,:3] = q.as_rotation_matrix(q.from_rotation_vector(v[:3]))
        rmat[:3,3] = v[3:]
        vs.append(rmat)
    return np.stack(vs, 0)

def add_noise(T_cam, nry=np.linspace(-5, 5, 100), ntx=np.linspace(-0.25, 0.25, 100), ntz=np.linspace(-0.25, 0.25, 100)):
    rv_cams = []
    for iT in T_cam:
        rv_cams.append(q.as_rotation_vector(q.from_rotation_matrix(iT[:3,:3].numpy())))
    rv_cams = np.concatenate([np.stack(rv_cams, 0), T_cam.numpy()[:, :3, 3]], 1)

    noise_roty = []
    for iry in nry:
        nit = np.array([0,iry,0,0,0,0])
        noise_roty.append( to_rMat(rv_cams + nit[np.newaxis, ...]))
    noise_roty = torch.from_numpy(np.stack(noise_roty, 0)).float()

    noise_transx = []
    for itx in ntx:
        nit = np.array([0,0,0,itx,0,0])
        noise_transx.append( to_rMat(rv_cams + nit[np.newaxis, ...]))
    noise_transx = torch.from_numpy(np.stack(noise_transx, 0)).float()

    noise_transz = []
    for itz in ntz:
        nit = np.array([0,0,0,0,0,itz])
        noise_transz.append( to_rMat(rv_cams + nit[np.newaxis, ...]))
    noise_transz = torch.from_numpy(np.stack(noise_transz, 0)).float()
            
    return rv_cams, noise_roty, noise_transx, noise_transz

def static_reproject_error(inputs, rec_target_name="texture"):
    hfov = float(70) * np.pi / 180.
    K = torch.from_numpy(np.array([
        [1 / np.tan(hfov / 2.), 0., 0., 0.],
        [0., 1 / np.tan(hfov / 2.), 0., 0.],
        [0., 0.,  1, 0],
        [0., 0., 0, 1]
    ]))[:3, :3].float()

    K_inv = K.inverse()

    rec_target, tgt_index, tgt_depth = warped_reconstruction_target(inputs, rec_target=rec_target_name)

    poses = get_pose(inputs)
    losses, reprojection_losses = reprojection_loss(rec_target, tgt_index, tgt_depth, poses, K, K_inv)

    # torchvision.utils.save_image(inputs["rgb"], "/home/cyj/code/pointgoalnav_unsup_rgbd/train_log/debug/rgb.png")
    # torchvision.utils.save_image(inputs["rgb"], "/home/cyj/code/pointgoalnav_unsup_rgbd/train_log/debug/1.png")
    batch = inputs["rgb"].shape[0]
    root = "/home/cyj/code/pointgoalnav_unsup_rgbd/train_log/debug/rgbd"
    for i in range(batch):
        cv2.imwrite(os.path.join(root, f"rgb0_{i:0>2d}.png"), rec_target[i][0:3].permute(1,2,0).cpu().numpy()*255)
        cv2.imwrite(os.path.join(root, f"depth0_{i:0>2d}.png"), rec_target[i][3:4].permute(1,2,0).cpu().numpy() * 255)
        cv2.imwrite(os.path.join(root, f"edge0_{i:0>2d}.png"), rec_target[i][4:7].permute(1,2,0).cpu().numpy() * 255)

        cv2.imwrite(os.path.join(root, f"rgb1_{i:0>2d}.png"), rec_target[i][7:10].permute(1,2,0).cpu().numpy()*255)
        cv2.imwrite(os.path.join(root, f"depth1_{i:0>2d}.png"), rec_target[i][10:11].permute(1,2,0).cpu().numpy() * 255)
        cv2.imwrite(os.path.join(root, f"edge1_{i:0>2d}.png"), rec_target[i][11:14].permute(1,2,0).cpu().numpy() * 255)

        cv2.imwrite(os.path.join(root, f"rgb2_{i:0>2d}.png"), rec_target[i][14:17].permute(1,2,0).cpu().numpy()*255)        
        cv2.imwrite(os.path.join(root, f"depth2_{i:0>2d}.png"), rec_target[i][17:18].permute(1,2,0).cpu().numpy() * 255)
        cv2.imwrite(os.path.join(root, f"edge2_{i:0>2d}.png"), rec_target[i][18:21].permute(1,2,0).cpu().numpy() * 255)

    pose0 = poses["pose", -1].clone()
    pose1 = poses["pose", 1].clone()
    nry=np.linspace(-1, 1, 101)
    ntx=np.linspace(-0.1, 0.1, 101)
    ntz=np.linspace(-0.1, 0.1, 101)
    _, nry_pose0, ntx_pose0, ntz_pose0 = add_noise(poses["pose", -1].cpu(), nry, ntx, ntz)
    _, nry_pose1, ntx_pose1, ntz_pose1 = add_noise(poses["pose", 1].cpu(), nry, ntx, ntz)
    
    n_noise = nry_pose0.shape[0]

    nry_losses = []
    for nid in range(n_noise):
        poses["pose", -1] = nry_pose0[nid]
        poses["pose", 1] = nry_pose1[nid]
        nlosses, _ = reprojection_loss(rec_target, tgt_index, tgt_depth, poses, K, K_inv)
        nry_losses.append(nlosses)
    nry_losses = torch.tensor(nry_losses)
    diff_nry_losses = (nry_losses - losses.item()).numpy()

    ntx_losses = []
    for nid in range(n_noise):
        poses["pose", -1] = ntx_pose0[nid]
        poses["pose", 1] = ntx_pose1[nid]
        nlosses, _ = reprojection_loss(rec_target, tgt_index, tgt_depth, poses, K, K_inv)
        ntx_losses.append(nlosses)
    ntx_losses = torch.tensor(ntx_losses)
    diff_ntx_losses = (ntx_losses - losses.item()).numpy()

    ntz_losses = []
    for nid in range(n_noise):
        poses["pose", -1] = ntz_pose0[nid]
        poses["pose", 1] = ntz_pose1[nid]
        nlosses, _ = reprojection_loss(rec_target, tgt_index, tgt_depth, poses, K, K_inv)
        ntz_losses.append(nlosses)
    ntz_losses = torch.tensor(ntz_losses)
    diff_ntz_losses = (ntz_losses - losses.item()).numpy()
    return diff_nry_losses, diff_ntx_losses, diff_ntz_losses, nry,ntx,ntz

def reprojection_loss(rec_target, tgt_index, tgt_depth, poses, K, K_inv):
    target = rec_target[:, tgt_index[1]]

    # reconstruction
    warped_images = warp_image(rec_target, tgt_index, tgt_depth, poses, K, K_inv)

    # automask
    identity_reprojection_losses = []
    for f_id in [-1, 1]:
        islice = tgt_index[0] if f_id < 0 else tgt_index[2]
        identity_reprojection_loss = compute_reprojection_loss(rec_target[:, islice[:3]], target[:, :3])
        identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).to(target.device) * 1e-5
        identity_reprojection_losses.append(identity_reprojection_loss)
    identity_reprojection_loss, _ = torch.min(torch.cat(identity_reprojection_losses, 1), dim=1, keepdim=True)

    # reconstruction loss 
    reprojection_losses = []
    for f_id in [-1, 1]:
        reprojection_losses.append(compute_reprojection_loss(warped_images['rgb', f_id], target))

    masks = []
    for f_id in [-1,1]:
        masks.append(warped_images['pixel_valid_mask', f_id])
    mask = torch.cat(masks, 1).prod(1,True)

    reprojection_loss = torch.cat(reprojection_losses, 1)
    min_reprojection_loss, _ = torch.min(reprojection_loss, dim=1, keepdim=True)

    stacked_losses = torch.cat([min_reprojection_loss, identity_reprojection_loss], dim=1)
    auto_mask = (torch.argmin(stacked_losses, dim=1, keepdim=True) == 0).float().detach() * mask
    losses = (min_reprojection_loss * auto_mask).sum() / (auto_mask.sum() + 1e-7)
    # torchvision.utils.save_image(mask, "/home/cyj/code/pointgoalnav_unsup_rgbd/train_log/debug/automask.png")
    s = int(rec_target.shape[1]/3)
    # torchvision.utils.save_image(rec_target[:, s:s+3,:,:], "/home/cyj/code/pointgoalnav_unsup_rgbd/train_log/debug/rgb.png")

    # losses = reprojection_loss.sum(1).mean()

    return losses, reprojection_losses

def warp_image(rec_target, rec_index, rec_depth, poses, K, K_inv):
    device = rec_target.device
    B, _, H, W = rec_depth.shape

    depth = rec_depth[:, 1:2]
    coords = torch.cat([coords_grid(B, H, W), torch.ones([B, 1, H, W])], 1).to(device)
    warped_rgb = {}
    for f_id in [-1, 1]:
        if f_id < 0:
            rgb, t_depth = rec_target[:, rec_index[0]], rec_depth[:, 0:1]
            rot_matrix = poses["pose", -1].inverse().to(device)
        else:
            rgb, t_depth = rec_target[:, rec_index[2]], rec_depth[:, 2:3]
            rot_matrix = poses["pose", -1].to(device)

        coords_proj = reproject(
            coords,
            depth, 
            rot_matrix,
            K.view(1, 3, 3).expand([B, 3, 3]).to(device), 
            K_inv.view(1, 3, 3).expand([B, 3, 3]).to(device),
        )
        warped_rgb['rgb', f_id] = grid_sampler(rgb, coords_proj)
        warped_rgb['out_boundary_mask', f_id] = out_boundary_mask(coords_proj)
        warped_rgb['pixel_valid_mask', f_id] = (depth > 0.1).float() * (t_depth > 0.1).float()
    return warped_rgb

def get_pose(inputs):
    def map_(mat):
        out = mat.clone()
        out[:, :3, :3] = mat[:, :3, :3].inverse()
        out[:, 2, 3] = -1 * mat[:, 2, 3]
        return out
    outputs = {}
    states = inputs["state"].split([4, 4, 4], dim=1)
    states = list(map(map_, states))
    outputs["pose", 1] = states[2].inverse().matmul(states[1])
    outputs["pose", -1] = states[1].inverse().matmul(states[0])
    return outputs

def warped_reconstruction_target(inputs, rec_target="rgb", texture=False):
    rec_depth = inputs["depth"] * 10.0

    if rec_target == "rgb":
        rec_target = inputs["rgb"] 

    elif rec_target == "rgbd":
        rec_target = torch.cat([inputs["rgb"], inputs["depth"]], dim=1)
        rearrange_id = torch.cat([torch.arange(9).view(3, -1), 9+torch.arange(3).view(3, -1)], 1).reshape(-1)
        rec_target = rec_target[:, rearrange_id, ...]

    elif rec_target == "rgbd_d":
        planes = 10
        depth_d = depth_discretization(inputs["depth"], 1, planes)
        rec_target = torch.cat([inputs["rgb"], depth_d], dim=1)
        rearrange_id = torch.cat([torch.arange(9).view(3, -1), 9+torch.arange(3*planes).view(3, -1)], 1).reshape(-1)
        rec_target = rec_target[:, rearrange_id, ...]

    elif rec_target == "texture":
        texture = gabor(inputs["rgb"])
        planes = 10
        depth_d = depth_discretization(inputs["depth"], 1, planes)
        # torchvision.utils.save_image(texture[:, 3:6,:,:], "/home/cyj/code/pointgoalnav_unsup_rgbd/train_log/debug/gabor.png")
        rec_target = torch.cat([inputs["rgb"], inputs["depth"], texture], dim=1)
        rearrange_id = torch.cat([torch.arange(9).view(3, -1), 
                                # 9+torch.arange(3*planes).view(3, -1), 
                                9+torch.arange(3).view(3, -1), 
                                9+3+torch.arange(9).view(3, -1)], 1).reshape(-1)
        rec_target = rec_target[:, rearrange_id, ...]
    else:
        raise NotImplementedError("pose_inputs must be (rgb, rgbd)")

    rec_index = torch.arange(rec_target.shape[1]).view(3, -1)
    rec_target = rec_target

    return rec_target, rec_index, rec_depth

def gabor(inputs, sigma=1.):
    b, c, h, w = inputs.shape

    r = int(sigma*3)
    yy, xx = torch.meshgrid(torch.linspace(-r,r,r*2+1), torch.linspace(-r,r,r*2+1))
    yy = -1 * yy
    coeff = 1 / (np.sqrt(2*np.pi) * sigma**3)
    A = -xx * torch.exp(-(xx**2 + (0.25*yy**2)) / (2*sigma**2))
    kernel = (coeff*A).view(1, 1, *A.shape).to(inputs.device)

    responses = []
    for i in range(c):
        responses.append( F.conv2d(inputs[:, i:i+1, ...], kernel, padding=r).abs() )
    response = torch.cat(responses, 1).contiguous()
    # response = response / response.view(b,c,-1).max(-1,True)[0][...,np.newaxis]
    # response[response<response.mean()] = 0
    return response

def compute_reprojection_loss(pred, target):
    pred_rgb = pred[:, :3, ...]
    target_rgb = target[:, :3, ...]
    l1_loss_rgb = robust_l1(pred_rgb, target_rgb).mean(1, True)
    ssim_loss_rgb = ssim(pred_rgb, target_rgb).mean(1, True)
    reprojection_loss = (0.85 * l1_loss_rgb + 0.15 * ssim_loss_rgb)

    if pred.shape[1] == 4:
        reprojection_loss = reprojection_loss + depth_l1(pred[:, 3:4, ...], target[:, 3:4, ...]).mean(1, True)

    elif pred.shape[1] > 4:
        reprojection_loss = reprojection_loss + depth_l1(pred[:, 3:4, ...], target[:, 3:4, ...]).mean(1, True)
        reprojection_loss = reprojection_loss + (pred[:, 4:, ...] - target[:, 4:, ...]).abs().mean(1, True)

    return reprojection_loss

def depth_l1(x, y, min_ndepth=0.01):
    return robust_l1(-torch.log(x.clamp(min=min_ndepth)), -torch.log(y.clamp(min=min_ndepth))) / -np.log(0.01)

def hist_(x, data, num_bins, density, suffix, cfgs):
    fig, axes = plt.subplots()
    num_bins = 50
    mu, sigma = data.mean(), data.std()
    # n, bins, patches = axes.hist(data, num_bins, density=density)
    axes.plot(x, data)
    axes.set_title("%s: $\mu$=%.2f, $\sigma$=%.2f" % (suffix, mu, sigma))

    sv_path = os.path.join(cfgs["log_path"], cfgs["log_name"], 'figures')
    if not os.path.exists(sv_path):
        os.makedirs(sv_path)
        
    plt.rcParams['savefig.dpi'] = 300
    plt.savefig(os.path.join(sv_path, suffix + '.png'), bbox_inches="tight", pad_inches=0.0)
    plt.show()
    plt.close()

def evaluation(cfgs, run_type):
    out_size = (cfgs["rgb_height"], cfgs["rgb_width"])
    cfgs["num_frame"] = 3
    dataset = VODataset_small_chunk(cfgs["dataset_path"], run_type, cfgs["num_frame"], out_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=False)

    device = "cuda:0"
    collects = {
        "rgb": [],
        "rgbd": [],
        "rgbdd": [],
        "rgbdd_texture": [],
    }
    nry = []
    ntx = []
    ntz = []
    for i, data in tqdm(enumerate(dataloader), desc="data vis"):
        for key, itms in data.items():
            data[key] = itms.to(device)

        diff_nry_losses, diff_ntx_losses, diff_ntz_losses, xnry,xntx,xntz = static_reproject_error(data)
        nry.append(diff_nry_losses)
        ntx.append(diff_ntx_losses)
        ntz.append(diff_ntz_losses)
        if i >= 5:
            break
    mean_nry = np.stack(nry, 0).mean(0)
    mean_ntx = np.stack(ntx, 0).mean(0)
    mean_ntz = np.stack(ntz, 0).mean(0)

    num_bins = 100
    density = 0.001
    hist_(xnry, mean_nry, num_bins, density, "nry_t", cfgs)
    hist_(xntx, mean_ntx, num_bins, density, "ntx_t", cfgs)
    hist_(xntz, mean_ntz, num_bins, density, "ntz_t", cfgs)

def run(cfgs):
    evaluation(cfgs, "train_no_noise")

def main():
    from vo_training import prepare_parser
    parser = prepare_parser()
    config = vars(parser.parse_args())
    run(config)

if __name__ == '__main__':
    main()