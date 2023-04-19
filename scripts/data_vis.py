import torch
import numpy as np
import copy
import matplotlib.pyplot as plt
import cv2
import os
import glob
from tqdm import tqdm

from scipy.stats import norm
from matplotlib.pyplot import Axes

import sys
sys.path.append(os.getcwd())
from models.vo.dataset import VODataset, VODataset_small_chunk
from models.vo.vo_utils import *
from models.vo.lie_algebra import SO3_CPU
from models.vo.icp import icp

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

def valid_gps(gt, gps):
    gt_mat = torch.stack(gt, 0)
    gps_mat = torch.stack(gps, 0)

    gt_mat[:, 2, 3] *= -1
    gt_mat[:, :3, :3] = gt_mat[:, :3, :3].inverse()
    T_cam = gt_mat[1:].inverse().matmul(gt_mat[:-1]) # ~ pred T_cam

    T_cam = T_cam.inverse()
    T_cam[:, 2, 3] *= -1
    T_cam[:, :3, :3] = T_cam[:, :3, :3].inverse()

    pred_gps = T_cam[:, :3, :3].inverse().matmul((gps_mat[:-1] - T_cam[:, :3, 3]).view(-1, 3, 1))
    error = pred_gps.view(-1, 3) - gps_mat[1:]

    print("e_x=%f, e_z=%f\n" % (error[:, 0].mean().item(), error[:,2].mean().item()))

def valid_icp(gt, depth, log_path, log_name, sid):
    gt_mat = torch.stack(gt, 0)
    gt_mat[:, 2, 3] *= -1
    gt_mat[:, :3, :3] = gt_mat[:, :3, :3].inverse()

    T_cam = gt_mat[1:].inverse().matmul(gt_mat[:-1])
    
    depth = torch.stack(depth, 0).permute(0, 3, 1, 2)
    B, _, H, W = depth.shape

    hfov = float(70) * np.pi / 180.
    K = torch.from_numpy(np.array([
        [1 / np.tan(hfov / 2.), 0., 0., 0.],
        [0., 1 / np.tan(hfov / 2.), 0., 0.],
        [0., 0.,  1, 0],
        [0., 0., 0, 1]
    ]))[:3, :3].float().view(1, 3, 3).expand([B-1, 3, 3])
    K_inv = K.inverse()

    coords = torch.cat([coords_grid(B-1, H, W), torch.ones([B-1, 1, H, W])], 1)
    xyz0 = torch.matmul(K_inv, coords.view(B-1, 3, -1)) * depth[:-1].view(B-1, 1, -1) * 10.
    xyz1 = torch.matmul(K_inv, coords.view(B-1, 3, -1)) * depth[1:].view(B-1, 1, -1) * 10.

    numpy_xyz0 = xyz0.numpy()
    numpy_xyz1 = xyz1.numpy()
    
    num_sample = 3000
    random_select_points = np.random.randint(numpy_xyz0.shape[2], size=num_sample)

    numpy_xyz0 = numpy_xyz0[:, :, random_select_points]
    numpy_xyz1 = numpy_xyz1[:, :, random_select_points]
    T_icp = []
    for ib in range(B-1):
        T_icp.append(icp(numpy_xyz0[ib].T, numpy_xyz1[ib].T, max_iterations=50)[0])
    T_icp = torch.from_numpy(np.stack(T_icp, 0)).float()

    err = T_cam.inverse().matmul(T_icp)
    errors = {}
    errors['rot_error'] = rotation_error(err).mean()
    errors['trans_error'] = translation_error(err[:, :3, 3]).mean()
    print("Sequence: %d, seq_lan: %d, avg rot err: %f, avg trans err: %f" % 
            (sid, gt_mat.shape[0], errors['rot_error'], errors['trans_error']))
    
    fig, ax = plt.subplots()
    init_gt = gt_mat[0:1].expand_as(gt_mat).clone()
    T_gt = init_gt.inverse().matmul(gt_mat)
    
    T_icps = torch.zeros_like(T_gt)
    T_icps[0] = T_gt[0]
    for i in range(B-1):
        T_icps[i+1] = T_icps[i].matmul(T_icp[i].inverse())
    ax.plot(T_gt[:, 0, 3], T_gt[:, 2, 3], label="gt")
    ax.plot(T_icps[:, 0, 3], T_icps[:, 2, 3], label="pred")

    font = {
        'weight' : 'normal',
        'size'   : 18,
    }
    ax.legend(loc='upper right', prop=font)
    
    sv_path = os.path.join(log_path, log_name, 'figures')
    if not os.path.exists(sv_path):
        os.makedirs(sv_path)
        
    plt.rcParams['savefig.dpi'] = 300
    plt.savefig(os.path.join(sv_path, str(sid).zfill(6) + '.png'), bbox_inches="tight", pad_inches=0.0)
    plt.show()
    plt.close()


def draw_warped_image(image, gt, depth, log_path, log_name, sid, action=None):
    gt_mat = torch.stack(gt, 0)
    image = torch.stack(image, 0).permute(0, 3, 1, 2)
    depth = torch.stack(depth, 0).permute(0, 3, 1, 2)

    if action is None:
        T_cam = gt_mat[:-1].inverse().matmul(gt_mat[1:])
        foldername = "warped_rgb"
    else:
        T_cam = []
        for i in action[1:]:
            T_cam.append(copy.deepcopy(action_piror[i.item()]))
        T_cam = torch.stack(T_cam, 0)
        foldername = "action_warped_rgb"

    B, _, H, W = image.shape
    coords = torch.cat([coords_grid(B-1, H, W), torch.ones([B-1, 1, H, W])], 1)

    hfov = float(70) * np.pi / 180.
    K = torch.from_numpy(np.array([
        [1 / np.tan(hfov / 2.), 0., 0., 0.],
        [0., 1 / np.tan(hfov / 2.), 0., 0.],
        [0., 0.,  1, 0],
        [0., 0., 0, 1]
    ]))[:3, :3].float()

    K_inv = K.inverse()

    coords_proj = reproject(coords,
                            depth[:-1] * 10.0, 
                            T_cam,
                            K.view(1, 3, 3).expand([B-1, 3, 3]), 
                            K_inv.view(1, 3, 3).expand([B-1, 3, 3]),
    )
    depth_mask = (depth[:-1] > 0).float()
    warped_rgb = grid_sampler(image[1:], coords_proj) * depth_mask
    out_mask = out_boundary_mask(coords_proj)
    error_image = (warped_rgb - image[1:]).abs() * depth_mask * out_mask * (depth[1:] > 0).float()

    warped_rgb = warped_rgb.permute(0, 2, 3, 1)
    for iid in range(warped_rgb.shape[0]):
        sv_path = os.path.join(log_path, log_name, foldername)
        if not os.path.exists(sv_path):
            os.makedirs(sv_path)
        basename = "%s_%s.png" % (str(sid).zfill(4), str(iid).zfill(4))
        cv2.imwrite(os.path.join(sv_path, basename), warped_rgb[iid].numpy() * 255)
    
    error_image = error_image.permute(0, 2, 3, 1)
    for iid in range(error_image.shape[0]):
        sv_path = os.path.join(log_path, log_name, "error_" + foldername)
        if not os.path.exists(sv_path):
            os.makedirs(sv_path)
        basename = "%s_%s.png" % (str(sid).zfill(4), str(iid).zfill(4))
        cv2.imwrite(os.path.join(sv_path, basename), error_image[iid].numpy() * 255)
    
    out_mask = out_mask.permute(0, 2, 3, 1)
    for iid in range(out_mask.shape[0]):
        sv_path = os.path.join(log_path, log_name, "out_mask_" + foldername)
        if not os.path.exists(sv_path):
            os.makedirs(sv_path)
        basename = "%s_%s.png" % (str(sid).zfill(4), str(iid).zfill(4))
        cv2.imwrite(os.path.join(sv_path, basename), out_mask[iid].numpy() * 255)

def draw_image(image, log_path, log_name, sid):
    if image[0].shape[2] == 3:
        save_folder_name = "rgb" 
    else:
        save_folder_name = "depth"

    sv_path = os.path.join(log_path, log_name, save_folder_name)
    if not os.path.exists(sv_path):
        os.makedirs(sv_path)

    for iid, im in enumerate(image):
        basename = "%s_%s.png" % (str(sid).zfill(4), str(iid).zfill(4))
        cv2.imwrite(os.path.join(sv_path, basename), im.numpy() * 255)

def evaluation(cfgs, run_type):
    out_size = (cfgs["rgb_height"], cfgs["rgb_width"])
    cfgs["num_frame"] = 2
    dataset = VODataset(cfgs["dataset_path"], run_type, cfgs["num_frame"], out_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    seqs = []
    for i, data in tqdm(enumerate(dataloader), desc="data vis"):

        curr_iid = data['iid'].item()
        if curr_iid == cfgs["num_frame"] - 1:
            if i > 0:
                seqs.append(copy.deepcopy(seq))

            seq = {
                "gt": [],
                "rgb": [],
                "depth": [],
                "action": [],
                "gps": [],
            }
            seq["gt"].append(data["state"][0, :4, :4].cpu())
            seq["rgb"].append( data["rgb"][0,:3].cpu().permute(1,2,0) )
            seq["depth"].append( data["depth"][0,:1].cpu().permute(1,2,0) )
            seq["action"].append( data["action"][0,:1].cpu() )
            seq["gps"].append( data["gps"][0,0:3].cpu() )

        if len(seqs) >= 5:
            break 

        seq["gt"].append(data["state"][0, 4:8, :4].cpu())
        seq["rgb"].append( data["rgb"][0, 3:].cpu().permute(1,2,0) )
        seq["depth"].append( data["depth"][0,1:].cpu().permute(1,2,0) )
        seq["action"].append( data["action"][0,1:].cpu() )
        seq["gps"].append( data["gps"][0,3:6].cpu() )

    for i, s in enumerate(seqs):
        # valid_gps(s["gt"], s["gps"])
        valid_icp(s["gt"], s["depth"], cfgs['log_path'], cfgs['log_name'], i)

        # draw_warped_image(s["rgb"], s["gt"], s["depth"], cfgs['log_path'], cfgs['log_name'], i, s['action'])
        # draw_warped_image(s["rgb"], s["gt"], s["depth"], cfgs['log_path'], cfgs['log_name'], i)
        
        # draw_image(s["rgb"], cfgs['log_path'], cfgs['log_name'], i)
        # draw_image(s["depth"], cfgs['log_path'], cfgs['log_name'], i)

def mat2rotv(T_cam):
    Bs = T_cam.shape[0]
    rotv = np.zeros([Bs, 3])
    trans = np.zeros([Bs, 3])
    for i in range(Bs):
        rotv[i] = q.as_rotation_vector(q.from_rotation_matrix(T_cam[i, 0:3, 0:3]))
        trans[i] = T_cam[i, 0:3, 3]
    return rotv, trans

def hist_and_fit(data, num_bins, density, suffix, cfgs):
    fig, axes = plt.subplots()
    num_bins = 50
    mu, sigma = data.mean(), data.std()
    n, bins, patches = axes.hist(data, num_bins, density=density)
    y = norm.pdf(bins, mu, sigma)
    axes.plot(bins, y, 'r--')
    axes.set_title("%s: $\mu$=%.2f, $\sigma$=%.2f" % (suffix, mu, sigma))

    sv_path = os.path.join(cfgs["log_path"], cfgs["log_name"], 'figures')
    if not os.path.exists(sv_path):
        os.makedirs(sv_path)
        
    plt.rcParams['savefig.dpi'] = 300
    plt.savefig(os.path.join(sv_path, suffix + '.png'), bbox_inches="tight", pad_inches=0.0)
    plt.show()
    plt.close()

def static_moving_noise(cfgs, run_type, max_static_step=5000):
    out_size = (cfgs["rgb_height"], cfgs["rgb_width"])
    cfgs["num_frame"] = 2
    dataset = VODataset_small_chunk(cfgs["dataset_path"], run_type, cfgs["num_frame"], out_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0, drop_last=False)

    forward_errors = []
    left_errors = []
    right_errors = []
    forward_gt = []
    left_gt = []
    right_gt = []
    all_gt = []
    SO3 = SO3_CPU()
    for i, data in tqdm(enumerate(dataloader), desc="static_moving_noise"):
        T_cam_gt_0 = data["state"][:, 0:4, 0:4]
        T_cam_gt_1 = data["state"][:, 4:8, 0:4]
        T_cam_gt_0[:, 2, 3] *= -1
        T_cam_gt_0[:, :3, :3] = T_cam_gt_0[:, :3, :3].inverse()
        T_cam_gt_1[:, 2, 3] *= -1
        T_cam_gt_1[:, :3, :3] = T_cam_gt_1[:, :3, :3].inverse()

        T_cam_gt = T_cam_gt_1.inverse().matmul(T_cam_gt_0)
        T_cam_act = torch.stack([action_piror[aid.item()] for aid in data["action"][:, 1]], 0)

        rotv_gt = SO3.log(T_cam_gt[:, :3, :3])
        trans_gt = T_cam_gt[:, :3, 3]
        # rotv_act, trans_act = mat2rotv(T_cam_act)

        # error_rotv = rotv_gt - rotv_act
        # error_trans = trans_gt - trans_act

        # error_rotv, error_trans = mat2rotv(T_cam_act.inverse().matmul(T_cam_gt))
        error_rotv = SO3.log(T_cam_act.inverse().matmul(T_cam_gt)[:, :3, :3])
        error_trans = T_cam_gt[:, :3, 3] - T_cam_act[:, :3, 3]

        forward_masks = (data["action"][:, 1] == 1)
        left_masks = (data["action"][:, 1] == 2)
        right_masks = (data["action"][:, 1] == 3)

        forward_errors.append(np.concatenate([error_rotv[forward_masks], error_trans[forward_masks]], 1))
        left_errors.append(np.concatenate([error_rotv[left_masks], error_trans[left_masks]], 1))
        right_errors.append(np.concatenate([error_rotv[right_masks], error_trans[right_masks]], 1))

        forward_gt.append(np.concatenate([rotv_gt[forward_masks], trans_gt[forward_masks]], 1))
        left_gt.append(np.concatenate([rotv_gt[left_masks], trans_gt[left_masks]], 1))
        right_gt.append(np.concatenate([rotv_gt[right_masks], trans_gt[right_masks]], 1))

        all_gt.append(np.concatenate([rotv_gt, trans_gt], 1))
        if i >= max_static_step:
            break

    forward_errors = np.concatenate(forward_errors, 0)
    left_errors = np.concatenate(left_errors, 0)
    right_errors = np.concatenate(right_errors, 0)

    forward_gt = np.concatenate(forward_gt, 0)
    left_gt = np.concatenate(left_gt, 0)
    right_gt = np.concatenate(right_gt, 0)

    all_gt = np.concatenate(all_gt, 0)

    num_bins = 50
    hist_and_fit(forward_errors[:, 1], num_bins, 1, "errors_forward_roty", cfgs)
    hist_and_fit(forward_errors[:, 3], num_bins, 1, "errors_forward_transx", cfgs)
    hist_and_fit(forward_errors[:, 5], num_bins, 1, "errors_forward_transz", cfgs)

    hist_and_fit(left_errors[:, 1], num_bins, 1, "errors_left_roty", cfgs)
    hist_and_fit(left_errors[:, 3], num_bins, 1, "errors_left_transx", cfgs)
    hist_and_fit(left_errors[:, 5], num_bins, 1, "errors_left_transz", cfgs)

    hist_and_fit(right_errors[:, 1], num_bins, 1, "errors_right_roty", cfgs)
    hist_and_fit(right_errors[:, 3], num_bins, 1, "errors_right_transx", cfgs)
    hist_and_fit(right_errors[:, 5], num_bins, 1, "errors_right_transz", cfgs)

    ####
    hist_and_fit(forward_gt[:, 1], num_bins, 1, "gt_forward_roty", cfgs)
    hist_and_fit(forward_gt[:, 3], num_bins, 1, "gt_forward_transx", cfgs)
    hist_and_fit(forward_gt[:, 5], num_bins, 1, "gt_forward_transz", cfgs)

    hist_and_fit(left_gt[:, 1], num_bins, 1, "gt_left_roty", cfgs)
    hist_and_fit(left_gt[:, 3], num_bins, 1, "gt_left_transx", cfgs)
    hist_and_fit(left_gt[:, 5], num_bins, 1, "gt_left_transz", cfgs)

    hist_and_fit(right_gt[:, 1], num_bins, 1, "gt_right_roty", cfgs)
    hist_and_fit(right_gt[:, 3], num_bins, 1, "gt_right_transx", cfgs)
    hist_and_fit(right_gt[:, 5], num_bins, 1, "gt_right_transz", cfgs)

    ###
    hist_and_fit(all_gt[:, 1], num_bins, 1, "gt_roty", cfgs)
    hist_and_fit(all_gt[:, 3], num_bins, 1, "gt_transx", cfgs)
    hist_and_fit(all_gt[:, 5], num_bins, 1, "gt_transz", cfgs)


def run(cfgs):
    # static_moving_noise(cfgs, "train_no_noise", 200)
    evaluation(cfgs, "eval_no_noise")

def main():
    from vo_training import prepare_parser
    parser = prepare_parser()
    config = vars(parser.parse_args())
    run(config)

if __name__ == '__main__':
    main()