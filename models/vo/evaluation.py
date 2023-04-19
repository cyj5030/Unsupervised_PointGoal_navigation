import torch
import numpy as np
import copy
import matplotlib.pyplot as plt
import cv2
import os
import glob
from tqdm import tqdm
import shutil

import sys
sys.path.append(os.getcwd())
from models.vo.vo_module import VO_Module
from models.vo.dataset import VODataset
from models.vo.vo_utils import *
from models.vo.training_utils import *

def compute_pose_error(pred, gt, sid):
    gt_mat = torch.stack(gt, 0)
    pred_mat = torch.stack(pred, 0)

    gt_mat[:, 2, 3] *= -1
    gt_mat[:, :3, :3] = gt_mat[:, :3, :3].inverse()
    init_gt = gt_mat[0:1].expand_as(gt_mat).clone()
    gt_mat = init_gt.inverse().matmul(gt_mat)
    
    state_errors = gt_mat.inverse().matmul(pred_mat)

    errors = {}
    errors['rot_error'] = rotation_error(state_errors).mean()
    errors['trans_error'] = translation_error(state_errors[:, :3, 3]).mean()
    log_str = "Sequence: %d, seq_lan: %d, avg rot err: %f, avg trans err: %f" % \
            (sid, gt_mat.shape[0], errors['rot_error'], errors['trans_error'])
    print(log_str)
    
    return log_str

def compute_depth_error(pred, gt, log_path, log_name, sid):
    pred_tensor = torch.stack(pred, 0)
    gt_tensor = torch.stack(gt, 0)
    
    sv_path = os.path.join(log_path, log_name, "depth_error")
    if not os.path.exists(sv_path):
        os.makedirs(sv_path)

    depth_error_maps = (pred_tensor - gt_tensor).abs().numpy()
    for did in range(depth_error_maps.shape[0]):
        basename = "%s_%s.png" % (str(sid).zfill(4), str(did).zfill(4))
        cv2.imwrite(os.path.join(sv_path, basename), depth_error_maps[did] * 255)

    errors = {}
    errors['depth_error'] = depth_error_maps.mean()


def draw_plot(pred, gt, log_path, log_name, sid, gps, rpose):
    gt_mat = torch.stack(gt, 0)
    pred_mat = torch.stack(pred, 0)

    gt_mat[:, 2, 3] *= -1
    gt_mat[:, :3, :3] = gt_mat[:, :3, :3].inverse()
    init_gt = gt_mat[0:1].expand_as(gt_mat).clone()
    gt_mat = init_gt.inverse().matmul(gt_mat)
    
    fig, ax = plt.subplots()
    ax.plot(gt_mat[:, 0, 3], gt_mat[:, 2, 3], label="gt")
    ax.plot(pred_mat[:, 0, 3], pred_mat[:, 2, 3], label="pred")
    if gps is not None:
        gps_mat = torch.stack(gps, 0)
        rpose = torch.stack(rpose, 0)[1:]
        ax.plot(gps_mat[:, 0], gps_mat[:, 2], label="gps")

        rpose[:, 2, 3] *= -1
        rpose[:, :3, :3] = rpose[:, :3, :3].inverse()
        pred_gps = gps_mat.clone()
        for i in range(rpose.shape[0]):
            pred_gps[i+1] = rpose[i, :3, :3].inverse().matmul((pred_gps[i] - rpose[i, :3, 3]).view(3, 1)).view(3)
        ax.plot(pred_gps[:, 0], pred_gps[:, 2], label="pred_gps")

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

def draw_image(image, log_path, log_name, sid, folder_name):
    sv_path = os.path.join(log_path, log_name, folder_name)
    if not os.path.exists(sv_path):
        os.makedirs(sv_path)

    for iid, im in enumerate(image):
        basename = "%s_%s.png" % (str(sid).zfill(4), str(iid).zfill(4))
        cv2.imwrite(os.path.join(sv_path, basename), im.numpy() * 255)

@torch.no_grad()
def evaluation(cfgs, run_type, net: VO_Module, plot=True):
    out_size = (cfgs["rgb_height"], cfgs["rgb_width"])
    dataset = VODataset(cfgs["dataset_path"], run_type, cfgs['num_frame'], out_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    device = cfgs["device"]
    net = net.to(device)
    net.eval()

    seqs = []
    
    for i, data in tqdm(enumerate(dataloader), desc="eval"):
        for key, itms in data.items():
            data[key] = itms.to(device)

        outputs = net.eval_infer(data)

        curr_iid = data['iid'].item()
        if curr_iid == cfgs["num_frame"] - 1:
            if i > 0:
                seqs.append(copy.deepcopy(seq))

            seq = {
                "pred_pose": [],
                "gt_pose": [],
                "rgb": [],
                "depth": [],
                # "pred_depth": [],
                "gps": [],
                "pr_pose": [],
            }
            seq["pred_pose"].append(torch.eye(4))
            seq["pr_pose"].append(torch.eye(4))
            # seq["pred_depth"].append(data["depth"][0, 0:1].cpu().permute(1,2,0))
            seq["gt_pose"].append(data["state"][0, 0:4, 0:4].cpu())
            seq["rgb"].append( data["rgb"][0, 0:3].cpu().permute(1,2,0) )
            seq["depth"].append( data["depth"][0, 0:1].cpu().permute(1,2,0) )
            seq["gps"].append( data["gps"][0,0:3].cpu() )

        # if len(seqs) >= 1:
        #     break 

        pred_pose = outputs["pose", -1][0].detach().cpu()
        seq["pred_pose"].append(torch.matmul(seq["pred_pose"][-1], pred_pose))
        seq["pr_pose"].append(pred_pose)
        # seq["pred_depth"].append( outputs['depth', 0, 0][0].cpu().permute(1,2,0) / net.max_depth )

        seq["gt_pose"].append(data["state"][0, 4:8, 0:4].cpu())
        seq["rgb"].append( data["rgb"][0, 3:6].cpu().permute(1,2,0) )
        seq["depth"].append( data["depth"][0, 1:2].cpu().permute(1,2,0) )
        seq["gps"].append( data["gps"][0,3:6].cpu() )
    
    errors = []
    for i, s in enumerate(seqs):
        errors.append(compute_pose_error(s["pred_pose"], s["gt_pose"], i))
        if plot:
            # compute_depth_error(s["pred_depth"], s["depth"], cfgs['log_path'], cfgs['log_name'], i)
            draw_plot(s["pred_pose"], s["gt_pose"], cfgs['log_path'], cfgs['log_name'], i, s["gps"], s['pr_pose'])
            # draw_image(s["pred_depth"], cfgs['log_path'], cfgs['log_name'], i, "pred_depth")
            # draw_image(s["depth"], cfgs['log_path'], cfgs['log_name'], i, "depth")
            # draw_warped_image(s["rgb"], s["gt"], s["depth"], cfgs['log_path'], cfgs['log_name'], i)

    sv_path = os.path.join(cfgs['log_path'], cfgs['log_name'], "vo_eval.log")
    errors = [e+"\n" for e in errors]
    with open(sv_path, 'w') as f:
        f.writelines(errors)


def run(cfgs):
    torch.backends.cudnn.benchmark = True

    cfgs['log_name'] = "vo_irgbd_trgbd_3frame_cmodel"
    with open(os.path.join(cfgs['log_path'], cfgs['log_name'], "config.yaml"),'r') as f:
        cfgs_new = yaml.load(f, Loader=yaml.SafeLoader)
    cfgs.update(cfgs_new)

    cfgs["device"] = "cuda:1"
    cfgs["num_frame"] = 2
    cfgs["reuse_iter"] = 170000

    net = VO_Module(cfgs)

    reuse_path = os.path.join(cfgs['log_path'], cfgs['log_name'], "checkpoints")
    if cfgs["reuse_iter"] < 0:
        reuse_files = glob.glob(os.path.join(reuse_path, '*.pth'))
        reuse_ids = sorted([int(filename.split('.')[-2]) for filename in reuse_files])
    else:
        reuse_ids = [cfgs["reuse_iter"]]

    reuse_file = os.path.join(reuse_path, "ckpt.%d.pth" % (reuse_ids[-1]))
    params = torch.load(reuse_file, map_location="cpu")
    net.load_state_dict(params["model_params"])
    print('Load params from %s.' % (reuse_file))

    evaluation(cfgs, "eval_no_noise", net, plot=True)

def main():
    from vo_training import prepare_parser
    parser = prepare_parser()
    config = vars(parser.parse_args())
    run(config)

if __name__ == '__main__':
    main()