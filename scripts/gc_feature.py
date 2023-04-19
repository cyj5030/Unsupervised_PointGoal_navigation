from collections import OrderedDict
import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2
import argparse
import matplotlib.pyplot as plt
import torchvision
import scipy.stats as sci_stats
from tqdm.std import tqdm

import sys
sys.path.append(os.getcwd())
from models.gc.gc_model import GC_Module
from models.gc.dataset import GCDataset

def prepare_parser():
    parser = argparse.ArgumentParser(description='Parser for all scripts.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    '''
    dataset path
    '''
    parser.add_argument('-dataset_path', type=str, default='./GC_data',  help='dataset.')
    parser.add_argument('-ckpt_path', type=str, default='./train_log/pointnav_gc_feature_wloss_detach_visual_irgbtrgb/checkpoints/ckpt.339.pth',  help='dataset.')
    parser.add_argument('-map_width', type=int, default=25,  help='map_width.')
    parser.add_argument('-map_height', type=int, default=25,  help='map_height.')
    
    '''
    basic setting
    '''
    parser.add_argument('-device', type=str, default='cuda:0',  help='cuda device.')

    '''
    log setting
    '''
    parser.add_argument('-log_path', type=str, default='./checkpoints',  help='path for log.')
    parser.add_argument('-log_name', type=str, default='test',  help='folder name for log.')
    return parser

def rate_map(x, y, feature, bins):
    ratemap = sci_stats.binned_statistic_2d(x, y, feature, bins=bins, statistic='mean')[0]

    ratemap[np.isnan(ratemap)] = np.nanmean(ratemap)
    ratemap = cv2.GaussianBlur(ratemap, (3,3), sigmaX=1.0, sigmaY=0.0)
    return ratemap

def binned_1d(x, feature, bins):
    sci_stats.binned_statistic(x, feature, statistic='mean', bins=bins)

def save_fig(map, id, path):
    save_name = str(id).zfill(4) + '.png'
    plt.imshow(map, interpolation='none', cmap='jet') # bilinear none
    plt.axis('off')
    plt.savefig(os.path.join(path, save_name))
    plt.show()
    plt.close()

@torch.no_grad()
def test(cfgs, dataset, net):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    device = cfgs['device']

    # load 
    params = torch.load(cfgs["ckpt_path"], map_location="cpu")["state_dict"]
    new_params = OrderedDict()
    for _key, _value in params.items():
        if "actor_critic.net.gc_lstm" in _key:
            new_key = _key.replace("actor_critic.net.gc_lstm.", "")
            new_params[new_key] = _value
    net.load_state_dict(new_params)
    print(f'Load params from {cfgs["ckpt_path"]}')

    net = net.eval().to(device)

    position, gc_features, out_features = [], [], []
    for i, data in tqdm(enumerate(dataloader)):
        pseudo_gps, gps, action, goal, state, collision = [itms.to(device) for key, itms in data.items()]
        if gps.shape[1] < 40:
            continue
        x, pc, hd, out, hidden_states = net.one_seq(action, pseudo_gps, collision)

        # feature act
        pos_gt = np.round(100 * gps[0].cpu().numpy())[:, [0,2]]

        position.append(pos_gt)
        gc_features.append(x.cpu().numpy())
        out_features.append(out.cpu().numpy())

        # if i == 6000:
        #     break

    pos_gt = np.concatenate(position, 0)
    gc_features = np.concatenate(gc_features, 0)
    out_feature = np.concatenate(out_features, 0)

    bins = 80
    save_root = os.path.dirname(os.path.dirname(cfgs["ckpt_path"]))

    save_path = os.path.join(save_root, "spatial_feature_map", "hidden_layer")
    mkdir(save_path)
    for f_id in tqdm(range(gc_features.shape[1]), desc='processing features'):
        gc_map = rate_map(pos_gt[:, 0], pos_gt[:, 1], gc_features[:, f_id], bins=bins)
        save_fig(gc_map, f_id, save_path)
        # ratemap = sci_stats.binned_statistic_2d(pos_gt[:, 0], pos_gt[:, 1], feature[:, f_id], bins=bins, statistic='mean')[0]

        # ratemap[np.isnan(ratemap)] = np.nanmean(ratemap)
        # ratemap = cv2.GaussianBlur(ratemap, (3,3), sigmaX=1.0, sigmaY=0.0)
        # ratemap = (ratemap - np.min(ratemap)) / (np.max(ratemap) - np.min(ratemap))

        # save_name = str(f_id).zfill(4) + '.png'
        # plt.imshow(gc_map, interpolation='none', cmap='jet') # bilinear none
        # plt.axis('off')
        # plt.savefig(os.path.join(save_path, save_name))
        # plt.show()
        # plt.close()
    
    save_path = os.path.join(save_root, "spatial_feature_map", "out_layer")
    mkdir(save_path)
    for f_id in tqdm(range(out_feature.shape[1]), desc='processing features'):
        out_map = rate_map(pos_gt[:, 0], pos_gt[:, 1], out_feature[:, f_id], bins=bins)
        save_fig(out_map, f_id, save_path)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def run(cfgs):
    torch.backends.cudnn.benchmark = True

    # loading dataset
    print('Loading dataset...')
    dataset = GCDataset(cfgs['dataset_path'], 'train', False)
    
    # model
    print('Loading model...')
    net = GC_Module(4, 256, 32)

    test(cfgs, dataset, net)

def main():
    parser = prepare_parser()
    config = vars(parser.parse_args())
    run(config)

if __name__ == '__main__':
    main()