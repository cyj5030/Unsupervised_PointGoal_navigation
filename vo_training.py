from sched import scheduler
import torch

import argparse
import os
import glob
import time
import math
import shutil

from models.vo.vo_module import VO_Module
from models.vo.dataset import VODataset, VODataset_small_chunk
from models.vo.training_utils import *

def prepare_parser():
    parser = argparse.ArgumentParser(description='Parser for all scripts.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset_path', type=str, default='./VO_data',  help='dataset path.')
    parser.add_argument('--run_type', type=str, default='train',  help='train or eval.')
    parser.add_argument('--device', type=str, default='cuda:1',  help='cuda device.')

    '''
    log setting
    '''
    parser.add_argument('--log_path', type=str, default='./train_log',  help='path for log.')
    parser.add_argument('--log_name', type=str, default='debug',  help='folder name for log.')
    parser.add_argument('--print_iter', type=int, default=10,  help='print losses during number of iteration.')
    parser.add_argument('--eval_iter', type=int, default=1000,  help='eval during number of iteration.')
    parser.add_argument('--save_iter', type=int, default=5000,  help='save during number of iteration.')

    '''
    training setting
    '''
    parser.add_argument('--batchsize', type=int, default=32,  help='batch size.')
    parser.add_argument('--lr', type=float, default=1.e-4,  help='learning rate.')
    parser.add_argument('--train_epoch', type=int, default=20,  help='training epoches.')

    parser.add_argument('--num_frame', type=int, default=3,  help='training number of frames.')
    parser.add_argument('--train_scale', type=int, default=1,  help='training number of scale.')
    parser.add_argument('--hfov', type=int, default=70,  help='training number of scale.')
    parser.add_argument('--rec_loss_coef', type=float, default=2.0,  help='trans coef.')

    parser.add_argument('--pose_width', type=int, default=256,  help='reshape inputs width.')
    parser.add_argument('--pose_height', type=int, default=256,  help='reshape inputs height.')
    parser.add_argument('--dof', type=int, default=3,  help='training number of scale.')
    parser.add_argument('--rot_coef', type=float, default=0.01,  help='rpy coef.')
    parser.add_argument('--trans_coef', type=float, default=0.01,  help='trans coef.')

    parser.add_argument('--rgb_width', type=int, default=256,  help='reshape inputs width.')
    parser.add_argument('--rgb_height', type=int, default=256,  help='reshape inputs height.')
    parser.add_argument('--min_depth', type=float, default=0.1,  help='min depth.')
    parser.add_argument('--max_depth', type=float, default=10,  help='max depth.')
    
    # rgb rgbd rgbd_d
    parser.add_argument('--pose_inputs', type=str, default='rgb', help='posenet inputs mode.')
    parser.add_argument('--depth_d_planes', type=int, default=10,  help='depth Discretization planes.')

    # rgb rgbd
    parser.add_argument('--rec_target', type=str, default='rgbd', help='reconstruct target.')
    
    parser.add_argument('--loop_loss', action="store_true",  help='is or not use loop loss.')
    parser.add_argument('--loop_loss_coef', type=float, default=0.01,  help='loop loss coeff.')
    parser.add_argument('--loop_loss_max_inc', type=int, default=3,  help='loop loss coeff.')

    parser.add_argument('--act_spec', action="store_true",  help='is or not use loop loss.')
    parser.add_argument('--classification_model', action="store_true",  help='is or not use loop loss.')

    '''
    fine-tune setting
    '''
    parser.add_argument('--reuse', action="store_true",  help='is or not reuse model.')
    parser.add_argument('--reuse_name', type=str, default=None,  help='reuse path if none equal to log name.')
    parser.add_argument('--reuse_iter', type=int, default=-1,  help='is or not reuse model.')
    return parser

def trainer(net, dataset, cfgs): 
    # make folders
    log_path = os.path.join(cfgs['log_path'], cfgs['log_name'], "checkpoints")

    dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=cfgs['batchsize'], 
                                            shuffle=True, 
                                            num_workers=2, 
                                            drop_last=True,
                                            # prefetch_factor=int(cfgs['batchsize'] / 4),
                                            )
    
    # parameters
    lr = cfgs['lr']
    device = cfgs['device']

    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0}

    # optim
    params = net.parameters()
    # optim = torch.optim.Adam(params=params, lr=lr, weight_decay=0)
    optim = torch.optim.RMSprop(params=params, lr=lr, weight_decay=0)
    # WARM_UP_EPOCH = 5
    # END_EPOCH = cfgs['train_epoch']
    # warm_up_with_cosine_lr = lambda epoch: epoch / WARM_UP_EPOCH if epoch <= WARM_UP_EPOCH else 0.5 * ( math.cos((epoch - WARM_UP_EPOCH) /(END_EPOCH - WARM_UP_EPOCH) * math.pi) + 1)
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR( optim, lr_lambda=warm_up_with_cosine_lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=cfgs['train_epoch']-5, gamma=0.1, last_epoch=-1)

    # use pretrain model?
    if cfgs['reuse']:
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        else:
            if 'y'== input("if you want to delete %s. (y/n)" % (os.path.dirname(log_path))):
                shutil.rmtree(os.path.dirname(log_path))
                os.makedirs(log_path)
                
        if cfgs['reuse_name'] is None:
            cfgs['reuse_name'] = cfgs['log_name']
        reuse_path = os.path.join(cfgs['log_path'], cfgs['reuse_name'], "checkpoints")

        if cfgs["reuse_iter"] < 0:
            reuse_files = glob.glob(os.path.join(reuse_path, '*.pth'))
            reuse_ids = sorted([int(filename.split('.')[-2]) for filename in reuse_files])
            cfgs["reuse_iter"] = reuse_ids[-1]

        reuse_file = os.path.join(reuse_path, 'ckpt.%d.pth' % (cfgs["reuse_iter"]))
        if not os.path.exists(reuse_file):
            raise Exception("not exit %s !" % (reuse_file))

        params = torch.load(reuse_file, map_location="cpu")
        net.load_state_dict(params["model_params"])
        state_dict = params["state"]
        optim.load_state_dict(params["optim_params"])
        optim = optim_to(optim, device)
        print('Load params from %s.' % (reuse_file))
    else:
        if os.path.exists(log_path):
            if 'y'== input("if you want to delete %s. (y/n)" % (os.path.dirname(log_path))):
                shutil.rmtree(os.path.dirname(log_path))
                os.makedirs(log_path)
        else:
            os.makedirs(log_path)
            
        print_config(cfgs, os.path.dirname(log_path), True)
        
    # star training
    net = net.to(device)
    iters = len(dataloader)
    for epoch in range(cfgs['train_epoch']):
        state_dict['epoch'] += 1
        for i, data in enumerate(dataloader):
            for key, itms in data.items():
                data[key] = itms.to(device)

            # calc loss and flow
            net.train()
            losses, loss = net(data, state_dict)

            # update params
            optim.zero_grad()
            loss.backward()
            optim.step()

            # update state dict
            state_dict['itr'] += 1

            # print for every 10 iter
            if state_dict['itr'] % cfgs['print_iter'] == 0:
                print_losses(losses, state_dict, os.path.dirname(log_path), write=True)
                
            # test all
            if state_dict['itr'] % cfgs['save_iter'] == 0:
                save_ckpt(net.state_dict(), state_dict, optim.state_dict(), log_path)

        lr_scheduler.step()

    # save in the end
    save_ckpt(net.state_dict(), state_dict, optim.state_dict(), log_path)

def run(cfgs):
    torch.backends.cudnn.benchmark = True

    # loading dataset
    print('Loading dataset...')
    out_size = (cfgs["rgb_height"], cfgs["rgb_width"])
    dataset = VODataset(cfgs["dataset_path"], cfgs["run_type"], cfgs['num_frame'], out_size)

    # model
    print('Loading model...')
    # if cfgs['reuse']:
    #     if cfgs['reuse_name'] is None:
    #         cfgs['reuse_name'] = cfgs['log_name']
    #     reuse_yaml = os.path.join(cfgs['log_path'], cfgs['reuse_name'], "config.yaml")
    #     with open(reuse_yaml, 'r') as f:
    #         cfgs = yaml.load(f, Loader=yaml.SafeLoader)

    net = VO_Module(cfgs)
    trainer(net, dataset, cfgs)
        
def main():
    parser = prepare_parser()
    config = vars(parser.parse_args())
    run(config)

if __name__ == '__main__':
    main()