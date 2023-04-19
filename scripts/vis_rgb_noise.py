import argparse
import os
import quaternion
import random
import numpy as np
import torch
import imageio

import sys
sys.path.append(os.getcwd())

from models.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
# from habitat.core.environments import get_env_class # habitat 0.2.2
from models.rl.ddppo.ddp_utils import is_slurm_batch_job

from configs.default import get_config
from habitat.config import Config as CN

import h5py
from tqdm import tqdm

def data_collect(config, exp_config):
    envs = construct_envs(
        config,
        get_env_class(config.ENV_NAME),
        workers_ignore_signals=is_slurm_batch_job(),
    )

    observations = envs.reset()
    eps = envs.current_episodes()
    agent_state = envs.get_agent_state()

    intensity = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.NOISE_MODEL_KWARGS.intensity_constant

    for _id in tqdm(range(int(exp_config.total_step))):
        actions = envs.get_next_action()
        outputs = envs.step(actions)
        eps = envs.current_episodes()
        agent_state = envs.get_agent_state()

        observations, rewards_l, dones, infos = [
            list(x) for x in zip(*outputs)
        ]

        rgb = observations[0]['rgb']
        save_name = os.path.join(exp_config.out_dir, f"{_id:0>3d}_{intensity:.2f}.png")
        imageio.imwrite(save_name, rgb)

    envs.close()

def run_exp(exp_config):
    os.environ['GLOG_minloglevel'] = "2"
    os.environ['MAGNUM_LOG'] = "quiet" # verbose quiet
    # os.environ['MAGNUM_GPU_VALIDATION'] = "ON" 
    # os.environ['CUDA_VISIBLE_DEVICES'] = exp_config.gpu

    config = get_config(exp_config.cfg_path)

    config.defrost()
    config.TASK_CONFIG.DATASET.SPLIT = exp_config.split
    config.NUM_ENVIRONMENTS = exp_config.num_env
    config.TASK_CONFIG.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.DIMENSIONALITY = 3
    config.TASK_CONFIG.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.GOAL_FORMAT = "CARTESIAN"
    config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
    config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.NOISE_MODEL = "GaussianNoiseModel"
    config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.NOISE_MODEL_KWARGS = CN()
    config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.NOISE_MODEL_KWARGS.intensity_constant = 0.05
    
    # config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.pop("NOISE_MODEL_KWARGS")
    # config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.pop("NOISE_MODEL")
    # config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    # config.TASK_CONFIG.SIMULATOR.ACTION_SPACE_CONFIG = "v0"
    config.freeze()
    
    # reproducibility set up
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    torch.cuda.manual_seed_all(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # print(torch.cuda.device_count())
    # if exp_config.split == "train":
    sigma = [0, 0.075, 0.1]
    for sigma_value in sigma:
        
        config.defrost()
        config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.NOISE_MODEL_KWARGS.intensity_constant = sigma_value
        config.freeze()

        data_collect(config, exp_config)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='./configs/pointnav/pointnav.yaml', 
                        help="path to config yaml containing info about experiment")
    parser.add_argument('--total_step', type=int, default=1000, help='number of eps for collection')
    parser.add_argument('--num_env', type=int, default=1, help='number of environment to run simultaneously')
    parser.add_argument('--save_step', type=int, default=100, help='number of step for collection')
    parser.add_argument("--gpu", type=str, default="3", help="gpus",)
    parser.add_argument('--split', type=str, default="val", choices=['train','val'], help='data split to use')
    parser.add_argument('--out_dir', type=str, default="./train_log/vis", help='directory to save the collected data')
    args = parser.parse_args()

    run_exp(args)

if __name__ == "__main__":
    main()
