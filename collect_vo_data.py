import argparse
import os
import quaternion
import random
import numpy as np
import torch

from models.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
# from habitat.core.environments import get_env_class # habitat 0.2.2
from models.rl.ddppo.ddp_utils import is_slurm_batch_job

from configs.default import get_config

import h5py
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='configs/pointnav/pointnav.yaml', 
                        help="path to config yaml containing info about experiment")
    parser.add_argument('--total_step', type=int, default=1e5, help='number of eps for collection')
    parser.add_argument('--num_env', type=int, default=6, help='number of environment to run simultaneously')
    parser.add_argument('--save_step', type=int, default=100, help='number of step for collection')
    parser.add_argument("--gpu", type=str, default="0", help="gpus",)
    parser.add_argument('--split', type=str, default="train", choices=['train','val'], help='data split to use')
    parser.add_argument('--out_dir', type=str, default="./VO_data", help='directory to save the collected data')
    args = parser.parse_args()

    run_exp(args)

def asMatrix(q, t):
    mat = np.eye(4)
    mat[:3, :3] = quaternion.as_rotation_matrix(q)
    mat[:3, 3] = t
    return mat

def data_collect(config, exp_config):
    envs = construct_envs(
        config,
        get_env_class(config.ENV_NAME),
        workers_ignore_signals=is_slurm_batch_job(),
    )
    
    gps_dim = config.TASK_CONFIG.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.DIMENSIONALITY

    rgb = np.zeros([exp_config.save_step + 1, exp_config.num_env, 256, 256, 3])
    depth = np.zeros([exp_config.save_step + 1, exp_config.num_env, 256, 256, 1])
    gps = np.zeros([exp_config.save_step + 1, exp_config.num_env, gps_dim])
    done_mask = np.ones([exp_config.save_step + 1, exp_config.num_env]).astype("bool")
    goal = np.zeros([exp_config.save_step + 1, exp_config.num_env, 3])
    state = np.zeros([exp_config.save_step + 1, exp_config.num_env, 4, 4])
    action = np.zeros([exp_config.save_step + 1, exp_config.num_env])
    collision = np.zeros([exp_config.save_step + 1, exp_config.num_env]).astype("bool")

    observations = envs.reset()
    eps = envs.current_episodes()
    agent_state = envs.get_agent_state()

    step_id = 0
    rgb[step_id] = np.stack([observations[i]['rgb'] for i in range(exp_config.num_env)], 0)
    depth[step_id] = np.stack([observations[i]['depth'] for i in range(exp_config.num_env)], 0)
    gps[step_id] = np.stack([observations[i]['pointgoal_with_gps_compass'] for i in range(exp_config.num_env)], 0)
    goal[step_id] = np.array([eps[i].goals[0].position for i in range(exp_config.num_env)])
    state[step_id] = np.stack([
        asMatrix(agent_state[i].rotation, 
                agent_state[i].position
        ) 
        for i in range(exp_config.num_env)], 0)
    step_id += 1

    save_id = 0
    save_eps_id = 0
    for _id in tqdm(range(int(exp_config.total_step))):
        actions = envs.get_next_action()
        outputs = envs.step(actions)
        eps = envs.current_episodes()
        agent_state = envs.get_agent_state()

        observations, rewards_l, dones, infos = [
            list(x) for x in zip(*outputs)
        ]

        rgb[step_id] = np.stack([observations[i]['rgb'] for i in range(exp_config.num_env)], 0)
        depth[step_id] = np.stack([observations[i]['depth'] for i in range(exp_config.num_env)], 0)
        gps[step_id] = np.stack([observations[i]['pointgoal_with_gps_compass'] for i in range(exp_config.num_env)], 0)
        goal[step_id] = np.array([eps[i].goals[0].position for i in range(exp_config.num_env)])
        state[step_id] = np.stack([
            asMatrix(agent_state[i].rotation, 
                    agent_state[i].position
            ) 
            for i in range(exp_config.num_env)], 0)
        action[step_id] = np.array(actions)
        collision[step_id] = np.array([_info["collisions"]["is_collision"] for _info in infos])
        done_mask[step_id] = np.array(dones)
        step_id += 1
        
        if step_id == exp_config.save_step + 1:
            seq_rgb = rgb.transpose((1, 0, 2, 3, 4)).reshape(-1, 256, 256, 3)
            seq_depth = depth.transpose((1, 0, 2, 3, 4)).reshape(-1, 256, 256, 1)
            seq_gps = gps.transpose((1, 0, 2)).reshape(-1, 3)
            seq_goal = goal.transpose((1, 0, 2)).reshape(-1, 3)
            seq_state = state.transpose((1, 0, 2, 3)).reshape(-1, 4, 4)
            seq_action = action.transpose((1, 0)).reshape(-1)
            seq_collision = collision.transpose((1, 0)).reshape(-1)
            done_mask[-1] = True
            seq_done_mask = done_mask.transpose((1, 0)).reshape(-1)

            non_zeros_ids = seq_done_mask.nonzero()[0]
            eps_star_ids, eps_end_ids = non_zeros_ids[:-1], non_zeros_ids[1:]

            dataset_name = os.path.join(exp_config.out_dir, exp_config.split, str(save_id).zfill(6) + '.hdf5')
            if not os.path.exists(os.path.dirname(dataset_name)):
                os.makedirs(os.path.dirname(dataset_name))

            with h5py.File(dataset_name, "w") as f:
                prev_save_eps_id = save_eps_id
                for i in range(len(eps_star_ids)):
                    if eps_end_ids[i] - eps_star_ids[i] < 2:
                        continue
                    seq_slice = slice(eps_star_ids[i], eps_end_ids[i])
                    suffix = str(save_eps_id).zfill(6)
                    f.create_dataset("rgb_" + suffix, data=seq_rgb[seq_slice].astype(np.uint8))
                    f.create_dataset("depth_" + suffix, data=seq_depth[seq_slice].astype(np.float32))
                    f.create_dataset("gps_" + suffix, data=seq_gps[seq_slice].astype(np.float32))
                    f.create_dataset("goal_" + suffix, data=seq_goal[seq_slice].astype(np.float32))
                    f.create_dataset("state_" + suffix, data=seq_state[seq_slice].astype(np.float32))
                    f.create_dataset("action_" + suffix, data=seq_action[seq_slice].astype(np.uint8))
                    f.create_dataset("collision_" + suffix, data=seq_collision[seq_slice].astype("bool"))
                    save_eps_id += 1
                # f.create_dataset("sequences", data=np.array([save_eps_id - prev_save_eps_id]))
            os.rename(dataset_name, dataset_name.replace(
                                    str(save_id).zfill(6) + '.hdf5', 
                                    "%s_%s.hdf5" % (str(prev_save_eps_id).zfill(6), str(save_eps_id - 1).zfill(6))))
            save_id += 1

            rgb[0] = rgb[-1]
            depth[0] = depth[-1]
            gps[0] = gps[-1]
            goal[0] = goal[-1]
            state[0] = state[-1]
            action[0] = action[-1]
            collision[0] = collision[-1]
            done_mask[0] = done_mask[-1]
            step_id = 1

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
    # config.TASK_CONFIG.SIMULATOR.TURN_ANGLE = 10
    # config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.pop("NOISE_MODEL")
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
    if exp_config.split == "train":
        data_collect(config, exp_config)


if __name__ == "__main__":
    main()
