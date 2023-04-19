import argparse
import random
import os

import numpy as np
import torch

from habitat.config import Config
from configs.default import get_config

from models.rl.ppo.ppo_trainer import PPOTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-config", type=str, default='./configs/pointnav/pointnav.yaml', 
                        help="path to config yaml containing info about experiment")

    parser.add_argument("--run-type", choices=["train", "eval"], default="train", 
                        help="run type of the experiment (train or eval)")

    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, 
                        help="Modify config options from command line")

    args = parser.parse_args()
    run_exp(**vars(args))

def execute_exp(config: Config, run_type: str) -> None:
    r"""This function runs the specified config with the specified runtype
    Args:
    config: Habitat.config
    runtype: str {train or eval}
    """
    config.defrost()
    config.LOG_FILE = config.LOG_FILE.format(log_file_name=config.LOG_FILE_NAME)
    config.CHECKPOINT_FOLDER = config.CHECKPOINT_FOLDER.format(log_file_name=config.LOG_FILE_NAME)
    config.TENSORBOARD_DIR = config.TENSORBOARD_DIR.format(log_file_name=config.LOG_FILE_NAME)
    config.VIDEO_DIR = config.VIDEO_DIR.format(log_file_name=config.LOG_FILE_NAME)
    config.EVAL_CKPT_PATH_DIR = config.EVAL_CKPT_PATH_DIR.format(log_file_name=config.LOG_FILE_NAME)
    config.freeze()

    log_path = os.path.dirname(config.LOG_FILE)
    if not config.REUSE_CKPT and os.path.exists(log_path):
        if 'y'== input("if you want to delete %s. (y/n)" % (log_path)):
            import shutil
            shutil.rmtree(log_path)

    # reproducibility set up
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    torch.cuda.manual_seed_all(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if config.FORCE_TORCH_SINGLE_THREADED and torch.cuda.is_available():
        torch.set_num_threads(1)

    trainer = PPOTrainer(config)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()


def run_exp(exp_config: str, run_type: str, opts=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    os.environ['GLOG_minloglevel'] = "2"
    os.environ['MAGNUM_LOG'] = "quiet"
    
    config = get_config(exp_config, opts)
    execute_exp(config, run_type)


if __name__ == "__main__":
    main()
