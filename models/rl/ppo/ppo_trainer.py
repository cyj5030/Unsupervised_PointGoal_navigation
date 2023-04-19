#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import contextlib
import os
import pickle
import random
import time
from collections import OrderedDict, defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tqdm
from gym import spaces
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, VectorEnv, logger
from habitat.utils import profiling_wrapper
from habitat.utils.visualizations.utils import observations_to_image

from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
# from habitat.core.environments import get_env_class # habitat 0.2.2
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from models.rl.ddppo.algo import DDPPO
from models.rl.ddppo.ddp_utils import (
    EXIT,
    add_signal_handlers,
    get_distrib_size,
    init_distrib_slurm,
    is_slurm_batch_job,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
)
# from habitat_baselines.rl.ddppo.policy import (  # noqa: F401.
#     PointNavResNetPolicy,
# )

from habitat_baselines.utils.common import (
    ObservationBatchingCache,
    action_to_velocity_control,
    batch_obs,
    generate_video,
)
from habitat_baselines.utils.env_utils import construct_envs
# from habitat.utils.env_utils import construct_envs # habitat 0.2.2

from models.rl.ppo.policy import PointNavBaselinePolicy
from models.rl.ppo import PPO
from models.rl.ppo.policy import Policy

from models.vo.vo_module import VO_Module
import yaml
import copy

@baseline_registry.register_trainer(name="ddppo")
@baseline_registry.register_trainer(name="ppo")
class PPOTrainer(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    SHORT_ROLLOUT_THRESHOLD: float = 0.25
    _is_distributed: bool
    _obs_batching_cache: ObservationBatchingCache
    envs: VectorEnv
    agent: PPO
    actor_critic: Policy

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        self.obs_transforms = []

        self._static_encoder = False
        self._encoder = None
        self._obs_space = None

        # Distributed if the world size would be
        # greater than 1
        self._is_distributed = get_distrib_size()[2] > 1
        self._obs_batching_cache = ObservationBatchingCache()

        self.using_velocity_ctrl = (
            self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS
        ) == ["VELOCITY_CONTROL"]

    @property
    def obs_space(self):
        if self._obs_space is None and self.envs is not None:
            self._obs_space = self.envs.observation_spaces[0]

        return self._obs_space

    @obs_space.setter
    def obs_space(self, new_obs_space):
        self._obs_space = new_obs_space

    def _all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        r"""All reduce helper method that moves things to the correct
        device and only runs if distributed
        """
        if not self._is_distributed:
            return t

        orig_device = t.device
        t = t.to(device=self.device)
        torch.distributed.all_reduce(t)

        return t.to(device=orig_device)

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        
        observation_space = self.obs_space
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        # policy = baseline_registry.get_policy(self.config.RL.POLICY.name)
        self.actor_critic  = PointNavBaselinePolicy(
            observation_space=observation_space,
            action_space=self.policy_action_space,
            config=self.config
        )
        # self.actor_critic = PointNavBaselinePolicy.from_config(
        #     self.config, observation_space, self.policy_action_space
        # )
        self.obs_space = observation_space
        self.actor_critic.to(self.device)

        if (
            self.config.RL.DDPPO.pretrained_encoder
            or self.config.RL.DDPPO.pretrained
        ):
            pretrained_state = torch.load(
                self.config.RL.DDPPO.pretrained_weights, map_location="cpu"
            )

        if self.config.RL.DDPPO.pretrained:
            self.actor_critic.load_state_dict(
                {
                    k[len("actor_critic.") :]: v
                    for k, v in pretrained_state["state_dict"].items()
                }
            )
        elif self.config.RL.DDPPO.pretrained_encoder:
            prefix = "actor_critic.net.visual_encoder."
            self.actor_critic.net.visual_encoder.load_state_dict(
                {
                    k[len(prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(prefix)
                }
            )

        if not self.config.RL.DDPPO.train_encoder:
            self._static_encoder = True
            for param in self.actor_critic.net.visual_encoder.parameters():
                param.requires_grad_(False)

        if self.config.RL.DDPPO.reset_critic:
            nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        self.agent = (DDPPO if self._is_distributed else PPO)(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )

    def _init_envs(self, config=None):
        if config is None:
            config = self.config

        self.envs = construct_envs(
            config,
            get_env_class(config.ENV_NAME),
            workers_ignore_signals=is_slurm_batch_job(),
        )

    def _init_train(self):
        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            self.config: Config = resume_state["config"]
            self.using_velocity_ctrl = (
                self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS
            ) == ["VELOCITY_CONTROL"]

        if self.config.RL.DDPPO.force_distributed:
            self._is_distributed = True

        if is_slurm_batch_job():
            add_signal_handlers()

        if self._is_distributed:
            local_rank, tcp_store = init_distrib_slurm(
                self.config.RL.DDPPO.distrib_backend
            )
            if rank0_only():
                logger.info(
                    "Initialized DD-PPO with {} workers".format(
                        torch.distributed.get_world_size()
                    )
                )

            self.config.defrost()
            self.config.TORCH_GPU_ID = local_rank
            self.config.SIMULATOR_GPU_ID = local_rank
            # Multiply by the number of simulators to make sure they also get unique seeds
            self.config.TASK_CONFIG.SEED += (
                torch.distributed.get_rank() * self.config.NUM_ENVIRONMENTS
            )
            self.config.freeze()

            random.seed(self.config.TASK_CONFIG.SEED)
            np.random.seed(self.config.TASK_CONFIG.SEED)
            torch.manual_seed(self.config.TASK_CONFIG.SEED)
            self.num_rollouts_done_store = torch.distributed.PrefixStore(
                "rollout_tracker", tcp_store
            )
            self.num_rollouts_done_store.set("num_done", "0")

        if rank0_only() and self.config.VERBOSE:
            logger.info(f"config: {self.config}")

        profiling_wrapper.configure(
            capture_start_step=self.config.PROFILING.CAPTURE_START_STEP,
            num_steps_to_capture=self.config.PROFILING.NUM_STEPS_TO_CAPTURE,
        )

        self._init_envs()

        if self.using_velocity_ctrl:
            self.policy_action_space = self.envs.action_spaces[0][
                "VELOCITY_CONTROL"
            ]
            action_shape = (2,)
            discrete_actions = False
        else:
            self.policy_action_space = self.envs.action_spaces[0]
            action_shape = None
            discrete_actions = True

        ppo_cfg = self.config.RL.PPO
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.TORCH_GPU_ID)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if rank0_only() and not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_actor_critic_agent(ppo_cfg)
        if self._is_distributed:
            self.agent.init_distributed(find_unused_params=True)

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        obs_space = self.obs_space
        if self._static_encoder:
            self._encoder = self.actor_critic.net.visual_encoder
            obs_space = spaces.Dict(
                {
                    "visual_features": spaces.Box(
                        low=np.finfo(np.float32).min,
                        high=np.finfo(np.float32).max,
                        shape=self._encoder.output_shape,
                        dtype=np.float32,
                    ),
                    **obs_space.spaces,
                }
            )

        self._nbuffers = 2 if ppo_cfg.use_double_buffered_sampler else 1

        self.rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            obs_space,
            self.policy_action_space,
            ppo_cfg.hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
            is_double_buffered=ppo_cfg.use_double_buffered_sampler,
            action_shape=action_shape,
            discrete_actions=discrete_actions,
        )
        self.rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        self.rollouts.buffers["observations"][0] = batch

        self.current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        self.running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        # add 
        if self.config.turn_policy:
            gc_hidden_size = self.config.GC.hidden_size
            self.gc_hidden = torch.zeros(ppo_cfg.num_steps + 1, self.envs.num_envs, 2, gc_hidden_size, dtype=torch.float32, device=self.device)
            self.collision = torch.zeros(ppo_cfg.num_steps + 1, self.envs.num_envs, dtype=torch.bool, device=self.device)
            self.rt = torch.eye(4, dtype=torch.float32, device=self.device)[None, None].repeat(ppo_cfg.num_steps + 1, self.envs.num_envs, 1, 1)
            self.rt[0, :, :3, 3] = batch["pointgoal_with_gps_compass"].float()

        self.env_time = 0.0
        self.pth_time = 0.0
        self.t_start = time.time()

    @rank0_only
    @profiling_wrapper.RangeContext("save_checkpoint")
    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def _compute_actions_and_step_envs(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )

        t_sample_action = time.time()

        # sample actions
        with torch.no_grad():
            step_batch = self.rollouts.buffers[
                self.rollouts.current_rollout_step_idxs[buffer_index],
                env_slice,
            ]

            profiling_wrapper.range_push("compute actions")
            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
                gc_info,
            ) = self.actor_critic.act(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
                self.collision[self.rollouts.current_rollout_step_idxs[buffer_index], env_slice],
                self.gc_hidden[self.rollouts.current_rollout_step_idxs[buffer_index], env_slice],
                self.rt[self.rollouts.current_rollout_step_idxs[buffer_index], env_slice],
            )

        # NB: Move actions to CPU.  If CUDA tensors are
        # sent in to env.step(), that will create CUDA contexts
        # in the subprocesses.
        # For backwards compatibility, we also call .item() to convert to
        # an int
        actions = actions.to(device="cpu")
        self.pth_time += time.time() - t_sample_action

        profiling_wrapper.range_pop()  # compute actions

        t_step_env = time.time()

        for index_env, act in zip(
            range(env_slice.start, env_slice.stop), actions.unbind(0)
        ):
            if self.using_velocity_ctrl:
                step_action = action_to_velocity_control(act)
            else:
                step_action = act.item()
            self.envs.async_step_at(index_env, step_action)

        self.env_time += time.time() - t_step_env

        self.rollouts.insert(
            next_recurrent_hidden_states=recurrent_hidden_states,
            actions=actions,
            action_log_probs=actions_log_probs,
            value_preds=values,
            buffer_index=buffer_index,
        )

        # # add
        if self.config.turn_policy:
            self.gc_hidden[self.rollouts.current_rollout_step_idxs[buffer_index] + 1, env_slice] = gc_info["hidden"]

    def _collect_environment_result(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )

        t_step_env = time.time()
        outputs = [
            self.envs.wait_step_at(index_env)
            for index_env in range(env_slice.start, env_slice.stop)
        ]

        observations, rewards_l, dones, infos = [
            list(x) for x in zip(*outputs)
        ]

        self.env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        rewards = torch.tensor(
            rewards_l,
            dtype=torch.float,
            device=self.current_episode_reward.device,
        )
        rewards = rewards.unsqueeze(1)

        not_done_masks = torch.tensor(
            [[not done] for done in dones],
            dtype=torch.bool,
            device=self.current_episode_reward.device,
        )
        done_masks = torch.logical_not(not_done_masks)

        self.current_episode_reward[env_slice] += rewards
        current_ep_reward = self.current_episode_reward[env_slice]
        self.running_episode_stats["reward"][env_slice] += current_ep_reward.where(done_masks, current_ep_reward.new_zeros(()))  # type: ignore
        self.running_episode_stats["count"][env_slice] += done_masks.float()  # type: ignore
        for k, v_k in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v_k,
                dtype=torch.float,
                device=self.current_episode_reward.device,
            ).unsqueeze(1)
            if k not in self.running_episode_stats:
                self.running_episode_stats[k] = torch.zeros_like(
                    self.running_episode_stats["count"]
                )

            self.running_episode_stats[k][env_slice] += v.where(done_masks, v.new_zeros(()))  # type: ignore

        self.current_episode_reward[env_slice].masked_fill_(done_masks, 0.0)

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        self.rollouts.insert(
            next_observations=batch,
            rewards=rewards,
            next_masks=not_done_masks,
            buffer_index=buffer_index,
        )

        if self.config.turn_policy:
            step_batch = self.rollouts.buffers[
                self.rollouts.current_rollout_step_idxs[buffer_index] + 1,
                env_slice,
            ]
            prev_step_batch = self.rollouts.buffers[
                self.rollouts.current_rollout_step_idxs[buffer_index],
                env_slice,
            ]
            with torch.no_grad():
                pred_gps, pose = self.vo_module.infer_rl_batch(
                    step_batch["observations"],
                    prev_step_batch["observations"],
                    step_batch["prev_actions"],
                    return_pose=True,
                )

            collision = [_info["collisions"]["is_collision"] for _info in infos]
            self.collision[self.rollouts.current_rollout_step_idxs[buffer_index] + 1, env_slice] = \
                    torch.tensor(collision, dtype=torch.bool, device=self.current_episode_reward.device)
            
            prev_rt = self.rt[self.rollouts.current_rollout_step_idxs[buffer_index], env_slice]
            eyes = torch.eye(4, device=self.rt.device).float()[None].repeat(6, 1, 1)
            # eyes[:, :3, 3] = prev_step_batch["observations"]["pointgoal_with_gps_compass"]
            # self.rt[self.rollouts.current_rollout_step_idxs[buffer_index], env_slice] = \
            #     torch.where(prev_step_batch["masks"].view(6,1,1), prev_rt, eyes)

            next_rt = torch.matmul(prev_rt, pose)
            next_rt[:, :3, 3] = pred_gps
            eyes[:, :3, 3] = step_batch["observations"]["pointgoal_with_gps_compass"]
            self.rt[self.rollouts.current_rollout_step_idxs[buffer_index] + 1, env_slice] = \
                torch.where(not_done_masks.to(self.device).view(6,1,1), next_rt, eyes)

            # update batch
            pred_gps_with_mask = torch.where(not_done_masks.to(self.device), pred_gps, step_batch["observations"]["pointgoal_with_gps_compass"])
            self.rollouts.buffers["observations"]["pointgoal_with_gps_compass"][
                self.rollouts.current_rollout_step_idxs[buffer_index] + 1,
                env_slice,
            ].copy_(pred_gps_with_mask)

        self.rollouts.advance_rollout(buffer_index)

        self.pth_time += time.time() - t_update_stats

        return env_slice.stop - env_slice.start

    @profiling_wrapper.RangeContext("_collect_rollout_step")
    def _collect_rollout_step(self):
        self._compute_actions_and_step_envs()
        return self._collect_environment_result()

    @profiling_wrapper.RangeContext("_update_agent")
    def _update_agent(self):
        ppo_cfg = self.config.RL.PPO
        t_update_model = time.time()
        with torch.no_grad():
            step_batch = self.rollouts.buffers[
                self.rollouts.current_rollout_step_idx
            ]

            next_value = self.actor_critic.get_value(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
                self.collision[self.rollouts.current_rollout_step_idx],
                self.gc_hidden[self.rollouts.current_rollout_step_idx],
                self.rt[self.rollouts.current_rollout_step_idx],
            )

        self.rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        self.agent.train()

        value_loss, action_loss, dist_entropy = self.agent.update(
            self.rollouts, self.collision, self.gc_hidden, self.rt
        )

        #
        if self.config.turn_policy:
            self.gc_hidden[0] = self.gc_hidden[self.rollouts.current_rollout_step_idx]
            self.collision[0] = self.collision[self.rollouts.current_rollout_step_idx]
            self.rt[0] = self.rt[self.rollouts.current_rollout_step_idx]
        
        self.rollouts.after_update()
        self.pth_time += time.time() - t_update_model

        return (
            value_loss,
            action_loss,
            dist_entropy,
        )

    def _coalesce_post_step(
        self, losses: Dict[str, float], count_steps_delta: int
    ) -> Dict[str, float]:
        stats_ordering = sorted(self.running_episode_stats.keys())
        stats = torch.stack(
            [self.running_episode_stats[k] for k in stats_ordering], 0
        )

        stats = self._all_reduce(stats)

        for i, k in enumerate(stats_ordering):
            self.window_episode_stats[k].append(stats[i])

        if self._is_distributed:
            loss_name_ordering = sorted(losses.keys())
            stats = torch.tensor(
                [losses[k] for k in loss_name_ordering] + [count_steps_delta],
                device="cpu",
                dtype=torch.float32,
            )
            stats = self._all_reduce(stats)
            count_steps_delta = int(stats[-1].item())
            stats /= torch.distributed.get_world_size()

            losses = {
                k: stats[i].item() for i, k in enumerate(loss_name_ordering)
            }

        if self._is_distributed and rank0_only():
            self.num_rollouts_done_store.set("num_done", "0")

        self.num_steps_done += count_steps_delta

        return losses

    @rank0_only
    def _training_log(
        self, writer, losses: Dict[str, float], prev_time: int = 0
    ):
        deltas = {
            k: (
                (v[-1] - v[0]).sum().item()
                if len(v) > 1
                else v[0].sum().item()
            )
            for k, v in self.window_episode_stats.items()
        }
        deltas["count"] = max(deltas["count"], 1.0)

        writer.add_scalar(
            "reward",
            deltas["reward"] / deltas["count"],
            self.num_steps_done,
        )

        # Check to see if there are any metrics
        # that haven't been logged yet
        metrics = {
            k: v / deltas["count"]
            for k, v in deltas.items()
            if k not in {"reward", "count"}
        }
        if len(metrics) > 0:
            writer.add_scalars("metrics", metrics, self.num_steps_done)

        writer.add_scalars(
            "losses",
            losses,
            self.num_steps_done,
        )

        # log stats
        if self.num_updates_done % self.config.LOG_INTERVAL == 0:
            logger.info(
                "update: {}\tfps: {:.3f}\t".format(
                    self.num_updates_done,
                    self.num_steps_done
                    / ((time.time() - self.t_start) + prev_time),
                )
            )

            logger.info(
                "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                "frames: {}".format(
                    self.num_updates_done,
                    self.env_time,
                    self.pth_time,
                    self.num_steps_done,
                )
            )

            logger.info(
                "Average window size: {}  {}".format(
                    len(self.window_episode_stats["count"]),
                    "  ".join(
                        "{}: {:.3f}".format(k, v / deltas["count"])
                        for k, v in deltas.items()
                        if k != "count"
                    ),
                )
            )

    def should_end_early(self, rollout_step) -> bool:
        if not self._is_distributed:
            return False
        # This is where the preemption of workers happens.  If a
        # worker detects it will be a straggler, it preempts itself!
        return (
            rollout_step
            >= self.config.RL.PPO.num_steps * self.SHORT_ROLLOUT_THRESHOLD
        ) and int(self.num_rollouts_done_store.get("num_done")) >= (
            self.config.RL.DDPPO.sync_frac * torch.distributed.get_world_size()
        )

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training DD/PPO.

        Returns:
            None
        """

        self._init_train()

        count_checkpoints = 0
        prev_time = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: 1 - self.percent_done(),
        )

        if self.config.use_pre_trained:
            params_dict = torch.load(self.config.pretrain_path, map_location="cpu")["state_dict"]
            # missing_keys, unexpected_keys = self.agent.load_state_dict(params_dict, strict=False)
            needed_paramed = OrderedDict()
            for _name, _param in self.agent.named_parameters():
                if _name in params_dict:
                    if _param.shape == params_dict[_name].shape:
                        needed_paramed[_name] = params_dict[_name]
            missing_keys, unexpected_keys = self.agent.load_state_dict(needed_paramed, strict=False)

        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            self.agent.load_state_dict(resume_state["state_dict"])
            self.agent.optimizer.load_state_dict(resume_state["optim_state"])
            lr_scheduler.load_state_dict(resume_state["lr_sched_state"])

            requeue_stats = resume_state["requeue_stats"]
            self.env_time = requeue_stats["env_time"]
            self.pth_time = requeue_stats["pth_time"]
            self.num_steps_done = requeue_stats["num_steps_done"]
            self.num_updates_done = requeue_stats["num_updates_done"]
            self._last_checkpoint_percent = requeue_stats[
                "_last_checkpoint_percent"
            ]
            count_checkpoints = requeue_stats["count_checkpoints"]
            prev_time = requeue_stats["prev_time"]

            self.running_episode_stats = requeue_stats["running_episode_stats"]
            self.window_episode_stats.update(
                requeue_stats["window_episode_stats"]
            )

        ppo_cfg = self.config.RL.PPO

        # load VO model
        if self.config.turn_policy:
            with open(self.config.VO.vo_config_path, 'r') as f:
                vo_cfgs = yaml.load(f, Loader=yaml.SafeLoader)
            self.vo_module = VO_Module(vo_cfgs, self.config)
            self.vo_module.to(self.device)
            self.vo_module.eval()

        with (
            TensorboardWriter(
                self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
            )
            if rank0_only()
            else contextlib.suppress()
        ) as writer:
            while not self.is_done():
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * (
                        1 - self.percent_done()
                    )

                if rank0_only() and self._should_save_resume_state():
                    requeue_stats = dict(
                        env_time=self.env_time,
                        pth_time=self.pth_time,
                        count_checkpoints=count_checkpoints,
                        num_steps_done=self.num_steps_done,
                        num_updates_done=self.num_updates_done,
                        _last_checkpoint_percent=self._last_checkpoint_percent,
                        prev_time=(time.time() - self.t_start) + prev_time,
                        running_episode_stats=self.running_episode_stats,
                        window_episode_stats=dict(self.window_episode_stats),
                    )

                    save_resume_state(
                        dict(
                            state_dict=self.agent.state_dict(),
                            optim_state=self.agent.optimizer.state_dict(),
                            lr_sched_state=lr_scheduler.state_dict(),
                            config=self.config,
                            requeue_stats=requeue_stats,
                        ),
                        self.config,
                    )

                if EXIT.is_set():
                    profiling_wrapper.range_pop()  # train update

                    self.envs.close()

                    requeue_job()

                    return

                self.agent.eval()
                count_steps_delta = 0
                profiling_wrapper.range_push("rollouts loop")

                profiling_wrapper.range_push("_collect_rollout_step")
                for buffer_index in range(self._nbuffers):
                    self._compute_actions_and_step_envs(buffer_index)

                for step in range(ppo_cfg.num_steps):
                    is_last_step = (
                        self.should_end_early(step + 1)
                        or (step + 1) == ppo_cfg.num_steps
                    )

                    for buffer_index in range(self._nbuffers):
                        count_steps_delta += self._collect_environment_result(
                            buffer_index
                        )

                        if (buffer_index + 1) == self._nbuffers:
                            profiling_wrapper.range_pop()  # _collect_rollout_step

                        if not is_last_step:
                            if (buffer_index + 1) == self._nbuffers:
                                profiling_wrapper.range_push(
                                    "_collect_rollout_step"
                                )

                            self._compute_actions_and_step_envs(buffer_index)

                    if is_last_step:
                        break

                profiling_wrapper.range_pop()  # rollouts loop

                if self._is_distributed:
                    self.num_rollouts_done_store.add("num_done", 1)

                (
                    value_loss,
                    action_loss,
                    dist_entropy,
                ) = self._update_agent()

                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()  # type: ignore

                self.num_updates_done += 1
                losses = self._coalesce_post_step(
                    dict(value_loss=value_loss, action_loss=action_loss),
                    count_steps_delta,
                )

                self._training_log(writer, losses, prev_time)

                # checkpoint model
                if rank0_only() and self.should_checkpoint():
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        dict(
                            step=self.num_steps_done,
                            wall_time=(time.time() - self.t_start) + prev_time,
                        ),
                    )
                    count_checkpoints += 1

                profiling_wrapper.range_pop()  # train update

            self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        if config.VERBOSE:
            logger.info(f"env config: {config}")

        self._init_envs(config)

        if self.using_velocity_ctrl:
            self.policy_action_space = self.envs.action_spaces[0][
                "VELOCITY_CONTROL"
            ]
            action_shape = (2,)
            action_type = torch.float
        else:
            self.policy_action_space = self.envs.action_spaces[0]
            action_shape = (1,)
            action_type = torch.long

        self._setup_actor_critic_agent(ppo_cfg)

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        observations = self.envs.reset()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device="cpu"
        )

        test_recurrent_hidden_states = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            self.actor_critic.net.num_recurrent_layers,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            *action_shape,
            device=self.device,
            dtype=action_type,
        )
        not_done_masks = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_ENVIRONMENTS)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        # load VO model
        if self.config.VO.USE_VO_MODEL:
            if self.config.VO.method == "ours":
                with open(self.config.VO.vo_config_path, 'r') as f:
                    vo_cfgs = yaml.load(f, Loader=yaml.SafeLoader)
                vo_module = VO_Module(vo_cfgs, self.config)
                vo_module = vo_module.to(self.device)
                vo_module.eval()

            rollout_gps_traj = batch["pointgoal_with_gps_compass"].cpu().numpy().reshape([self.envs.num_envs,1,3]).tolist()
            rollout_vo_traj = batch["pointgoal_with_gps_compass"].cpu().numpy().reshape([self.envs.num_envs,1,3]).tolist()
            rollout_action = [[] for _ in range(self.envs.num_envs)]

            # all_traj
            vo_traj_infos = collections.OrderedDict()
        
        # if self.config.GC.USE_GC_EVAL:
        test_gc_recurrent_hidden_states = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            2,
            config.GC.hidden_size,
            device=self.device,
        )
        test_gc_collision = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            dtype=torch.bool,
            device=self.device,
        )
        test_rt = torch.eye(4, dtype=torch.float32, device=self.device)[None].repeat(self.config.NUM_ENVIRONMENTS, 1, 1)
        test_rt[:, :3, 3] = batch["pointgoal_with_gps_compass"].float()
        prev_rt = torch.eye(4, dtype=torch.float32, device=self.device)[None].repeat(self.config.NUM_ENVIRONMENTS, 1, 1)

        #
        pbar = tqdm.tqdm(total=number_of_eval_episodes)
        self.actor_critic.eval()
        while (
            len(stats_episodes) < number_of_eval_episodes
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                    gc_info,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    test_gc_collision,
                    test_gc_recurrent_hidden_states,
                    test_rt,
                    deterministic=False,
                )

                prev_actions.copy_(actions)  # type: ignore
                test_gc_recurrent_hidden_states = gc_info["hidden"]
                prev_rt.copy_(test_rt)
                if self.config.VO.USE_VO_MODEL:
                    prev_batch = copy.deepcopy(batch)
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            # For backwards compatibility, we also call .item() to convert to
            # an int
            if self.using_velocity_ctrl:
                step_data = [
                    action_to_velocity_control(a)
                    for a in actions.to(device="cpu")
                ]
            else:
                step_data = [a.item() for a in actions.to(device="cpu")]

            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            #
            _collision = [_info["collisions"]["is_collision"] for _info in infos]
            test_gc_collision = torch.tensor(_collision, dtype=torch.bool, device=self.device)
            #
            batch = batch_obs(
                observations,
                device=self.device,
                cache=self._obs_batching_cache,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)
                    
            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = {}
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            tb_writer=writer,
                        )

                        rgb_frames[i] = []

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, infos[i]
                    )
                    rgb_frames[i].append(frame)

            ## start vo
            if self.config.VO.USE_VO_MODEL:
                with torch.no_grad():
                    if self.config.VO.method == "ours":
                        pred_gps, pose = vo_module.infer_rl_batch(
                            batch,
                            prev_batch,
                            prev_actions,
                            True,
                        )
                    else:
                        pred_gps = vo_module.infer(batch,prev_batch)
                
                current_batch = batch["pointgoal_with_gps_compass"].shape[0]
                eyes = torch.eye(4, device=self.device).float()[None].repeat(current_batch, 1, 1)
                next_rt = torch.matmul(prev_rt, pose)
                next_rt[:, :3, 3] = pred_gps
                eyes[:, :3, 3] = batch["pointgoal_with_gps_compass"]
                test_rt = torch.where(not_done_masks.to(self.device).view(current_batch,1,1), next_rt, eyes)
                # for env_id in range(n_envs):
                #     # episode ended
                #     if not not_done_masks[env_id].item():
                #         test_rt[env_id] = eyes
                #     else:
                #         test_rt[env_id] = next_rt

                for env_id in range(n_envs):
                    # episode ended
                    if not not_done_masks[env_id].item():
                        ndarray_rollout_gps_traj = np.array(rollout_gps_traj[env_id])
                        ndarray_rollout_vo_traj = np.array(rollout_vo_traj[env_id])
                        ndarray_rollout_error = np.abs(ndarray_rollout_gps_traj-ndarray_rollout_vo_traj)

                        rollout_action[env_id].append(prev_actions[env_id].cpu().item())
                        ndarray_rollout_action = np.array(rollout_action[env_id])
                        vo_traj_infos[
                            (
                                current_episodes[env_id].scene_id,
                                current_episodes[env_id].episode_id,
                            )
                        ] = [ ndarray_rollout_gps_traj, ndarray_rollout_vo_traj, ndarray_rollout_error, ndarray_rollout_action ]

                        rollout_gps_traj[env_id] = batch["pointgoal_with_gps_compass"][env_id].cpu().numpy().reshape([1,3]).tolist()
                        rollout_vo_traj[env_id] = batch["pointgoal_with_gps_compass"][env_id].cpu().numpy().reshape([1,3]).tolist()
                        rollout_action[env_id] = []
                    # episode continues
                    else:
                        # collect data
                        rollout_gps_traj[env_id].append(batch["pointgoal_with_gps_compass"][env_id].cpu().numpy().tolist())
                        rollout_vo_traj[env_id].append(pred_gps[env_id].cpu().numpy().tolist())
                        rollout_action[env_id].append(prev_actions[env_id].cpu().item())
                        
                        # update batch
                        batch["pointgoal_with_gps_compass"][env_id].copy_(pred_gps[env_id].double())
            ## end vo

            not_done_masks = not_done_masks.to(device=self.device)
            if len(envs_to_pause) > 0:
                state_index = list(range(self.envs.num_envs))
                for idx in reversed(envs_to_pause):
                    state_index.pop(idx)

                # indexing along the batch dimensions
                test_gc_recurrent_hidden_states = test_gc_recurrent_hidden_states[state_index]
                test_gc_collision = test_gc_collision[state_index]
                prev_rt = prev_rt[state_index]
                test_rt = test_rt[state_index]

            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values())
                / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalars(
            "eval_reward",
            {"average reward": aggregated_stats["reward"]},
            step_id,
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        if len(metrics) > 0:
            writer.add_scalars("eval_metrics", metrics, step_id)

        self.envs.close()

        # if self.config.VO.USE_VO_MODEL:
        #     pkl_path = os.path.join(os.path.dirname(self.config.VO.vo_config_path), "rl_vo_infos.pkl")
        #     with open(pkl_path, 'wb') as f:
        #         pickle.dump(vo_traj_infos, f)
            
        #     pkl_path = os.path.join(os.path.dirname(self.config.VO.vo_config_path), "rl_infos.pkl")
        #     with open(pkl_path, 'wb') as f:
        #         pickle.dump(stats_episodes, f)
            
        #     rl_eval_path = os.path.join(os.path.dirname(self.config.VO.vo_config_path), "rl_eval.txt")
        #     with open(rl_eval_path, 'a') as f:
        #         f.write(f"rl_checkpoints {checkpoint_index}, vo_checcckpoints {self.config.VO.vo_reuse_id}\n")
        #         for k, v in aggregated_stats.items():
        #             f.write(f"Average episode {k}: {v:.4f} \n")
            
        rl_eval_path = os.path.join(os.path.dirname(self.config.LOG_FILE), "eval.log")
        with open(rl_eval_path, 'a') as f:
            f.write(f"checkpoints {checkpoint_index}\n")
            for k, v in aggregated_stats.items():
                f.write(f"Average episode {k}: {v:.4f} \n")
