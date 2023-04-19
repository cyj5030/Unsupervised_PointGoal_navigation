import os
import torch
import yaml
import copy
import numpy as np
import collections
import tqdm

from habitat import logger
from torch.optim.lr_scheduler import LambdaLR

from models.rl.ppo.ppo_trainer import PPOTrainer
from models.vo.vo_module import VO_Module

from habitat_baselines.utils.common import (
    ObservationBatchingCache,
    action_to_velocity_control,
    batch_obs
)

from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)

from typing import Dict, List, Optional, Tuple, Union, Any
# from habitat.utils.visualizations import maps
from scripts import maps
import imageio
from habitat.utils.visualizations.utils import images_to_video

def observations_to_image(observation: Dict, info: Dict) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    render_obs_images: List[np.ndarray] = []
    for sensor_name in observation:
        if "rgb" in sensor_name:
            rgb = observation[sensor_name]
            if not isinstance(rgb, np.ndarray):
                rgb = rgb.cpu().numpy()

            render_obs_images.append(rgb)
        elif "depth" in sensor_name:
            depth_map = observation[sensor_name].squeeze() * 255.0
            if not isinstance(depth_map, np.ndarray):
                depth_map = depth_map.cpu().numpy()

            depth_map = depth_map.astype(np.uint8)
            depth_map = np.stack([depth_map for _ in range(3)], axis=2)
            render_obs_images.append(depth_map)

    assert (
        len(render_obs_images) > 0
    ), "Expected at least one visual sensor enabled."

    shapes_are_equal = len(set(x.shape for x in render_obs_images)) == 1
    render_frame = np.concatenate(render_obs_images, axis=1)

    # draw collision
    if "collisions" in info and info["collisions"]["is_collision"]:
        render_frame = draw_collision(render_frame)

    if "top_down_map" in info:
        top_down_map = maps.colorize_draw_agent_and_fit_to_height(
            info["top_down_map"], render_frame.shape[0]
        )
        render_frame = np.concatenate((render_frame, top_down_map), axis=1)
    return render_frame

def generate_video(
    video_option: List[str],
    video_dir: Optional[str],
    images: List[np.ndarray],
    episode_id: Union[int, str],
    checkpoint_idx: int,
    metrics: Dict[str, float],
    scene_id,
    fps: int = 10,
    verbose: bool = True,
) -> None:
    r"""Generate video according to specified information.

    Args:
        video_option: string list of "tensorboard" or "disk" or both.
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        episode_id: episode id for video naming.
        checkpoint_idx: checkpoint index for video naming.
        metric_name: name of the performance metric, e.g. "spl".
        metric_value: value of metric.
        tb_writer: tensorboard writer object for uploading video.
        fps: fps for generated video.
    Returns:
        None
    """
    if len(images) < 1:
        return

    metric_strs = []
    for k, v in metrics.items():
        metric_strs.append(f"{k}={v:.2f}")

    video_name = f"scene={scene_id}-episode={episode_id}-ckpt={checkpoint_idx}-" + "-".join(
        metric_strs
    )
    if "disk" in video_option:
        assert video_dir is not None
        images_to_video(images, video_dir, video_name, verbose=verbose)

def draw_collision(view: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    r"""Draw translucent red strips on the border of input view to indicate
    a collision has taken place.
    Args:
        view: input view of size HxWx3 in RGB order.
        alpha: Opacity of red collision strip. 1 is completely non-transparent.
    Returns:
        A view with collision effect drawn.
    """
    strip_width = view.shape[0] // 20
    mask = np.ones(view.shape)
    mask[strip_width:-strip_width, strip_width:-strip_width] = 0
    mask = mask == 1
    view[mask] = (alpha * np.array([255, 0, 0]) + (1.0 - alpha) * view)[mask]
    return view

class PPOEval(PPOTrainer):
    def __init__(self, config=None):
        super().__init__(config)
        # self.device = "cuda:0"
    def eval(self) -> None:
        checkpoint_path = self.config.EVAL_CKPT_PATH_DIR
        checkpoint_index = int(os.path.basename(checkpoint_path).split(".")[-2])
        # Map location CPU is almost always better than mapping to a CUDA device.
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

        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.TORCH_GPU_ID)
            torch.cuda.set_device(self.device)
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
                            scene_id=os.path.basename(current_episodes[i].scene_id),
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

        self.envs.close()
