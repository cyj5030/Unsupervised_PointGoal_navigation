#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn as nn
from torch import optim as optim
import torch.nn.functional as F

from habitat.utils import profiling_wrapper
from habitat_baselines.common.rollout_storage import RolloutStorage

from models.rl.ppo.policy import Policy

EPS_PPO = 1e-5


class PPO(nn.Module):
    def __init__(
        self,
        actor_critic: Policy,
        clip_param: float,
        ppo_epoch: int,
        num_mini_batch: int,
        value_loss_coef: float,
        entropy_coef: float,
        lr: Optional[float] = None,
        eps: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        use_clipped_value_loss: bool = True,
        use_normalized_advantage: bool = True,
    ) -> None:

        super().__init__()

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(
            list(filter(lambda p: p.requires_grad, actor_critic.parameters())),
            lr=lr,
            eps=eps,
        )
        self.device = next(actor_critic.parameters()).device
        self.use_normalized_advantage = use_normalized_advantage

    def forward(self, *x):
        raise NotImplementedError

    def get_advantages(self, rollouts: RolloutStorage) -> Tensor:
        advantages = (
            rollouts.buffers["returns"][:-1]
            - rollouts.buffers["value_preds"][:-1]
        )
        if not self.use_normalized_advantage:
            return advantages

        return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)

    def update(self, rollouts: RolloutStorage, collision, gc_hidden, rt) -> Tuple[float, float, float]:
        advantages = self.get_advantages(rollouts)

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0

        for _e in range(self.ppo_epoch):
            profiling_wrapper.range_push("PPO.update epoch")
            data_generator = rollouts.recurrent_generator_v2(
                advantages, self.num_mini_batch, gc_hidden, collision, rt
            )

            for batch, _gc_hidden, _collision, _rt in data_generator:
                (
                    values,
                    action_log_probs,
                    dist_entropy,
                    _,
                    gc_info,
                ) = self._evaluate_actions(
                    batch["observations"],
                    batch["recurrent_hidden_states"],
                    batch["prev_actions"],
                    batch["masks"],
                    batch["actions"],
                    _collision,
                    _gc_hidden,
                    _rt,
                )

                ratio = torch.exp(action_log_probs - batch["action_log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    * batch["advantages"]
                )
                action_loss = -(torch.min(surr1, surr2).mean())

                if self.use_clipped_value_loss:
                    value_pred_clipped = batch["value_preds"] + (
                        values - batch["value_preds"]
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - batch["returns"]).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - batch["returns"]
                    ).pow(2)
                    value_loss = 0.5 * torch.max(
                        value_losses, value_losses_clipped
                    )
                else:
                    value_loss = 0.5 * (batch["returns"] - values).pow(2)

                value_loss = value_loss.mean()
                dist_entropy = dist_entropy.mean()

                self.optimizer.zero_grad()
                total_loss = (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                )
                if self.actor_critic.net.config.GC.hd_loss:
                    hd_loss = self.calc_hd_loss(gc_info["rt_seqs"], gc_info["hd_seqs"])
                    total_loss += self.actor_critic.net.config.GC.coeff_hd * hd_loss
                if self.actor_critic.net.config.GC.pc_loss:
                    pc_loss = self.calc_pc_loss(gc_info["rt_seqs"], gc_info["pc_seqs"])
                    total_loss += self.actor_critic.net.config.GC.coeff_pc * pc_loss

                self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)

                self.before_step()
                self.optimizer.step()
                self.after_step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

            profiling_wrapper.range_pop()  # PPO.update epoch

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def _evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action, collision, gc_hidden, rt
    ):
        r"""Internal method that calls Policy.evaluate_actions.  This is used instead of calling
        that directly so that that call can be overrided with inheritance
        """
        return self.actor_critic.evaluate_actions(
            observations, rnn_hidden_states, prev_actions, masks, action, collision, gc_hidden, rt
        )

    def before_backward(self, loss: Tensor) -> None:
        pass

    def after_backward(self, loss: Tensor) -> None:
        pass

    def before_step(self) -> None:
        nn.utils.clip_grad_norm_(
            self.actor_critic.parameters(), self.max_grad_norm
        )

    def after_step(self) -> None:
        pass
    
    def calc_hd_loss(self, gps, pred):
        lens_unpacked = gps[1]
        # x, y, z = gps[0].split(1, dim=-1)
        # hd = torch.atan2(z, x)
        hd = gps[0][:, :, 0:1]

        loss = 0
        min_len = 0
        for i_batch, length in enumerate(lens_unpacked):
            if length > min_len:
                seq_label = self.actor_critic.net.gc_lstm.hd_ensemble(hd[:length, i_batch, :].view(length, 1)).view(length, -1) # 384 12
                seq_pred = pred[0][:length, i_batch, :]
                # loss += self.nll_loss(seq_pred[:-1], seq_label[1:])
                loss += self.nll_loss(seq_pred, seq_label)
        loss = loss / (lens_unpacked > min_len).float().sum()
        return loss
    
    def calc_pc_loss(self, gps, pred):
        lens_unpacked = gps[1]
        # x, y, z = gps[0].split(1, dim=-1)
        # pc = torch.cat([x, z], -1)
        pc = gps[0][:, :, 1:3]

        loss = 0
        min_len = 0
        for i_batch, length in enumerate(lens_unpacked):
            if length > min_len:
                seq_label = self.actor_critic.net.gc_lstm.pc_ensemble(pc[:length, i_batch, :].view(length, 1, 2)).view(length, -1) # 384 N
                seq_pred = pred[0][:length, i_batch, :]
                # loss = self.nll_loss(seq_pred[:-1], seq_label[1:])
                loss += self.nll_loss(seq_pred, seq_label)
        loss = loss / (lens_unpacked > min_len).float().sum()
        return loss
    
    def nll_loss(self, logits, labels):
        aN = logits.shape[-1]
        eps = 1e-6
        logits = logits.clamp(eps,1.0-eps)

        cross_entropy = F.nll_loss(logits.log().view(-1, aN), torch.argmax(labels, 1).view(-1), reduction='none').mean()
        return cross_entropy