#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc

import torch
from gym import spaces
from torch import nn as nn

from habitat.config import Config
from habitat.tasks.nav.nav import (
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
)

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.utils.common import CategoricalNet, GaussianNet

from models.networks.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from models.networks.resnet import ResNetEncoder
from models.gc.gc_model import GC_Module

import yaml
import os
import numpy as np
# from models.vo.vo_utils import bilinear_interp, disp_to_depth

class Policy(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, net, dim_actions, policy_config=None):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        if policy_config is None:
            self.action_distribution_type = "categorical"
        else:
            self.action_distribution_type = (
                policy_config.action_distribution_type
            )

        if self.action_distribution_type == "categorical":
            self.action_distribution = CategoricalNet(
                self.net.output_size, self.dim_actions
            )
        elif self.action_distribution_type == "gaussian":
            self.action_distribution = GaussianNet(
                self.net.output_size,
                self.dim_actions,
                policy_config.ACTION_DIST,
            )
        else:
            ValueError(
                f"Action distribution {self.action_distribution_type}"
                "not supported."
            )

        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        collision,
        gc_hidden,
        rt,
        deterministic=False,
    ):
        features, rnn_hidden_states, gc_info = self.net(
            observations, rnn_hidden_states, prev_actions, masks, collision, gc_hidden, rt
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            if self.action_distribution_type == "categorical":
                action = distribution.mode()
            elif self.action_distribution_type == "gaussian":
                action = distribution.mean
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states, gc_info

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks, collision, gc_hidden, rt):
        features, _, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks, collision, gc_hidden, rt
        )
        return self.critic(features)

    def evaluate_actions(self, observations, rnn_hidden_states, prev_actions, masks, action, collision, gc_hidden, rt):
        features, rnn_hidden_states, gc_info = self.net(
            observations, rnn_hidden_states, prev_actions, masks, collision, gc_hidden, rt
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states, gc_info

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config, observation_space, action_space):
        pass


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


# @baseline_registry.register_policy
class PointNavBaselinePolicy(Policy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        config,
    ):
        super().__init__(
            PointNavBaselineNet(  # type: ignore
                observation_space=observation_space,
                dim_actions = action_space.n,
                config=config,
            ),
            action_space.n,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: spaces.Dict, action_space
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            config=config,
        )


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass

class Flatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1).contiguous()

class PointNavBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        dim_actions: int,
        config,
    ):
        super().__init__()
        self.config = config
        hidden_size = config.RL.PPO.hidden_size
        self.num_steps = config.RL.PPO.num_steps
        self.dim_actions = dim_actions

        self._hidden_size = hidden_size

        self.visual_encoder = ResNetEncoder(observation_space)
        if not self.visual_encoder.is_blind:
            self.visual_fc = nn.Sequential(
                Flatten(),
                nn.Linear(np.prod(self.visual_encoder.output_shape), hidden_size),
                nn.ReLU(True),
            )

        # self.prev_action_embedding = nn.Embedding(dim_actions + 1, 32)
        self.gc_lstm = GC_Module(dim_actions, config.GC.hidden_size, config.GC.out_size)
        self.tgt_embeding = nn.Linear(3, 32)

        rnn_input_dims = (0 if self.is_blind else hidden_size) + 32 + config.GC.out_size
        self.state_encoder = build_rnn_state_encoder(
            rnn_input_dims,
            self._hidden_size,
            rnn_type='LSTM',
            num_layers=2,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks, collision, gc_hidden, rt):

        x = []
        ## visual embeding 
        if not self.is_blind:
            visual_feats = self.visual_encoder(observations)
            visual_feats = self.visual_fc(visual_feats).detach()
            x.append(visual_feats)

        target_rep = cartesian2polar(observations["pointgoal_with_gps_compass"]).float()
        target_encoding = torch.stack([target_rep[:, 0],
                torch.cos(-target_rep[:, 1]),
                torch.sin(-target_rep[:, 1]),], -1,)
        x.append(self.tgt_embeding(target_encoding))
        
        ## previous action embeding
        # action_embed = self.prev_action_embedding(((prev_actions.float() + 1) * masks).long().squeeze(dim=-1))
        # x.append(action_embed)

        gc_info = \
            self.gc_lstm(prev_actions, masks, collision, gc_hidden, observations["pointgoal_with_gps_compass"], rt)
        x.append(gc_info["compressed_feature"].detach())
        

        ## merge and feed to rnn
        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(
            x, rnn_hidden_states, masks
        )

        return x, rnn_hidden_states, gc_info

def cartesian2polar(xyz):
    x, z = xyz[:, 0], xyz[:, 2]
    rho = torch.sqrt(x ** 2 + z ** 2)
    phi = torch.atan2(z, x)
    polar = torch.stack([rho, phi], 1)
    return polar