import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import torchvision
import numpy as np

from torch.nn.utils.rnn import PackedSequence
from typing import Tuple
from models.networks.rnn_state_encoder import _build_pack_info_from_dones, build_rnn_out_from_seq

from models.vo.lie_algebra import SO3_CUDA

def build_rnn_inputs(
    x: torch.Tensor, not_dones: torch.Tensor, rnn_states: torch.Tensor, init_hidden_states: torch.Tensor,
) -> Tuple[
    PackedSequence, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    N = rnn_states.size(1)
    T = x.size(0) // N
    dones = torch.logical_not(not_dones)

    (
        select_inds,
        batch_sizes,
        episode_starts,
        rnn_state_batch_inds,
        last_episode_in_batch_mask,
    ) = _build_pack_info_from_dones(dones.detach().to(device="cpu"), T)

    select_inds = select_inds.to(device=x.device)
    episode_starts = episode_starts.to(device=x.device)
    rnn_state_batch_inds = rnn_state_batch_inds.to(device=x.device)
    last_episode_in_batch_mask = last_episode_in_batch_mask.to(device=x.device)

    x_seq = PackedSequence(
        x.index_select(0, select_inds), batch_sizes, None, None
    )

    # Just select the rnn_states by batch index, the masking bellow will set things
    # to zero in the correct locations
    rnn_states = rnn_states.index_select(1, rnn_state_batch_inds)
    # Now zero things out in the correct locations
    rnn_states = torch.where(
        not_dones.view(1, -1, 1).index_select(1, episode_starts),
        rnn_states,
        init_hidden_states.index_select(1, episode_starts),
    )

    return (
        x_seq,
        rnn_states,
        select_inds,
        rnn_state_batch_inds,
        last_episode_in_batch_mask,
        batch_sizes,
    )

class GC_Module(nn.Module):
    PI = 3.1415926
    def __init__(
        self,
        dim_actions: int,
        hidden_size: int,
        out_dims: int,
    ):
        super().__init__()
        action_dim = 32
        collision_dim = 16
        self.action_embeding = nn.Embedding(dim_actions + 1, action_dim)
        self.collision_embeding = nn.Embedding(2, collision_dim)

        self.init_sample()

        self.drop = nn.Dropout()
        self.rnn = nn.LSTM(
            input_size=action_dim + collision_dim,
            hidden_size=hidden_size,
            num_layers=1,
        )
        self.h_hidden_size = int(hidden_size / 2)
        self.fc1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_position = nn.Sequential(nn.Linear(hidden_size, self.sp_pc, bias=False), nn.Softmax(-1))
        self.fc_hd = nn.Sequential(nn.Linear(hidden_size, self.sp_hd, bias=False), nn.Softmax(-1))
        self.fc_out = nn.Linear(hidden_size, out_dims, bias=False)

        self.fc_pc_init = nn.Linear(self.sp_pc, hidden_size)
        self.fc_hd_init = nn.Linear(self.sp_hd, hidden_size)

        self.layer_init()

    def layer_init(self):
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def init_sample(self):
        # max gps xyz [21.469297   1.4000001 20.189959 ]
        # min gps xyz [-24.27427  -0.65236956 -23.853443  ]
        # we use -25, 25
        self.width = 25

        self.sp_hd = 12
        self.k = 20
        self.samples_hd = (torch.rand((self.sp_hd, 1)).clamp(1e-7) * 2 - 1) * self.PI # N 1

        self.sigma = 0.5
        xy_nums = int(self.width * 2 / (self.sigma * 5))
        self.sp_pc = xy_nums * xy_nums
        xy = torch.linspace(-self.width, self.width, xy_nums + 2)[1:-1]
        ys = xy.view(1, xy_nums, 1).expand(-1, -1, xy_nums)
        xs = xy.view(1, 1, xy_nums).expand(-1, xy_nums, -1)
        self.samples = torch.cat([xs, -ys], 0).view(2, -1).permute(1, 0) # N 2

    def pc_ensemble(self, cell):
        batch, seq_length, _ = cell.shape

        cell = cell.view(batch, seq_length, 1, 2)
        samples = self.samples.view(1, 1, self.sp_pc, 2).to(cell.device)

        diff = cell - samples
        logp = -0.5* diff.pow(2).sum(-1, False) / (self.sigma * self.sigma)
        # log_posteriors = logp# - logp.exp().sum(-1, True).log()
        activations = F.softmax(logp, 2)
        return activations
    
    def hd_ensemble(self, cell):
        batch, seq_length = cell.shape

        cell = cell.view(batch, seq_length, 1)
        act = self.samples_hd.view(1, 1, self.sp_hd).to(cell.device).float()

        logp = torch.cos((cell - act)) * self.k
        activations = F.softmax(logp, 2)
        return activations
    
    def init_hidden(self, masks, gps, rt):
        gps = gps.float()
        batch = rt.shape[0]
        x, y, z = gps.split(1, dim=-1)
        # hd = torch.atan2(z, x)
        # pc = torch.cat([x, z], 1)
        self.SO3 = SO3_CUDA(gps.device)
        hd = self.SO3.log(rt[:, :3, :3])[:, 1:2]
        pc = torch.stack([rt[:, 0, 3], rt[:, 2, 3]], 1)

        pc_a = self.pc_ensemble(pc.view(batch, 1, 2)).permute(1, 0, 2)
        hd_a = self.hd_ensemble(hd.view(batch, 1)).permute(1, 0, 2)

        hidden = []
        hidden.append(self.fc_pc_init(pc_a))
        hidden.append(self.fc_hd_init(hd_a))
        return hidden, hd, pc
    
    def input_embedding(self, action, collision, masks):
        action_embed = self.action_embeding(((action.float() + 1) * masks).long().squeeze(dim=-1))
        collision_embed = self.collision_embeding(collision.long())
        embed = torch.cat([action_embed, collision_embed], 1)
        return embed
    
    def output_branch(self, x):
        x = self.fc1(x)
        pc = self.fc_position(x)
        hd = self.fc_hd(x)
        # pc = self.fc_position(x[..., :self.h_hidden_size])
        # hd = self.fc_hd(x[..., :self.h_hidden_size])
        out = self.fc_out(x)
        return x, pc, hd, out

    def pack_hidden(self, hidden_states: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return torch.cat(hidden_states, 0)

    def unpack_hidden(self, hidden_states) -> Tuple[torch.Tensor, torch.Tensor]:
        lstm_states = torch.chunk(hidden_states, 2, 0)
        return (lstm_states[0], lstm_states[1])

    def one_seq(self, action, gps, collision): # without rl
        device = gps.device
        # action, gps, collision = action[0], gps[0], collision[0]

        hd = torch.tensor([0.])[None].to(device)
        pc = gps[0, 0:1, [0, 2]]
        pc_a = self.pc_ensemble(pc.view(1, 1, 2)).permute(1, 0, 2)
        hd_a = self.hd_ensemble(hd.view(1, 1)).permute(1, 0, 2)

        hidden = []
        hidden.append(self.fc_pc_init(pc_a))
        hidden.append(self.fc_hd_init(hd_a))

        masks = torch.ones_like(action, device=device)[0]
        x = self.input_embedding(action[0], collision[0], masks)

        x, hidden_states = self.rnn(x.unsqueeze(1), hidden)

        x = x.squeeze(1)
        x, pc, hd, out = self.output_branch(x)
        return x, pc, hd, out, hidden_states


    def single_forward(self, action, hidden_states, masks, gps, collision, rt) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward for a non-sequence input

        Args:
            action: [envs, 1],
            hidden_states: [2, envs, feature_size],
            masks: [envs, 1],
            gps: [envs, 3], -- rho phi
            collision: [envs]
        """
        hidden, _, _ = self.init_hidden(masks, gps, rt)
        hidden_states = torch.where(
            masks.view(1, -1, 1), hidden_states, self.pack_hidden(hidden)
        )

        x = self.input_embedding(action, collision, masks)

        x, hidden_states = self.rnn(
            x.unsqueeze(0), self.unpack_hidden(hidden_states)
        )
        hidden_states = self.pack_hidden(hidden_states)

        x = x.squeeze(0)
        x, pc, hd, out = self.output_branch(x)
        return x, pc, hd, out, hidden_states

    def toSeqs(self, x, select_inds, batch_sizes):
        x = torch.nn.utils.rnn.pad_packed_sequence(PackedSequence(x.index_select(0, select_inds), batch_sizes, None, None))
        return x

    def seq_forward(self, action, hidden_states, masks, gps, collision, rt) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward for a sequence of length T

        Args:
            action: [step_size x envs, 1],
            hidden_states: [2, envs, feature_size],
            masks: [step_size x envs, 1],
            gps: [step_size x envs, 3], -- rho phi
            collision: [step_size x envs]
        """
        N = hidden_states.size(1)
        x = self.input_embedding(action, collision, masks)

        init_hidden_states, hd_embed, pc_embed = self.init_hidden(masks, gps, rt)
        init_hidden_states = self.pack_hidden(init_hidden_states)

        (
            x_seq,
            hidden_states,
            select_inds,
            rnn_state_batch_inds,
            last_episode_in_batch_mask,
            batch_sizes,
        ) = build_rnn_inputs(x, masks, hidden_states, init_hidden_states)
        # self.select_inds = select_inds
        # self.batch_size = batch_size

        x_seq, hidden_states = self.rnn(
            x_seq, self.unpack_hidden(hidden_states)
        )
        hidden_states = self.pack_hidden(hidden_states)

        x, hidden_states = build_rnn_out_from_seq(
            x_seq,
            hidden_states,
            select_inds,
            rnn_state_batch_inds,
            last_episode_in_batch_mask,
            N,
        )

        x, pc, hd, out = self.output_branch(x)

        pc_seqs = self.toSeqs(pc, select_inds, batch_sizes)
        hd_seqs = self.toSeqs(hd, select_inds, batch_sizes)
        gps_seqs = self.toSeqs(gps, select_inds, batch_sizes)
        rt_seqs = self.toSeqs(torch.cat([hd_embed, pc_embed], 1), select_inds, batch_sizes)
        return x, pc, hd, out, hidden_states, gps_seqs, pc_seqs, hd_seqs, rt_seqs

    def forward(self, action, masks, collision, hidden_states, gps, rt) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.permute(1, 0, 2)
        if action.size(0) == hidden_states.size(1):
            x, pc, hd, out, hidden_states = self.single_forward(action, hidden_states, masks, gps, collision, rt)
            gps_seqs, pc_seqs, hd_seqs = None, None, None
        else:
            x, pc, hd, out, hidden_states, gps_seqs, pc_seqs, hd_seqs, rt_seqs = \
                self.seq_forward(action, hidden_states, masks, gps, collision, rt)

        hidden_states = hidden_states.permute(1, 0, 2)

        gc_info = {
            "feature": x,
            "pc": pc,
            "hd": hd,
            "compressed_feature": out,
            "hidden": hidden_states,
            "gps_seqs": gps_seqs,
            "pc_seqs": pc_seqs,
            "hd_seqs": hd_seqs,
            "rt_seqs": hd_seqs,
        }

        return gc_info