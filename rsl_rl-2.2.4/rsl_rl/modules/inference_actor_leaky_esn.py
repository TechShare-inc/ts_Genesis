from __future__ import annotations
import math
from typing import Union, List, Optional

import torch
import torch.nn as nn
import numpy as np

# from rsl_rl.modules.actor_critic import ActorCritic
# from rsl_rl.utils import resolve_nn_activation, unpad_trajectories

class InferenceActorMultiLeakyESN(nn.Module):
    """
    推論専用 Actor ESN モデル
    - actor_mlp : 学習済み Actor (MLP)
    - esn_cells : List[nn.Module]  各 LeakyESN セル（学習済み）
    - leak_rate : List[float]
    """
    def __init__(self, actor_mlp, esn: InferenceMultiLeakyESN):
        super().__init__()
        self.actor = actor_mlp
        self.esn = esn

    def forward(self, obs):
        out = self.esn(obs)
        action = self.actor(out)
        return action

class InferenceMultiLeakyESN(nn.Module):
    def __init__(
        self,
        W_in: torch.Tensor,      # (N, H, U)
        W: torch.Tensor,         # (N, H, H)
        leak_rates: torch.Tensor,# (N, 1, 1)
        num_envs: int,
    ):
        super().__init__()

        self.num_reservoir = W.shape[0]
        self.reservoir_size = W.shape[1]

        # fixed parameters
        self.register_buffer("W_in", W_in)
        self.register_buffer("W", W)
        self.register_buffer("leak_rates", leak_rates)

        # internal state
        self.register_buffer(
            "reservoir_states",
            torch.zeros(self.num_reservoir, num_envs, self.reservoir_size)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, U)
        """
        h = self.reservoir_states  # (N, B, H)

        # input projection
        x1 = torch.einsum("nru,bu->nbr", self.W_in, obs)

        # recurrent
        x2 = torch.einsum("nrh,nbh->nbr", self.W, h)

        # activation
        pre = x1 + x2
        act = torch.tanh(pre)

        # leaky integration
        next_h = (1.0 - self.leak_rates) * h + self.leak_rates * act

        # state update (NO rebind)
        self.reservoir_states.copy_(next_h)

        # flatten
        out = next_h.permute(1, 0, 2).reshape(obs.size(0), -1)
        return out

    @torch.jit.export
    def reset(self):
        self.reservoir_states.zero_()

class InferenceActorMultiLeakyESNWrapper(nn.Module):
    def __init__(self, inference_actor: InferenceActorMultiLeakyESN,
                 num_envs: int, num_reservoir: int, reservoir_size: int):
        super().__init__()
        self.inference_actor = inference_actor

        # hidden state（ESN の reservoir state）を保持
        self.register_buffer(
            "reservoir_states",
            torch.zeros(num_reservoir, num_envs, reservoir_size)
        )

    def forward(self, obs):
        return self.inference_actor(obs)

    @torch.jit.export
    def reset(self):
        self.inference_actor.esn.reset()