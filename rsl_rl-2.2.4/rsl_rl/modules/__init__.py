# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions for neural-network components for RL-agents."""

from .actor_critic import ActorCritic
from .actor_critic_recurrent import ActorCriticRecurrent
from .actor_critic_leaky_esn import ActorCriticMultiLeakyESN
from .inference_actor_lstm import InferenceActorLSTM, InferenceActorLSTMWrapper
from .inference_actor_leaky_esn import InferenceActorMultiLeakyESN, InferenceMultiLeakyESN, InferenceActorMultiLeakyESNWrapper
from .normalizer import EmpiricalNormalization
from .rnd import RandomNetworkDistillation

__all__ = ["ActorCritic",
           "ActorCriticRecurrent",
           "InferenceActorLSTM", "InferenceActorLSTMWrapper",
           "ActorCriticMultiLeakyESN",
           "InferenceActorMultiLeakyESN",
           "InferenceMultiLeakyESN",
           "InferenceActorMultiLeakyESNWrapper",
           "EmpiricalNormalization", "RandomNetworkDistillation"]
