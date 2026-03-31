from __future__ import annotations
import math
from typing import Union, List, Optional

import torch
import torch.nn as nn
import numpy as np

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.utils import resolve_nn_activation, unpad_trajectories

class ActorCriticMultiLeakyESN(ActorCritic):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        num_reservoir:int=1,
        reservoir_size:int=128,
        leak_rate:Optional[List[float]] = None,
        W_in_scale:Union[float, List[float]]=0.7,
        sparsity:Union[float, List[float]]=0.92,
        W_in_sparsity:Union[float, List[float]]=0.,
        spectral_radius:Union[float, List[float]]=0.9,
        structure:Union[str, List[str]]="erdos_renyi",
        reservoir_activateion:str="tanh",
        init_noise_std=1.0,
        device:str="cpu",):

        modules_config = None

        total_res_size = reservoir_size*num_reservoir

        # leak rate
        if leak_rate is None:
            leak_rate = [0.8] * num_reservoir
        elif isinstance(leak_rate, list):
            if any(not isinstance(lr, float) for lr in leak_rate):
                leak_rate = [0.8] * num_reservoir
            else:
                len_leak_rate = len(leak_rate)
                if num_reservoir > len_leak_rate:
                    for _ in range(num_reservoir-len_leak_rate):
                        leak_rate.append(leak_rate[-1])
                else:
                    leak_rate = leak_rate[:num_reservoir]
        else:
            leak_rate = [0.8] * num_reservoir
        
        # change list W_in_scale
        if isinstance(W_in_scale, float):
            W_in_scale = [W_in_scale] * num_reservoir
        elif isinstance(W_in_scale, list):
            if any(not isinstance(scale, float) for scale in W_in_scale):
                W_in_scale = [0.7] * num_reservoir
            else:
                len_W_in_scale = len(W_in_scale)
                if num_reservoir > len_W_in_scale:
                    for _ in range(num_reservoir-len_W_in_scale):
                        W_in_scale.append(W_in_scale[-1])
                else:
                    W_in_scale = W_in_scale[:num_reservoir]
        else:
            W_in_scale = [0.7] * num_reservoir
        
        # change list sparsity (W)
        if isinstance(sparsity, float):
            sparsity = [sparsity] * num_reservoir
        elif isinstance(sparsity, list):
            if any(not isinstance(sp, float) for sp in sparsity):
                sparsity = [0.92] * num_reservoir
            else:
                len_sparsity = len(sparsity)
                if num_reservoir > len_sparsity:
                    for _ in range(num_reservoir-len_sparsity):
                        sparsity.append(sparsity[-1])
                else:
                    sparsity = sparsity[:num_reservoir]
        else:
            sparsity = [0.92] * num_reservoir
        
        # change list sparsity (W_in)
        if isinstance(W_in_sparsity, float):
            W_in_sparsity = [W_in_sparsity] * num_reservoir
        elif isinstance(W_in_sparsity, list):
            if any(not isinstance(sp, float) for sp in W_in_sparsity):
                W_in_sparsity = [0.] * num_reservoir
            else:
                len_W_in_sparsity = len(W_in_sparsity)
                if num_reservoir > len_W_in_sparsity:
                    for _ in range(num_reservoir-len_W_in_sparsity):
                        W_in_sparsity.append(W_in_sparsity[-1])
                else:
                    W_in_sparsity = W_in_sparsity[:num_reservoir]
        else:
            W_in_sparsity = [0.] * num_reservoir
        
        # change list spectral radius
        if isinstance(spectral_radius, float):
            spectral_radius = [spectral_radius] * num_reservoir
        elif isinstance(spectral_radius, list):
            if any(not isinstance(sr, float) for sr in spectral_radius):
                spectral_radius = [0.9] * num_reservoir
            else:
                len_sr = len(spectral_radius)
                if num_reservoir > len_sr:
                    for _ in range(num_reservoir-len_sr):
                        spectral_radius.append(spectral_radius[-1])
                else:
                    spectral_radius = spectral_radius[:num_reservoir]
        else:
            spectral_radius = [0.9] * num_reservoir
        
        # change list structure
        if isinstance(structure, str):
            structure = [structure] * num_reservoir
        elif isinstance(structure, list):
            if any(not isinstance(st, str) for st in structure):
                structure = ["erdos_renyi"] * num_reservoir
            else:
                len_structure = len(structure)
                if num_reservoir > len_structure:
                    for _ in range(num_reservoir-len_structure):
                        structure.append(structure[-1])
                else:
                    structure = structure[:num_reservoir]
        else:
            structure = ["erdos_renyi"] * num_reservoir

        if modules_config is None:
            modules_config = []
            for i in range(num_reservoir):
                modules_config.append({
                    "leak_rate": leak_rate[i],
                    "W_in_scale": W_in_scale[i],
                    "sparsity": sparsity[i],
                    "W_in_sparsity": W_in_sparsity[i],
                    "spectral_radius": spectral_radius[i],
                    "structure": structure[i],
                })

        super().__init__(
            num_actor_obs=total_res_size,
            num_critic_obs=total_res_size,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        self.device = device
    
        activation = resolve_nn_activation(activation)
        self.reservoir_a = MultiLeakyESN(input_size=num_actor_obs, reservoir_size=reservoir_size, modules_config=modules_config, activation=reservoir_activateion, device=device)
        self.reservoir_c = MultiLeakyESN(input_size=num_critic_obs, reservoir_size=reservoir_size, modules_config=modules_config, activation=reservoir_activateion, device=device)
        
        print(f"Actor Reservoir: {self.reservoir_a}")
        print(f"Critic Reservoir: {self.reservoir_c}")

    def reset(self, dones=None):
        self.reservoir_a.reset(dones)
        self.reservoir_c.reset(dones)
    
    def act(self, observations, masks=None, hidden_states=None):
        if hidden_states is None:
            hidden_states = self.reservoir_a.reservoir_states

        input_a = self.reservoir_a(observations, masks, hidden_states)

        out = super().act(input_a)
        # print("out size", out.size())
        return out
        
    def act_inference(self, observations):
        input_a = self.reservoir_a(observations)

        return super().act_inference(input_a)

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        input_c = self.reservoir_c(critic_observations, masks, hidden_states)
        
        out = super().evaluate(input_c)
        return out

    def get_hidden_states(self):
        # if self.reservoir_a.reservoir_states is not None:
        #     print("large reservor state rate: ", torch.mean((torch.abs(self.reservoir_a.reservoir_states)>0.9).float()))
        #     print("min reservoir_states: ", torch.min(self.reservoir_a.reservoir_states))
        #     print("max reservoir_states: ", torch.max(self.reservoir_a.reservoir_states))
        #     print("mean reservoir_states: ", torch.mean(self.reservoir_a.reservoir_states))
        #     print("std reservoir_states: ", torch.std(self.reservoir_a.reservoir_states))
        return (self.reservoir_a.reservoir_states, self.reservoir_c.reservoir_states)

class MultiLeakyESN(nn.Module):
    def __init__(self,
                 input_size:int,
                 reservoir_size:int,
                 modules_config:List[dict],
                 activation:str,
                 device:str="cpu"):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.num_reservoir = len(modules_config)
        
        W = torch.zeros(self.num_reservoir, self.reservoir_size, self.reservoir_size)
        W_in = torch.zeros(self.num_reservoir, self.reservoir_size, self.input_size)
        leak_rates = []

        for i, cfg in enumerate(modules_config):
            leak_rate = cfg.get("leak_rate", 0.8)
            W_in_scale = cfg.get("W_in_scale", 0.7)
            sparsity = cfg.get("sparsity", 0.92)
            W_in_sparsity = cfg.get("W_in_sparsity", 0.)
            spectral_radius = cfg.get("spectral_radius", 0.9)
            structure = cfg.get("structure", "erdos_renyi")

            # W_in_i = nn.Parameter(
            #     generate_W_in(input_size=self.input_size,
            #                   reservoir_size=self.reservoir_size,
            #                   W_in_sparsity=W_in_sparsity,
            #                   g=W_in_scale,
            #                   device=device))

            # W_i  = nn.Parameter(
            #     generate_W(reservoir_size=self.reservoir_size,
            #                structure=structure,
            #                sparsity=sparsity,
            #                spectral_radius=spectral_radius,
            #                device=device))

            W_in_i = generate_W_in(input_size=self.input_size,
                                   reservoir_size=self.reservoir_size,
                                   W_in_sparsity=W_in_sparsity,
                                   g=W_in_scale,
                                   device=device)

            W_i  = generate_W(reservoir_size=self.reservoir_size,
                              structure=structure,
                              sparsity=sparsity,
                              spectral_radius=spectral_radius,
                              device=device)

            W_in[i] = W_in_i
            W[i] = W_i

            leak_rates.append(leak_rate)
        
        # self.register_buffer("W", W)
        # self.register_buffer("W_in", W_in)
        self.W = nn.Parameter(W)
        self.W_in = nn.Parameter(W_in)
        
        self.register_buffer(
            "leak_rates",
            torch.tensor(leak_rates).view(self.num_reservoir, 1, 1)
        )
        
        self.reservoir_states = None
        self.activation = resolve_nn_activation("relu") if activation == "relu" else resolve_nn_activation("tanh")

    def forward(self, input, masks=None, hidden_states=None):        
        if input.ndim == 1:
            input = input.unsqueeze(0)
        batch = input.size(0)
        
        # print(input.size())
        
        if hidden_states is None:
            if self.reservoir_states is None or self.reservoir_states.size(1) != batch:
                self.reservoir_states = torch.randn(
                    self.num_reservoir, batch, self.reservoir_size,
                    device=input.device, dtype=input.dtype
                ) *0.1
            hidden_states = self.reservoir_states
        
        # print(hidden_states.size())

        batch_mode = masks is not None

        if batch_mode:
            out = self.forward_batch(input, masks, hidden_states)
        else:
            out = self.forward_step(input, hidden_states)

        return out
    

    def forward_batch(self, input, masks, hidden_states):
        """
        input: (T, B, U)
        hidden_states: (N, B, H)
        """
        T, B, _ = input.shape
        h = hidden_states.detach()
        outputs = []

        for t in range(T):
            x_t = input[t]  # (B, U)

            x1 = torch.einsum("nru,bu->nbr", self.W_in, x_t)
            x2 = torch.einsum("nrh,nbh->nbr", self.W, h)

            h = (1.0 - self.leak_rates) * h \
                + self.leak_rates * self.activation(x1 + x2)

            h = h * masks[t].view(1, B, 1)
            outputs.append(h)

        outputs = torch.stack(outputs, dim=0)  # (T, N, B, H)
        out = outputs.permute(0, 2, 1, 3).reshape(T, B, -1)

        return unpad_trajectories(out, masks)


    def forward_step(self, input, hidden_states):
        # print(input.size())
        # print(hidden_states.size())

        x1 = torch.einsum("nru,bu->nbr", self.W_in, input)
        x2 = torch.einsum("nrh,nbh->nbr", self.W, hidden_states)

        next_states = (1.0 - self.leak_rates) * hidden_states + self.leak_rates * self.activation(x1 + x2)
        self.reservoir_states = next_states.detach()
        out = next_states.permute(1, 0, 2).contiguous().reshape(input.size(0), -1)
        return out

    def reset(self, dones=None):
        if self.reservoir_states is None:
            return

        with torch.no_grad():
            if dones is None:
                self.reservoir_states.copy_(
                    torch.randn_like(self.reservoir_states) / self.reservoir_size
                )
            else:
                dones_t = torch.as_tensor(
                    dones, device=self.reservoir_states.device
                ).view(-1).bool()
                self.reservoir_states[:, dones_t, :] = torch.randn_like(self.reservoir_states[:, dones_t, :]) * 0.1

# ---
# W, W_in generator
# ---
def generate_W(reservoir_size,
               structure="erdos_renyi", # "erdos_renyi" | "small_world" | "ring" | "grid"
               sparsity=0.92,
               spectral_radius=0.9,
               small_world_k=4,         # for small_world: each node connects to k neighbors on each side (total 2k)
               small_world_p=0.1,       # rewiring probability for small_world
               rng_seed: Optional[int]=None,
               device="cpu",
               dtype=torch.float32):
    rng = np.random.RandomState(None if rng_seed is None else rng_seed)
    N = reservoir_size
    
    mask = np.zeros((N, N), dtype=np.bool_)

    if structure == "erdos_renyi":
        prob_nonzero = 1.0 - sparsity
        mask = rng.rand(N, N) < prob_nonzero
        np.fill_diagonal(mask, False)
    
    elif structure == "ring":
        for i in range(N):
            mask[i, (i+1) % N] = True
            mask[i, (i-1) % N] = True
    
    elif structure == "small_world":
        k = small_world_k  # neighbors to each side
        if k <= 0 or 2*k >= N:
            raise ValueError("small_world_k invalid")
        # initial nearest-neighbor ring (undirected adjacency)
        for i in range(N):
            for j in range(1, k+1):
                mask[i, (i+j) % N] = True
                mask[i, (i-j) % N] = True

        # rewire each outgoing edge (i -> i+j) with prob p
        p = small_world_p
        for i in range(N):
            for j in range(1, k+1):
                if rng.rand() < p:
                    mask[i, (i+j) % N] = False
                    cand = list(range(N))
                    cand.remove(i)
                    target = rng.choice(cand)
                    mask[i, target] = True
    
    elif structure == "grid":
        raise NotImplementedError("grid structure not implemented in this util")

    else:
        raise ValueError("Unknown structure: " + str(structure))
    
    W_np = np.zeros((N, N), dtype=np.float32)
    nz = np.where(mask)
    if len(nz[0]) > 0:
        W_np[nz] = rng.randn(len(nz[0])).astype(np.float32)

    W = torch.from_numpy(W_np).to(device=device, dtype=dtype)

    # spectral radius normalization
    with torch.no_grad():
        # b = torch.randn(N, device=device, dtype=dtype)
        # b = b / (b.norm() + 1e-12)
        # for _ in range(40):
        #     b = W.matmul(b)
        #     n = b.norm() + 1e-12
        #     b = b / n
        # rho = float(torch.abs(torch.dot(W.matmul(b), b)).item())
        # print("rho: ", rho)
        # if rho > spectral_radius/2:
        #     W = W * float(spectral_radius/rho)
        
        # x = torch.randn(W.shape[0], 1, device=W.device)
        # iters = 20
        # for _ in range(iters):
        #     x = W @ x
        #     x = x / x.norm()
        # rho = (x.t() @ (W @ x)).item()
        # print("rho: ", rho)
        # W = W * (spectral_radius / max(abs(rho), spectral_radius/2))
        # print("scaling W: ", spectral_radius / max(abs(rho), spectral_radius))

        if structure == "erdos_renyi":
            W = W * (prob_nonzero) / N
        if structure == "small_world":
            W = W * spectral_radius / (2*k)

    return W

def generate_W_in(input_size:int,
                  reservoir_size:int,
                  W_in_sparsity:float=0.,
                  g:float=0.7,
                  device="cpu",
                  dtype=torch.float32):
    scale = float(g) / math.sqrt(float(input_size))
    # scale = float(g) / float(input_size)
    W_in = torch.randn(reservoir_size, input_size, device=device, dtype=dtype) * scale
    if W_in_sparsity > 0:
        prob_nonzero = 1.0 - float(W_in_sparsity)
        mask = (torch.rand(reservoir_size, input_size, device=device) < prob_nonzero).to(dtype)
        W_in = W_in * mask
    return W_in