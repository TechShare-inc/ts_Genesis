# only actor
import torch
import torch.nn as nn

from typing import Optional, Tuple

class InferenceActorLSTM(nn.Module):
    def __init__(self, actor_mlp: nn.Module, lstm: nn.LSTM):
        """
        推論専用の Actor + LSTM
        - actor_mlp: 学習済み Actor (MLP 部分)
        - lstm: 学習済み LSTM 部分
        """
        super().__init__()
        self.actor = actor_mlp # MLP
        self.lstm = lstm # RNN

    def forward(
            self,
            obs: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor,
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        obs: (N, obs_dim)
        hx, cx: (1, N, hidden_size)
        """
        obs = obs.unsqueeze(0) # (1, N, obs_dim)
        # LSTMで処理
        out, (hx_new, cx_new) = self.lstm(obs, (hx, cx))
        
        # 最新の出力を MLP に入力
        action = self.actor(out.squeeze(0))  # seq の最後の出力だけ使う
        
        return action, hx_new, cx_new

class InferenceActorLSTMWrapper(nn.Module):
    def __init__(self, inference_actor: nn.Module, num_envs: int, rnn_hidden_size: int):
        super().__init__()
        self.inference_actor = inference_actor
        self.num_envs = num_envs
        self.rnn_hidden_size = rnn_hidden_size

    @torch.jit.export
    def init_hidden(self, device: torch.device):
        """LSTM 初期状態を生成"""
        hx = torch.zeros(1, self.num_envs, self.rnn_hidden_size, device=device)
        cx = torch.zeros(1, self.num_envs, self.rnn_hidden_size, device=device)
        return hx, cx

    def forward(self, obs: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor):
        """
        TorchScript 用 forward。
        obs: [num_envs, obs_dim]
        hx, cx: [1, num_envs, hidden_size]
        """
        action, new_h, new_c = self.inference_actor(obs, hx, cx)
        return action, new_h, new_c