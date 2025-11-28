
from __future__ import annotations
from typing import List
import torch, torch.nn as nn, torch.optim as optim
from dt.model import DecisionTransformer
from dt.dataset import Trajectory, make_minibatches

class BaseStrategy:
    def __init__(self, obs_shape, n_actions: int, seq_len: int=20, device: str='cuda', lr: float=3e-4):
        self.device = device
        self.model = DecisionTransformer(
            obs_shape=obs_shape,
            n_actions=n_actions,
            seq_len=seq_len,
        ).to(device)
        self.opt = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.seq_len = seq_len

    def train_task(self, task_trajs: List[Trajectory], steps: int=2000, batch_size: int=64):
        loader = make_minibatches(task_trajs, self.seq_len, batch_size, self.device)
        self.model.train()
        for _ in range(steps):
            obs, actions, rtg, ts = next(loader)
            logits = self.model(obs, torch.roll(actions, shifts=1, dims=1), rtg, ts)
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), actions.reshape(-1), ignore_index=-1)
            self.opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
        return {}

    def after_task(self, task_trajs: List[Trajectory]): pass
