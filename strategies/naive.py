from __future__ import annotations
from .base import BaseStrategy
from dt.dataset import Trajectory, make_minibatches



from typing import List

import torch
import torch.nn.functional as F

from dt.dataset_panda import make_minibatches_panda
from dt.panda_dt import PandaDecisionTransformer

class NaiveStrategy(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_task(
        self,
        task_trajs: List[Trajectory],
        steps: int = 2000,
        batch_size: int = 64,
    ):
        # minibatches only from the current task (no rehearsal buffer)
        loader_task = make_minibatches(
            task_trajs,
            self.seq_len,
            batch_size,
            self.device,
        )

        self.model.train()
        for _ in range(steps):
            obs, actions, rtg, ts = next(loader_task)

            # shift actions by 1 along the sequence (same as in cumulative replay)
            logits = self.model(
                obs,
                torch.roll(actions, shifts=1, dims=1),
                rtg,
                ts,
            )

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                actions.reshape(-1),
                ignore_index=-1,
            )

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()

        return {}

    def after_task(self, task_trajs: List[Trajectory]):
        # Naive strategy: do nothing after finishing a task
        # (no rehearsal buffer, no weight consolidation)
        pass



class PandaNaiveStrategy:
    """
    Naive continual strategy for Panda with continuous actions.
    """

    def __init__(
        self,
        obs_shape,
        act_dim: int,
        seq_len: int,
        device: str,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        p_drop: float = 0.1,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
    ):
        self.seq_len = seq_len
        self.device = device

        obs_dim = obs_shape[0]
        self.model = PandaDecisionTransformer(
            obs_dim=obs_dim,
            act_dim=act_dim,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            seq_len=seq_len,
            p_drop=p_drop,
        ).to(device)

        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def train_task(
        self,
        task_trajs: List[Trajectory],
        steps: int = 2000,
        batch_size: int = 64,
    ):
        loader = make_minibatches_panda(
            task_trajs, self.seq_len, batch_size, self.device
        )

        self.model.train()
        for _ in range(steps):
            obs, actions, rtg, ts, mask = next(loader)
            # previous actions = shifted actions; first prev action = 0
            prev_actions = torch.roll(actions, shifts=1, dims=1)
            prev_actions[:, 0, :] = 0.0

            pred = self.model(obs, prev_actions, rtg, ts)  # [B, L, act_dim]

            # MSE z maską (ignorujemy padding)
            mse_per_step = ((pred - actions) ** 2).mean(dim=-1)  # [B, L]
            loss = (mse_per_step * mask).sum() / mask.sum()

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()

        return {}

    def after_task(self, task_trajs: List[Trajectory]):
        # Naive: nic po tasku
        return
