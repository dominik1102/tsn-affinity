
from __future__ import annotations

import random
from typing import List

import torch
import torch.nn.functional as F  # F is not strictly needed but often handy

from dt.dataset import Trajectory
from dt.dataset import make_minibatches
from dt.dataset_panda import make_minibatches_panda
from dt.panda_dt import PandaDecisionTransformer
from .base import BaseStrategy


class CumulativeReplayStrategy(BaseStrategy):
    def __init__(self, *args, rehearsal_capacity: int=500, **kwargs):
        super().__init__(*args, **kwargs)
        self.rehearsal: List[Trajectory] = []
        self.rehearsal_capacity = rehearsal_capacity

    def train_task(self, task_trajs: List[Trajectory], steps: int=2000, batch_size: int=64, mix: float=0.3):
        loader_task = make_minibatches(task_trajs, self.seq_len, int(batch_size*(1-mix)), self.device)
        loader_reh = make_minibatches(self.rehearsal, self.seq_len, int(batch_size*mix), self.device) if self.rehearsal else None
        self.model.train()
        for _ in range(steps):
            batches = [next(loader_task)]
            if loader_reh: batches.append(next(loader_reh))
            obs = torch.cat([b[0] for b in batches], dim=0)
            actions = torch.cat([b[1] for b in batches], dim=0)
            rtg = torch.cat([b[2] for b in batches], dim=0)
            ts = torch.cat([b[3] for b in batches], dim=0)
            logits = self.model(obs, torch.roll(actions, shifts=1, dims=1), rtg, ts)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), actions.reshape(-1), ignore_index=-1)
            self.opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
        return {}

    def after_task(self, task_trajs: List[Trajectory]):
        pool = (self.rehearsal + task_trajs)
        random.shuffle(pool)
        self.rehearsal = pool[: self.rehearsal_capacity]


class PandaCumulativeReplayStrategy:
    """
    Cumulative replay strategy for Panda with continuous actions.

    - Uses PandaDecisionTransformer (continuous action prediction).
    - Maintains a rehearsal buffer of trajectories from previous tasks.
    - On each SGD step, mixes current-task and rehearsal trajectories.
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
        rehearsal_capacity: int = 500,
    ):
        self.seq_len = seq_len
        self.device = device
        self.rehearsal_capacity = rehearsal_capacity
        self.rehearsal: List[Trajectory] = []
        self.act_dim = act_dim
        # For Panda we treat obs_shape as [obs_dim]
        obs_dim = obs_shape[0]
        self.obs_dim = obs_dim


        # IMPORTANT:
        #   act_dim should be the global Panda action dimension
        #   (e.g. 4 for [x, y, z, gripper]),
        #   and must match what you used in PandaDecisionTransformer
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
        mix: float = 0.3,
    ):
        """
        Train on the current task trajectories with cumulative replay.

        Args:
            task_trajs: trajectories from the current Panda task.
            steps: number of SGD steps.
            batch_size: total batch size per step.
            mix: fraction of the batch taken from the rehearsal buffer (0..1).
                 Example: mix=0.3 -> 70% current task, 30% rehearsal (if available).
        """
        # Compute how many samples come from current task vs rehearsal
        main_bs = max(1, int(batch_size * (1.0 - mix)))
        reh_bs = batch_size - main_bs
        if reh_bs < 0:
            reh_bs = 0

        # Minibatch generators
        loader_task = make_minibatches_panda( task_trajs, self.seq_len, main_bs, self.device,  self.act_dim, self.obs_dim)
        loader_reh = (
            make_minibatches_panda(self.rehearsal, self.seq_len, reh_bs, self.device,  self.act_dim, self.obs_dim)
            if (self.rehearsal and reh_bs > 0)
            else None
        )

        self.model.train()
        for _ in range(steps):
            batch_obs = []
            batch_actions = []
            batch_rtg = []
            batch_ts = []
            batch_mask = []

            # --- current-task part ---
            obs_t, actions_t, rtg_t, ts_t, mask_t = next(loader_task)
            batch_obs.append(obs_t)
            batch_actions.append(actions_t)
            batch_rtg.append(rtg_t)
            batch_ts.append(ts_t)
            batch_mask.append(mask_t)

            # --- rehearsal part (if available) ---
            if loader_reh is not None:
                obs_r, actions_r, rtg_r, ts_r, mask_r = next(loader_reh)
                batch_obs.append(obs_r)
                batch_actions.append(actions_r)
                batch_rtg.append(rtg_r)
                batch_ts.append(ts_r)
                batch_mask.append(mask_r)

            # Concatenate along batch dimension
            obs = torch.cat(batch_obs, dim=0)         # [B, L, obs_dim]
            actions = torch.cat(batch_actions, dim=0) # [B, L, act_dim]
            rtg = torch.cat(batch_rtg, dim=0)         # [B, L, 1]
            ts = torch.cat(batch_ts, dim=0)           # [B, L]
            mask = torch.cat(batch_mask, dim=0)       # [B, L]

            # Previous actions: shift along time dimension, first prev = 0
            prev_actions = torch.roll(actions, shifts=1, dims=1)
            prev_actions[:, 0, :] = 0.0

            # Forward through PandaDecisionTransformer
            pred = self.model(obs, prev_actions, rtg, ts)  # [B, L, act_dim]

            # MSE per step, then mask out padding
            mse_per_step = ((pred - actions) ** 2).mean(dim=-1)  # [B, L]

            valid = mask.sum()
            if valid.item() > 0:
                loss = (mse_per_step * mask).sum() / valid
            else:
                # Safety fallback (should not happen)
                loss = mse_per_step.mean()

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()

        return {}

    def after_task(self, task_trajs: List[Trajectory]):
        """
        Update the rehearsal buffer after finishing a task:

        - pool = old buffer + current task trajectories
        - shuffle and keep only up to `rehearsal_capacity`
        """
        pool = self.rehearsal + task_trajs
        random.shuffle(pool)
        self.rehearsal = pool[: self.rehearsal_capacity]



