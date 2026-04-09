from __future__ import annotations

import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from dt.dataset import Trajectory, make_minibatches
from dt.panda_dt import PandaDecisionTransformer
from .base import BaseStrategy
from .utils import (
    make_panda_loader,
    masked_mse,
    prepare_panda_trajs,
    unpack_batch_continuous,
    unpack_batch_discrete,
)


class CumulativeReplayStrategy(BaseStrategy):
    """
    Discrete-action cumulative replay for Atari DT.

    Important:
      - aligned actions (no torch.roll),
      - attention mask comes from the shared discrete batch helper,
      - replay memory stores trajectories and mixes them into every batch,
      - if replay is empty, the current task uses the FULL batch_size
        (instead of shrinking it by `mix`).
    """

    def __init__(self, *args, rehearsal_capacity: int = 500, **kwargs):
        super().__init__(*args, **kwargs)
        self.rehearsal: List[Trajectory] = []
        self.rehearsal_capacity = int(rehearsal_capacity)

    def train_task(
        self,
        task_trajs: List[Trajectory],
        steps: int = 2000,
        batch_size: int = 64,
        mix: float = 0.3,
    ):
        mix = max(0.0, min(1.0, float(mix)))

        # If replay is empty, do NOT shrink the main batch.
        have_replay = (len(self.rehearsal) > 0) and (mix > 0.0)

        if not have_replay:
            main_bs = int(batch_size)
            reh_bs = 0
        else:
            main_bs = max(1, int(round(batch_size * (1.0 - mix))))
            reh_bs = max(0, int(batch_size) - main_bs)

        print(
            f"[cumulative-atari] steps={int(steps)} batch_size={int(batch_size)} mix={mix:.2f} "
            f"-> main_bs={main_bs} reh_bs={reh_bs} rehearsal={len(self.rehearsal)}"
        )

        loader_task = make_minibatches(task_trajs, self.seq_len, main_bs, self.device)
        loader_reh = (
            make_minibatches(self.rehearsal, self.seq_len, reh_bs, self.device)
            if (have_replay and reh_bs > 0)
            else None
        )

        self.model.train()
        for _ in range(int(steps)):
            obs_list = []
            act_list = []
            rtg_list = []
            ts_list = []
            mask_list = []

            # current-task batch
            b = next(loader_task)
            obs, actions, rtg, ts, mask = unpack_batch_discrete(b)
            obs_list.append(obs)
            act_list.append(actions)
            rtg_list.append(rtg)
            ts_list.append(ts)
            mask_list.append(mask)

            # replay batch (if available)
            if loader_reh is not None:
                b = next(loader_reh)
                obs, actions, rtg, ts, mask = unpack_batch_discrete(b)
                obs_list.append(obs)
                act_list.append(actions)
                rtg_list.append(rtg)
                ts_list.append(ts)
                mask_list.append(mask)

            obs = torch.cat(obs_list, dim=0)
            actions = torch.cat(act_list, dim=0)
            rtg = torch.cat(rtg_list, dim=0)
            ts = torch.cat(ts_list, dim=0)
            mask = torch.cat(mask_list, dim=0)

            logits = self.model(obs, actions, rtg, ts, attention_mask=mask)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                actions.reshape(-1),
                ignore_index=-1,
            )

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()

        return {}

    def after_task(self, task_trajs: List[Trajectory]):
        # Simple cumulative buffer:
        # keep old + new, shuffle, truncate to capacity.
        pool = list(self.rehearsal) + list(task_trajs)
        random.shuffle(pool)
        self.rehearsal = pool[: self.rehearsal_capacity]


class PandaCumulativeReplayStrategy:
    """
    Continuous-action cumulative replay for Panda DT.
    Loss = masked MSE on action vectors.
    """

    def __init__(
        self,
        obs_shape,
        act_dim: int,
        seq_len: int,
        device: str,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 1,
        p_drop: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        grad_clip: float = 0.25,
        rehearsal_capacity: int = 5000,
        max_ep_len: int = 50,
        rtg_scale: float = 1000.0,
    ):
        self.seq_len = int(seq_len)
        self.device = torch.device(str(device))
        self.rehearsal_capacity = int(rehearsal_capacity)
        self.rehearsal: List[Trajectory] = []
        self.n_seen_trajs = 0

        self.obs_dim = int(obs_shape[0])
        self.act_dim = int(act_dim)
        self.grad_clip = float(grad_clip)

        self.model_hparams = {
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "seq_len": int(seq_len),
            "d_model": int(d_model),
            "n_layers": int(n_layers),
            "n_heads": int(n_heads),
            "p_drop": float(p_drop),
            "max_ep_len": int(max_ep_len),
            "rtg_scale": float(rtg_scale),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "grad_clip": float(grad_clip),
            "rehearsal_capacity": int(rehearsal_capacity),
        }

        self.model = PandaDecisionTransformer(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            d_model=int(d_model),
            n_layers=int(n_layers),
            n_heads=int(n_heads),
            seq_len=self.seq_len,
            p_drop=float(p_drop),
            max_ep_len=int(max_ep_len),
            rtg_scale=float(rtg_scale),
        ).to(self.device)

        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )

    def _prepare_trajs(self, trajs: List[Trajectory]) -> List[Trajectory]:
        return prepare_panda_trajs(trajs, obs_dim=self.obs_dim, act_dim=self.act_dim)

    def train_task(
        self,
        task_trajs: List[Trajectory],
        steps: int = 2000,
        batch_size: int = 128,
        mix: float = 0.5,
    ):
        mix = float(np.clip(mix, 0.0, 1.0))

        task_trajs = self._prepare_trajs(task_trajs)
        replay_trajs = self.rehearsal if self.rehearsal else []

        have_replay = (len(replay_trajs) > 0) and (mix > 0.0)
        if not have_replay:
            main_bs = int(batch_size)
            reh_bs = 0
        else:
            main_bs = max(1, int(round(batch_size * (1.0 - mix))))
            reh_bs = max(0, int(batch_size) - main_bs)

        print(
            f"[cumulative] steps={int(steps)} batch_size={int(batch_size)} mix={mix:.2f} "
            f"-> main_bs={main_bs} reh_bs={reh_bs} rehearsal={len(self.rehearsal)}"
        )

        loader_task = make_panda_loader(
            task_trajs,
            seq_len=self.seq_len,
            batch_size=main_bs,
            device=self.device,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
        )

        loader_reh = (
            make_panda_loader(
                replay_trajs,
                seq_len=self.seq_len,
                batch_size=reh_bs,
                device=self.device,
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
            )
            if (have_replay and reh_bs > 0)
            else None
        )

        self.model.train()
        for _ in range(int(steps)):
            b = next(loader_task)
            obs, actions, rtg, ts, mask = unpack_batch_continuous(b)

            if loader_reh is not None:
                b2 = next(loader_reh)
                obs2, actions2, rtg2, ts2, mask2 = unpack_batch_continuous(b2)
                obs = torch.cat([obs, obs2], dim=0)
                actions = torch.cat([actions, actions2], dim=0)
                rtg = torch.cat([rtg, rtg2], dim=0)
                ts = torch.cat([ts, ts2], dim=0)
                mask = torch.cat([mask, mask2], dim=0)

            pred = self.model(obs, actions, rtg, ts, attention_mask=mask)
            loss = masked_mse(pred, actions, mask)

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()

        return {}

    def after_task(self, task_trajs: List[Trajectory]):
        task_trajs = self._prepare_trajs(task_trajs)

        for tr in task_trajs:
            self.n_seen_trajs += 1

            if len(self.rehearsal) < self.rehearsal_capacity:
                self.rehearsal.append(tr)
            else:
                j = random.randint(0, self.n_seen_trajs - 1)
                if j < self.rehearsal_capacity:
                    self.rehearsal[j] = tr
