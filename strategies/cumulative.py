from __future__ import annotations

import random
from typing import List, Tuple, Optional, Any

import torch
import torch.nn.functional as F

from dt.dataset import Trajectory, make_minibatches
from dt.dataset_panda import make_minibatches_panda
from dt.panda_dt import PandaDecisionTransformer
from .base import BaseStrategy


# ============================================================
# Helpers
# ============================================================

def _unpack_batch(batch: Tuple[Any, ...]):
    """
    Accepts both:
      - (obs, actions, rtg, ts, mask)
      - (obs, actions, rtg, ts)   -> mask derived from actions != -1
    """
    if len(batch) == 5:
        obs, actions, rtg, ts, mask = batch
    elif len(batch) == 4:
        obs, actions, rtg, ts = batch
        # derive mask from padding in actions
        if isinstance(actions, torch.Tensor):
            mask = actions.ne(-1)
        else:
            # very defensive fallback (should not happen in your pipeline)
            mask = torch.tensor(actions != -1, device=obs.device, dtype=torch.bool)
    else:
        raise ValueError(f"Unexpected batch tuple length={len(batch)}. Expected 4 or 5.")

    # ensure mask is bool tensor on same device
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, device=obs.device, dtype=torch.bool)
    else:
        mask = mask.to(device=obs.device, dtype=torch.bool)

    return obs, actions, rtg, ts, mask


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    pred/target: [B,L,D]
    mask: [B,L] bool
    """
    mse_per_step = ((pred - target) ** 2).mean(dim=-1)  # [B,L]
    m = mask.float()
    denom = m.sum().clamp(min=1.0)
    return (mse_per_step * m).sum() / denom


def _make_panda_loader(
    trajs: List[Trajectory],
    seq_len: int,
    batch_size: int,
    device: str,
    *,
    obs_dim: Optional[int] = None,
    act_dim: Optional[int] = None,
):
    """
    make_minibatches_panda w Twoich wersjach bywa w 2 wariantach sygnatury:
      - make_minibatches_panda(trajs, seq_len, batch_size, device)
      - make_minibatches_panda(trajs, seq_len=..., batch_size=..., device=..., obs_dim=..., act_dim=...)
    """
    try:
        # newer / explicit
        return make_minibatches_panda(
            trajs,
            seq_len=seq_len,
            batch_size=batch_size,
            device=device,
            obs_dim=obs_dim,
            act_dim=act_dim,
        )
    except TypeError:
        # older / positional
        return make_minibatches_panda(trajs, seq_len, batch_size, device)


# ============================================================
# Atari / Discrete DT (CL) - cumulative replay
# ============================================================

class CumulativeReplayStrategy(BaseStrategy):
    """
    Discrete-action cumulative replay for Atari DT.
    IMPORTANT: For DT token order (R,s,a) you feed actions AS-IS (no shift).
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
        # split batch
        main_bs = max(1, int(round(batch_size * (1.0 - mix))))
        reh_bs = batch_size - main_bs

        loader_task = make_minibatches(task_trajs, self.seq_len, main_bs, self.device)
        loader_reh = (
            make_minibatches(self.rehearsal, self.seq_len, reh_bs, self.device)
            if (self.rehearsal and reh_bs > 0)
            else None
        )

        self.model.train()
        for _ in range(int(steps)):
            obs_list = []
            act_list = []
            rtg_list = []
            ts_list = []
            mask_list = []

            # current task batch
            b = next(loader_task)
            obs, actions, rtg, ts, mask = _unpack_batch(b)
            obs_list.append(obs)
            act_list.append(actions)
            rtg_list.append(rtg)
            ts_list.append(ts)
            mask_list.append(mask)

            # rehearsal batch
            if loader_reh is not None:
                b = next(loader_reh)
                obs, actions, rtg, ts, mask = _unpack_batch(b)
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

            # ✅ actions AS-IS (no roll); pass attention_mask
            logits = self.model(obs, actions, rtg, ts, attention_mask=mask)  # [B,L,A]

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
        pool = list(self.rehearsal) + list(task_trajs)
        random.shuffle(pool)
        self.rehearsal = pool[: self.rehearsal_capacity]


# ============================================================
# Panda / Continuous DT (CL) - cumulative replay
# ============================================================

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
        n_heads: int = 4,
        p_drop: float = 0.1,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        rehearsal_capacity: int = 500,
    ):
        self.seq_len = int(seq_len)
        self.device = str(device)
        self.rehearsal_capacity = int(rehearsal_capacity)
        self.rehearsal: List[Trajectory] = []

        obs_dim = int(obs_shape[0])
        self.obs_dim = obs_dim
        self.act_dim = int(act_dim)

        self.model = PandaDecisionTransformer(
            obs_dim=obs_dim,
            act_dim=self.act_dim,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            seq_len=self.seq_len,
            p_drop=p_drop,
        ).to(self.device)

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
        main_bs = max(1, int(round(batch_size * (1.0 - mix))))
        reh_bs = batch_size - main_bs

        loader_task = _make_panda_loader(
            task_trajs,
            seq_len=self.seq_len,
            batch_size=main_bs,
            device=self.device,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
        )
        loader_reh = (
            _make_panda_loader(
                self.rehearsal,
                seq_len=self.seq_len,
                batch_size=reh_bs,
                device=self.device,
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
            )
            if (self.rehearsal and reh_bs > 0)
            else None
        )

        self.model.train()
        for _ in range(int(steps)):
            obs_list = []
            act_list = []
            rtg_list = []
            ts_list = []
            mask_list = []

            b = next(loader_task)
            obs, actions, rtg, ts, mask = _unpack_batch(b)
            obs_list.append(obs)
            act_list.append(actions)
            rtg_list.append(rtg)
            ts_list.append(ts)
            mask_list.append(mask)

            if loader_reh is not None:
                b = next(loader_reh)
                obs, actions, rtg, ts, mask = _unpack_batch(b)
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

            # ✅ Continuous: feed actions AS-IS, use mask
            pred = self.model(obs, actions, rtg, ts, attention_mask=mask)  # [B,L,act_dim]
            loss = _masked_mse(pred, actions, mask)

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()

        return {}

    def after_task(self, task_trajs: List[Trajectory]):
        pool = list(self.rehearsal) + list(task_trajs)
        random.shuffle(pool)
        self.rehearsal = pool[: self.rehearsal_capacity]
