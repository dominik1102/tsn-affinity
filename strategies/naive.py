from __future__ import annotations

from typing import List, Tuple, Any, Optional

import torch
import torch.nn.functional as F

from .base import BaseStrategy
from dt.dataset import Trajectory, make_minibatches

from dt.dataset_panda import make_minibatches_panda
from dt.panda_dt import PandaDecisionTransformer


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _device_type(x: Any) -> str:
    """Return 'cpu' or 'cuda' (make_minibatches_panda zwykle chce typ, nie 'cuda:0')."""
    try:
        return torch.device(x).type
    except Exception:
        s = str(x)
        return "cuda" if "cuda" in s else "cpu"


def _unpack_discrete(batch: Tuple[Any, ...]):
    """
    Discrete batches:
      - (obs, actions, rtg, ts, mask)
      - (obs, actions, rtg, ts) -> mask from actions != -1
    """
    if len(batch) == 5:
        obs, actions, rtg, ts, mask = batch
    elif len(batch) == 4:
        obs, actions, rtg, ts = batch
        mask = actions.ne(-1)
    else:
        raise ValueError(f"Unexpected batch len={len(batch)} (expected 4 or 5).")

    mask = mask.to(device=obs.device, dtype=torch.bool)
    return obs, actions, rtg, ts, mask


def _unpack_continuous(batch: Tuple[Any, ...]):
    """
    Continuous batches:
      - (obs, actions, rtg, ts, mask)
      - (obs, actions, rtg, ts) -> mask = ones
    """
    if len(batch) == 5:
        obs, actions, rtg, ts, mask = batch
        mask = mask.to(device=obs.device, dtype=torch.bool)
    elif len(batch) == 4:
        obs, actions, rtg, ts = batch
        B, L = actions.shape[:2]
        mask = torch.ones((B, L), device=obs.device, dtype=torch.bool)
    else:
        raise ValueError(f"Unexpected batch len={len(batch)} (expected 4 or 5).")

    return obs, actions, rtg, ts, mask


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    pred/target: [B,L,act_dim]
    mask: [B,L] bool
    """
    mse_per_step = ((pred - target) ** 2).mean(dim=-1)  # [B,L]
    m = mask.float()
    denom = m.sum().clamp(min=1.0)
    return (mse_per_step * m).sum() / denom


# ------------------------------------------------------------
# Atari / Discrete Naive strategy (BaseStrategy)
# ------------------------------------------------------------

class NaiveStrategy(BaseStrategy):
    """
    Discrete-action naive (Atari).
    IMPORTANT: No action shift for DT token order (R,s,a) with prediction at s_t.
    """

    def train_task(self, task_trajs: List[Trajectory], steps: int = 2000, batch_size: int = 64):
        loader = make_minibatches(task_trajs, self.seq_len, batch_size, self.device)

        self.model.train()
        for _ in range(int(steps)):
            obs, actions, rtg, ts, mask = _unpack_discrete(next(loader))

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
        return


# ------------------------------------------------------------
# Panda / Continuous Naive strategy (separate class)
# ------------------------------------------------------------

class PandaNaiveStrategy:
    """
    Continuous-action naive for Panda.
    Loss: masked MSE on continuous actions.
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
        self.seq_len = int(seq_len)
        self.device = str(device)

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

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_task(self, task_trajs: List[Trajectory], steps: int = 2000, batch_size: int = 64):
        dev = _device_type(self.device)

        # make_minibatches_panda bywa w 2 wariantach sygnatury
        try:
            loader = make_minibatches_panda(
                task_trajs,
                seq_len=self.seq_len,
                batch_size=batch_size,
                device=dev,
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
            )
        except TypeError:
            loader = make_minibatches_panda(task_trajs, self.seq_len, batch_size, dev)

        self.model.train()
        for _ in range(int(steps)):
            obs, actions, rtg, ts, mask = _unpack_continuous(next(loader))

            # ✅ NO SHIFT
            pred = self.model(obs, actions, rtg, ts, attention_mask=mask)  # [B,L,act_dim]
            loss = _masked_mse(pred, actions, mask)

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()

        return {}

    def after_task(self, task_trajs: List[Trajectory]):
        return
