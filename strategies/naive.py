from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from dt.dataset import Trajectory
from dt.panda_dt import PandaDecisionTransformer
from .base import BaseStrategy
from .utils import make_panda_loader, masked_mse, unpack_batch_continuous


class NaiveStrategy(BaseStrategy):
    """
    Atari / discrete naive strategy.

    BaseStrategy already implements the correct aligned-action DT training loop,
    so this class only exists to keep the import path stable.
    """
    pass


class PandaNaiveStrategy:
    """
    Continuous-action naive strategy for Panda.
    Loss: masked MSE on continuous actions.
    """

    def __init__(
        self,
        obs_shape,
        act_dim: int,
        seq_len: int,
        device: str,
        *,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 1,
        p_drop: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        grad_clip: float = 0.25,
        max_ep_len: int = 50,
        rtg_scale: float = 1000.0,
        obs_mean: Optional[np.ndarray] = None,
        obs_std: Optional[np.ndarray] = None,
    ):
        self.seq_len = int(seq_len)
        self.device = torch.device(str(device))
        self.grad_clip = float(grad_clip)

        self.obs_dim = int(obs_shape[0])
        self.act_dim = int(act_dim)

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
            act_tanh=False,
            obs_mean=obs_mean,
            obs_std=obs_std,
            rtg_scale=float(rtg_scale),
        ).to(self.device)

        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )

    def train_task(self, task_trajs: List[Trajectory], steps: int = 2000, batch_size: int = 64):
        loader = make_panda_loader(
            task_trajs,
            seq_len=self.seq_len,
            batch_size=int(batch_size),
            device=self.device,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
        )

        self.model.train()
        for _ in range(int(steps)):
            obs, actions, rtg, ts, mask = unpack_batch_continuous(next(loader))
            pred = self.model(obs, actions, rtg, ts, attention_mask=mask)
            loss = masked_mse(pred, actions, mask)

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()

        return {}

    def after_task(self, task_trajs: List[Trajectory]):
        return
