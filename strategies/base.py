from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

from dt.dataset import Trajectory, make_minibatches
from dt.model import DecisionTransformer
from .utils import unpack_batch_discrete


class BaseStrategy:
    """
    Base strategy for discrete-action Atari Decision Transformer training.

    This class implements the default "naive" single-task offline DT training
    loop with:
      - aligned actions (no torch.roll),
      - reconstructed attention masks when the loader does not provide them,
      - configurable model/optimizer hyperparameters.

    Other Atari strategies (naive / cumulative / EWC / SI / TSN) reuse this
    class and override only the continual-learning-specific parts.
    """

    def __init__(
        self,
        obs_shape,
        n_actions: int,
        seq_len: int = 20,
        device: str = "cuda",
        *,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        p_drop: float = 0.1,
        max_ep_len: int = 10000,
        rtg_scale: float = 1000.0,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
    ):
        self.device = device
        self.seq_len = int(seq_len)
        self.grad_clip = float(grad_clip)

        self.model_hparams = {
            "obs_shape": tuple(obs_shape),
            "n_actions": int(n_actions),
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

        self.model = DecisionTransformer(
            obs_shape=obs_shape,
            n_actions=n_actions,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            seq_len=seq_len,
            p_drop=p_drop,
            max_ep_len=max_ep_len,
            rtg_scale=rtg_scale,
        ).to(device)

        self.opt = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def train_task(self, task_trajs: List[Trajectory], steps: int = 2000, batch_size: int = 64):
        """
        Default discrete offline DT training loop for one task.
        """
        loader = make_minibatches(task_trajs, self.seq_len, batch_size, self.device)
        self.model.train()

        for _ in range(int(steps)):
            obs, actions, rtg, ts, mask = unpack_batch_discrete(next(loader))

            logits = self.model(obs, actions, rtg, ts, attention_mask=mask)
            loss = nn.functional.cross_entropy(
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
        pass
