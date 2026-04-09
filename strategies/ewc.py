from __future__ import annotations

from typing import Dict, List, Optional

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


class EWCStrategy(BaseStrategy):
    """
    Atari / discrete EWC with aligned DT actions.
    """

    def __init__(
        self,
        *args,
        ewc_lambda: float = 50.0,
        fisher_n_batches: int = 50,
        fisher_batch_size: int = 8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ewc_lambda = float(ewc_lambda)
        self.fisher_n_batches = int(fisher_n_batches)
        self.fisher_batch_size = int(fisher_batch_size)

        self.prev_params: Dict[str, torch.Tensor] = {}
        self.fisher_diag: Dict[str, torch.Tensor] = {}

    def _ewc_loss(self) -> torch.Tensor:
        if not self.fisher_diag:
            return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)
        for n, p in self.model.named_parameters():
            if (n in self.fisher_diag) and (n in self.prev_params):
                loss = loss + torch.sum(self.fisher_diag[n] * (p - self.prev_params[n]) ** 2)

        return (self.ewc_lambda / 2.0) * loss

    def train_task(
        self,
        task_trajs: List[Trajectory],
        steps: int = 2000,
        batch_size: int = 64,
    ):
        loader = make_minibatches(task_trajs, self.seq_len, batch_size, self.device)
        self.model.train()

        for _ in range(int(steps)):
            obs, actions, rtg, ts, mask = unpack_batch_discrete(next(loader))
            logits = self.model(obs, actions, rtg, ts, attention_mask=mask)

            bc = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                actions.reshape(-1),
                ignore_index=-1,
            )
            loss = bc + self._ewc_loss()

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()

        return {}

    def _estimate_fisher(
        self,
        trajs: List[Trajectory],
        n_batches: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        self.model.eval()

        n_batches = self.fisher_n_batches if n_batches is None else int(n_batches)
        batch_size = self.fisher_batch_size if batch_size is None else int(batch_size)

        fisher = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        loader = make_minibatches(trajs, self.seq_len, batch_size, self.device)
        for _ in range(max(1, n_batches)):
            obs, actions, rtg, ts, mask = unpack_batch_discrete(next(loader))
            self.model.zero_grad(set_to_none=True)

            logits = self.model(obs, actions, rtg, ts, attention_mask=mask)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                actions.reshape(-1),
                ignore_index=-1,
            )
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None and n in fisher:
                    fisher[n] += p.grad.detach() ** 2

        for n in fisher:
            fisher[n] /= float(max(1, n_batches))

        with torch.no_grad():
            self.fisher_diag = {n: f.clone() for n, f in fisher.items()}
            self.prev_params = {
                n: p.detach().clone()
                for n, p in self.model.named_parameters()
                if p.requires_grad
            }

    def after_task(self, task_trajs: List[Trajectory]):
        self._estimate_fisher(task_trajs)

class PandaEWCStrategy:
    """
    Continuous-action Panda EWC.
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
        ewc_lambda: float = 50.0,
        fisher_n_batches: int = 50,
        fisher_batch_size: int = 32,
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
            "ewc_lambda": float(ewc_lambda),
            "fisher_n_batches": int(fisher_n_batches),
            "fisher_batch_size": int(fisher_batch_size),
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

        self.ewc_lambda = float(ewc_lambda)
        self.fisher_n_batches = int(fisher_n_batches)
        self.fisher_batch_size = int(fisher_batch_size)

        self.prev_params: Dict[str, torch.Tensor] = {}
        self.fisher_diag: Dict[str, torch.Tensor] = {}

    def _prepare_trajs(self, trajs: List[Trajectory]) -> List[Trajectory]:
        return prepare_panda_trajs(trajs, obs_dim=self.obs_dim, act_dim=self.act_dim)

    def _ewc_loss(self) -> torch.Tensor:
        if not self.fisher_diag:
            return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)
        for n, p in self.model.named_parameters():
            if (n in self.fisher_diag) and (n in self.prev_params):
                loss = loss + torch.sum(self.fisher_diag[n] * (p - self.prev_params[n]) ** 2)

        return (self.ewc_lambda / 2.0) * loss

    def train_task(self, task_trajs: List[Trajectory], steps: int = 2000, batch_size: int = 64):
        task_trajs = self._prepare_trajs(task_trajs)
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

            bc = masked_mse(pred, actions, mask)
            loss = bc + self._ewc_loss()

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()

        return {}

    def _estimate_fisher(
        self,
        trajs: List[Trajectory],
        *,
        n_batches: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        n_batches = int(self.fisher_n_batches if n_batches is None else n_batches)
        batch_size = int(self.fisher_batch_size if batch_size is None else batch_size)

        trajs = self._prepare_trajs(trajs)
        self.model.eval()

        fisher: Dict[str, torch.Tensor] = {
            n: torch.zeros_like(p, device=self.device)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        loader = make_panda_loader(
            trajs,
            seq_len=self.seq_len,
            batch_size=batch_size,
            device=self.device,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
        )

        for _ in range(n_batches):
            obs, actions, rtg, ts, mask = unpack_batch_continuous(next(loader))
            self.model.zero_grad(set_to_none=True)

            pred = self.model(obs, actions, rtg, ts, attention_mask=mask)
            loss = masked_mse(pred, actions, mask)
            loss.backward()

            for n, p in self.model.named_parameters():
                if (p.grad is not None) and (n in fisher):
                    fisher[n] += (p.grad.detach() ** 2)

        for n in fisher:
            fisher[n] /= float(max(1, n_batches))

        with torch.no_grad():
            self.fisher_diag = {n: f.detach().clone() for n, f in fisher.items()}
            self.prev_params = {
                n: p.detach().clone()
                for n, p in self.model.named_parameters()
                if n in self.fisher_diag
            }

    def after_task(self, task_trajs: List[Trajectory]):
        self._estimate_fisher(task_trajs)
        return
