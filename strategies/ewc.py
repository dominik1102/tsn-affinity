from __future__ import annotations
from typing import List, Dict
import torch, torch.nn as nn, torch.nn.functional as F
from .base import BaseStrategy
from dt.dataset import Trajectory, make_minibatches

class EWCStrategy(BaseStrategy):
    def __init__(self, *args, ewc_lambda: float = 50.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ewc_lambda = ewc_lambda
        self.prev_params: Dict[str, torch.Tensor] = {}
        self.fisher_diag: Dict[str, torch.Tensor] = {}

    def _ewc_loss(self) -> torch.Tensor:
        if not self.fisher_diag:
            return torch.tensor(0.0, device=self.device)
        loss = 0.0
        for n, p in self.model.named_parameters():
            if n in self.fisher_diag:
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
        for _ in range(steps):
            obs, actions, rtg, ts = next(loader)
            logits = self.model(obs, torch.roll(actions, shifts=1, dims=1), rtg, ts)
            bc = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                actions.reshape(-1),
                ignore_index=-1,
            )
            loss = bc + self._ewc_loss()
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
        return {}

    def _estimate_fisher(
        self,
        trajs: List[Trajectory],
        n_batches: int = 50,
        batch_size: int = 8,
    ):
        # usually we want deterministic behavior here (no dropout noise)
        self.model.eval()

        fisher = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        loader = make_minibatches(trajs, self.seq_len, batch_size, self.device)
        for _ in range(n_batches):
            obs, actions, rtg, ts = next(loader)
            self.model.zero_grad(set_to_none=True)

            logits = self.model(obs, torch.roll(actions, shifts=1, dims=1), rtg, ts)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                actions.reshape(-1),
                ignore_index=-1,
            )
            loss.backward()

            # accumulate squared gradients
            for n, p in self.model.named_parameters():
                if p.grad is not None and n in fisher:
                    fisher[n] += p.grad.detach() ** 2

        for n in fisher:
            fisher[n] /= float(n_batches)

        # store Fisher diag and parameter snapshot without tracking grads
        with torch.no_grad():
            self.fisher_diag = {n: f.clone() for n, f in fisher.items()}
            self.prev_params = {
                n: p.detach().clone() for n, p in self.model.named_parameters()
            }

    def after_task(self, task_trajs: List[Trajectory]):
        self._estimate_fisher(task_trajs)
