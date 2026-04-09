from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from dt.dataset import Trajectory, make_minibatches
from dt.panda_dt import PandaDecisionTransformer
from .base import BaseStrategy
from .utils import (
    infer_active_action_mask,
    make_panda_loader,
    masked_mse,
    prepare_panda_trajs,
    unpack_batch_continuous,
    unpack_batch_discrete,
)

class SIStrategy(BaseStrategy):
    """
    Atari / discrete Synaptic Intelligence with aligned DT actions.

    Important fix:
      - path integral `w` is accumulated using gradients of the task loss (bc)
        only, not gradients of the total loss (bc + si_reg).
    """

    def __init__(
        self,
        *args,
        si_lambda: float = 1.0,
        si_epsilon: float = 0.1,
        clamp_omega: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.si_lambda = float(si_lambda)
        self.si_epsilon = float(si_epsilon)
        self.clamp_omega = bool(clamp_omega)

        self.theta_star: Dict[str, torch.Tensor] = {}
        self.omega: Dict[str, torch.Tensor] = {}

        self._w: Dict[str, torch.Tensor] = {}
        self._theta_start: Dict[str, torch.Tensor] = {}
        self._theta_prev_step: Dict[str, torch.Tensor] = {}

    def _si_loss(self) -> torch.Tensor:
        if (not self.omega) or (not self.theta_star):
            return torch.tensor(0.0, device=self.device)

        reg = torch.tensor(0.0, device=self.device)
        for n, p in self.model.named_parameters():
            if (n in self.omega) and (n in self.theta_star):
                reg = reg + torch.sum(self.omega[n] * (p - self.theta_star[n]) ** 2)

        return (self.si_lambda / 2.0) * reg

    @torch.no_grad()
    def _begin_task(self) -> None:
        self._w.clear()
        self._theta_start.clear()
        self._theta_prev_step.clear()

        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            self._w[n] = torch.zeros_like(p)
            self._theta_start[n] = p.detach().clone()
            self._theta_prev_step[n] = p.detach().clone()

        if not self.theta_star:
            self.theta_star = {
                n: p.detach().clone()
                for n, p in self.model.named_parameters()
                if p.requires_grad
            }
        if not self.omega:
            self.omega = {
                n: torch.zeros_like(p)
                for n, p in self.model.named_parameters()
                if p.requires_grad
            }

    @torch.no_grad()
    def _accumulate_w_after_step(self, bc_grads: Dict[str, Optional[torch.Tensor]]) -> None:
        """
        Call AFTER optimizer.step().

        Uses parameter change delta and gradients of task loss only
        (not gradients of the SI-regularized total loss).
        """
        for n, p in self.model.named_parameters():
            if n not in self._w:
                continue

            old = self._theta_prev_step[n]
            new = p.detach()
            delta = new - old

            g = bc_grads.get(n, None)
            if g is not None:
                self._w[n].add_(-g * delta)

            old.copy_(new)

    def train_task(
        self,
        task_trajs: List[Trajectory],
        steps: int = 2000,
        batch_size: int = 64,
    ):
        self._begin_task()

        loader = make_minibatches(task_trajs, self.seq_len, batch_size, self.device)
        self.model.train()

        named_params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        params_only = [p for _, p in named_params]

        for _ in range(int(steps)):
            obs, actions, rtg, ts, mask = unpack_batch_discrete(next(loader))
            logits = self.model(obs, actions, rtg, ts, attention_mask=mask)

            bc = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                actions.reshape(-1),
                ignore_index=-1,
            )

            # grads of task loss only -> used for SI path integral
            bc_grads_list = torch.autograd.grad(
                bc,
                params_only,
                retain_graph=True,
                allow_unused=True,
            )
            bc_grads = {}
            for (n, _), g in zip(named_params, bc_grads_list):
                bc_grads[n] = None if g is None else g.detach()

            loss = bc + self._si_loss()

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()

            self._accumulate_w_after_step(bc_grads)

        return {}

    @torch.no_grad()
    def after_task(self, task_trajs: List[Trajectory]):
        for n, p in self.model.named_parameters():
            if n not in self._w:
                continue

            delta_total = p.detach() - self._theta_start[n]
            denom = delta_total.pow(2).add(self.si_epsilon)
            omega_add = self._w[n] / denom

            if self.clamp_omega:
                omega_add = torch.clamp(omega_add, min=0.0)

            self.omega[n] = self.omega.get(n, torch.zeros_like(p)) + omega_add

        self.theta_star = {
            n: p.detach().clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        self._w.clear()
        self._theta_start.clear()
        self._theta_prev_step.clear()


class PandaSIStrategy:
    """
    Panda / continuous Synaptic Intelligence.
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
        si_lambda: float = 0.5,
        si_epsilon: float = 0.1,
        clamp_min0: bool = True,
        omega_max: Optional[float] = 10.0,
        debug_every: int = 0,
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
            "si_lambda": float(si_lambda),
            "si_epsilon": float(si_epsilon),
            "omega_max": None if omega_max is None else float(omega_max),
            "debug_every": int(debug_every),
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

        self.si_lambda = float(si_lambda)
        self.si_epsilon = float(si_epsilon)
        self.clamp_min0 = bool(clamp_min0)
        self.omega_max = float(omega_max) if omega_max is not None else None
        self.debug_every = int(debug_every)

        self.theta_star: Dict[str, torch.Tensor] = {}
        self.omega: Dict[str, torch.Tensor] = {}

        self._w: Dict[str, torch.Tensor] = {}
        self._theta_start: Dict[str, torch.Tensor] = {}
        self._theta_prev_step: Dict[str, torch.Tensor] = {}

        self._seen_act_mask = torch.zeros(self.act_dim, dtype=torch.float32, device=self.device)
        self._cur_act_mask: Optional[torch.Tensor] = None

    def _prepare_trajs(self, trajs: List[Trajectory]) -> List[Trajectory]:
        return prepare_panda_trajs(trajs, obs_dim=self.obs_dim, act_dim=self.act_dim)

    def _is_action_head_like(self, p: torch.Tensor) -> bool:
        return (p.requires_grad and p.ndim in (1, 2) and p.shape[0] == self.act_dim)

    def _si_loss(self) -> torch.Tensor:
        if (not self.omega) or (not self.theta_star):
            return torch.tensor(0.0, device=self.device)

        reg = torch.tensor(0.0, device=self.device)
        for n, p in self.model.named_parameters():
            if (n not in self.omega) or (n not in self.theta_star):
                continue

            om = self.omega[n]
            diff = (p - self.theta_star[n])

            if self._is_action_head_like(p):
                seen = self._seen_act_mask.view(-1, *([1] * (p.ndim - 1)))
                om = om * seen
                diff = diff * seen

            reg = reg + torch.sum(om * diff * diff)

        return (self.si_lambda / 2.0) * reg

    @torch.no_grad()
    def _begin_task(self) -> None:
        self._w.clear()
        self._theta_start.clear()
        self._theta_prev_step.clear()

        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            self._w[n] = torch.zeros_like(p)
            self._theta_start[n] = p.detach().clone()
            self._theta_prev_step[n] = p.detach().clone()

        if not self.theta_star:
            self.theta_star = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}
        if not self.omega:
            self.omega = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def _accumulate_w_after_step(self, bc_grads: Dict[str, Optional[torch.Tensor]]) -> None:
        for n, p in self.model.named_parameters():
            if n not in self._w:
                continue

            old = self._theta_prev_step[n]
            new = p.detach()
            delta = new - old

            g = bc_grads.get(n, None)
            if g is not None:
                if self._cur_act_mask is not None and self._is_action_head_like(p):
                    am = self._cur_act_mask.view(-1, *([1] * (p.ndim - 1)))
                    self._w[n].add_(-(g * am) * (delta * am))
                else:
                    self._w[n].add_(-g * delta)

            old.copy_(new)

    def train_task(self, task_trajs: List[Trajectory], *, steps: int = 2000, batch_size: int = 64) -> dict:
        self._begin_task()

        task_trajs = self._prepare_trajs(task_trajs)
        act_mask = infer_active_action_mask(task_trajs, self.act_dim).to(self.device)
        self._cur_act_mask = act_mask
        print(f"[si] active_action_dim_mask={act_mask.int().tolist()} si_lambda={self.si_lambda} si_eps={self.si_epsilon}")

        loader = make_panda_loader(
            task_trajs,
            seq_len=self.seq_len,
            batch_size=int(batch_size),
            device=self.device,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
        )

        named_params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        params_only = [p for _, p in named_params]

        self.model.train()

        for it in range(int(steps)):
            obs, actions, rtg, ts, mask = unpack_batch_continuous(next(loader))
            pred = self.model(obs, actions, rtg, ts, attention_mask=mask)

            bc = masked_mse(pred, actions, mask, action_dim_mask=act_mask)
            reg = self._si_loss()
            loss = bc + reg

            bc_grads_list = torch.autograd.grad(bc, params_only, retain_graph=True, allow_unused=True)
            bc_grads = {}
            for (n, _), g in zip(named_params, bc_grads_list):
                bc_grads[n] = None if g is None else g.detach()

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()

            self._accumulate_w_after_step(bc_grads)

            if self.debug_every > 0 and (it % self.debug_every == 0 or it == int(steps) - 1):
                bc_v = float(bc.detach().cpu().item())
                reg_v = float(reg.detach().cpu().item())
                ratio = reg_v / max(bc_v, 1e-12)
                print(f"[si] it={it} bc={bc_v:.6e} reg={reg_v:.6e} ratio={ratio:.2f}")

        return {}

    @torch.no_grad()
    def after_task(self, task_trajs: List[Trajectory]) -> None:
        if self._cur_act_mask is not None:
            self._seen_act_mask = torch.maximum(self._seen_act_mask, self._cur_act_mask)

        for n, p in self.model.named_parameters():
            if n not in self._w:
                continue

            delta_total = p.detach() - self._theta_start[n]
            denom = delta_total.pow(2).add(self.si_epsilon)
            omega_add = self._w[n] / denom

            if self.clamp_min0:
                omega_add = torch.clamp(omega_add, min=0.0)

            if self._cur_act_mask is not None and self._is_action_head_like(p):
                am = self._cur_act_mask.view(-1, *([1] * (p.ndim - 1)))
                omega_add = omega_add * am

            self.omega[n] = self.omega.get(n, torch.zeros_like(p)) + omega_add

            if self.omega_max is not None:
                self.omega[n] = torch.clamp(self.omega[n], max=self.omega_max)

        self.theta_star = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}

        self._w.clear()
        self._theta_start.clear()
        self._theta_prev_step.clear()
        self._cur_act_mask = None
