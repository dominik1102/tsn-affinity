from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from dt.dataset import Trajectory

from .tsn_common import _iter_tsn_modules
from .tsn_original_reuse_panda import (
    TSNOriginalReusePandaStrategy,
    _infer_raw_act_dim,
    _compute_obs_stats,
    _masked_mse_with_action_mask,
)
from .utils import make_panda_loader, prepare_panda_trajs, unpack_batch_continuous


"""
Patched Panda improved-reuse TSN.

Fixes in this version:
  1) __init__ accepts warmstart_on_new_copy, so it no longer leaks into the
     parent __init__ and raises TypeError;
  2) warmstart_on_new_copy really works by reading masks from the SOURCE copy
     instead of the newly created empty copy;
  3) latent routing explicitly uses model.eval();
  4) hybrid routing has a valid single-source fallback instead of collapsing
     to 0.0 after min-max normalization.
"""


class TSNImprovedReusePandaStrategy(TSNOriginalReusePandaStrategy):
    def __init__(
        self,
        *args,
        reuse_score_mode: str = "action",   # action | latent | hybrid
        routing_n_batches: int = 4,
        routing_batch_size: int = 64,
        action_reuse_threshold: float = 0.10,
        latent_reuse_threshold: float = 25.0,
        hybrid_reuse_threshold: float = 0.50,
        hybrid_alpha: float = 0.70,
        normalize_similarity_scores: bool = True,
        warmstart_source_scores: bool = True,
        warmstart_strength: float = 2.0,
        warmstart_noise_std: float = 0.02,
        warmstart_on_new_copy: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.reuse_score_mode = str(reuse_score_mode)
        self.routing_n_batches = int(routing_n_batches)
        self.routing_batch_size = int(routing_batch_size)
        self.action_reuse_threshold = float(action_reuse_threshold)
        self.latent_reuse_threshold = float(latent_reuse_threshold)
        self.hybrid_reuse_threshold = float(hybrid_reuse_threshold)
        self.hybrid_alpha = float(hybrid_alpha)
        self.normalize_similarity_scores = bool(normalize_similarity_scores)
        self.warmstart_source_scores = bool(warmstart_source_scores)
        self.warmstart_strength = float(warmstart_strength)
        self.warmstart_noise_std = float(warmstart_noise_std)
        self.warmstart_on_new_copy = bool(warmstart_on_new_copy)

        self.task_latent_stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # Latent helpers
    # ------------------------------------------------------------------
    def _extract_obs_latents(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() != 3:
            raise ValueError(f"Expected obs [B,L,D], got {tuple(obs.shape)}")

        x = obs.to(self.device, dtype=torch.float32)

        if hasattr(self.model, "obs_mean") and hasattr(self.model, "obs_std"):
            mean = self.model.obs_mean.view(1, 1, -1).to(device=x.device, dtype=x.dtype)
            std = self.model.obs_std.view(1, 1, -1).to(device=x.device, dtype=x.dtype)
            x = (x - mean) / std

        B, L, D = x.shape
        flat = x.reshape(B * L, D)
        z = self.model.obs_enc(flat).reshape(B, L, -1)
        return z

    @staticmethod
    def _diag_stats(z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if z.dim() == 3:
            if mask is None:
                x = z.reshape(-1, z.shape[-1])
            else:
                keep = mask.reshape(-1).bool()
                x = z.reshape(-1, z.shape[-1])[keep]
        else:
            x = z

        if x.numel() == 0:
            H = z.shape[-1]
            return torch.zeros(H, device=z.device), torch.ones(H, device=z.device)

        mu = x.mean(dim=0)
        var = x.var(dim=0, unbiased=False).clamp(min=1e-6)
        return mu, var

    @staticmethod
    def _sym_kl_diag_gaussians(
        mu_a: torch.Tensor,
        var_a: torch.Tensor,
        mu_b: torch.Tensor,
        var_b: torch.Tensor,
    ) -> torch.Tensor:
        kl_ab = 0.5 * torch.sum(torch.log(var_b / var_a) + (var_a + (mu_a - mu_b).pow(2)) / var_b - 1.0)
        kl_ba = 0.5 * torch.sum(torch.log(var_a / var_b) + (var_b + (mu_b - mu_a).pow(2)) / var_a - 1.0)
        return 0.5 * (kl_ab + kl_ba)

    def _compute_current_memory_latent_stats(self, task_memory_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            obs = task_memory_obs.to(self.device, dtype=torch.float32).unsqueeze(0)
            z = self._extract_obs_latents(obs)
            mu, var = self._diag_stats(z)
            return mu.detach().cpu(), var.detach().cpu()

    def _store_task_latent_stats(self, task_id: int) -> None:
        if int(task_id) not in self.task_memories:
            return
        mem = self.task_memories[int(task_id)]
        self.set_eval_task(int(task_id))
        mu, var = self._compute_current_memory_latent_stats(mem)
        self.task_latent_stats[int(task_id)] = (mu, var)
        self.clear_eval_task()

    # ------------------------------------------------------------------
    # Improved routing scores
    # ------------------------------------------------------------------
    def _make_routing_loader(self, task_trajs_pad: List[Trajectory]):
        return make_panda_loader(
            task_trajs_pad,
            seq_len=self.seq_len,
            batch_size=self.routing_batch_size,
            device=self.device,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
        )

    def _estimate_action_compatibility(
        self,
        source_task_id: int,
        task_trajs_pad: List[Trajectory],
        action_mask_t: torch.Tensor,
    ) -> float:
        self.set_eval_task(int(source_task_id))
        self.model.eval()

        loader = self._make_routing_loader(task_trajs_pad)
        vals: List[float] = []
        with torch.no_grad():
            for _ in range(max(1, self.routing_n_batches)):
                obs, actions, rtg, ts, mask = unpack_batch_continuous(next(loader))
                pred = self.model(obs, actions, rtg, ts, attention_mask=mask)
                mse = _masked_mse_with_action_mask(pred, actions, mask, action_mask_t)
                vals.append(float(mse.detach().cpu().item()))
        self.clear_eval_task()
        return float(np.mean(vals)) if vals else float("inf")

    def _estimate_latent_similarity(
        self,
        source_task_id: int,
        task_memory_obs: torch.Tensor,
    ) -> float:
        if int(source_task_id) not in self.task_latent_stats:
            return float("inf")

        self.set_eval_task(int(source_task_id))
        self.model.eval()
        mu_ref, var_ref = self.task_latent_stats[int(source_task_id)]
        with torch.no_grad():
            mu_new, var_new = self._compute_current_memory_latent_stats(task_memory_obs)
            skl = self._sym_kl_diag_gaussians(
                mu_new.to(self.device),
                var_new.to(self.device),
                mu_ref.to(self.device),
                var_ref.to(self.device),
            )
        self.clear_eval_task()
        return float(skl.detach().cpu().item())

    @staticmethod
    def _normalize_score_dict(values: Dict[int, float]) -> Dict[int, float]:
        if not values:
            return {}
        arr = np.array(list(values.values()), dtype=np.float32)
        vmin = float(arr.min())
        vmax = float(arr.max())
        if abs(vmax - vmin) < 1e-12:
            return {k: 0.0 for k in values}
        return {k: float((v - vmin) / (vmax - vmin)) for k, v in values.items()}

    def _single_source_hybrid_score(self, action_score: float, latent_score: float) -> float:
        a_ratio = float(action_score / max(self.action_reuse_threshold, 1e-12))
        l_ratio = float(latent_score / max(self.latent_reuse_threshold, 1e-12))
        return float(self.hybrid_alpha * a_ratio + (1.0 - self.hybrid_alpha) * l_ratio)

    def _select_copy_for_new_task_improved(
        self,
        task_memory_obs: torch.Tensor,
        task_trajs_pad: List[Trajectory],
        action_mask_t: torch.Tensor,
    ) -> Tuple[int, Optional[int], Dict[str, Optional[float]], bool]:
        if self.current_task_id == 0 or not self.task_to_copy:
            return 0, None, {"best_action": None, "best_latent": None, "best_score": None}, False

        action_scores: Dict[int, float] = {}
        latent_scores: Dict[int, float] = {}

        prev_tasks = sorted(self.task_to_copy.keys())
        for t in prev_tasks:
            action_scores[int(t)] = self._estimate_action_compatibility(int(t), task_trajs_pad, action_mask_t)
            if self.reuse_score_mode in ("latent", "hybrid"):
                latent_scores[int(t)] = self._estimate_latent_similarity(int(t), task_memory_obs)

        if self.reuse_score_mode == "action":
            final_scores = dict(action_scores)
            best_task = min(final_scores, key=final_scores.get)
            best_score = final_scores[best_task]
            create_new = best_score > self.action_reuse_threshold
            details = {
                "best_action": float(action_scores[best_task]),
                "best_latent": None,
                "best_score": float(best_score),
            }

        elif self.reuse_score_mode == "latent":
            final_scores = dict(latent_scores)
            best_task = min(final_scores, key=final_scores.get)
            best_score = final_scores[best_task]
            create_new = best_score > self.latent_reuse_threshold
            details = {
                "best_action": float(action_scores.get(best_task, float("nan"))),
                "best_latent": float(latent_scores[best_task]),
                "best_score": float(best_score),
            }

        elif self.reuse_score_mode == "hybrid":
            if len(prev_tasks) == 1:
                best_task = int(prev_tasks[0])
                best_score = self._single_source_hybrid_score(
                    action_scores[best_task],
                    latent_scores[best_task],
                )
                create_new = best_score > 1.0
            else:
                act_n = self._normalize_score_dict(action_scores) if self.normalize_similarity_scores else action_scores
                lat_n = self._normalize_score_dict(latent_scores) if self.normalize_similarity_scores else latent_scores
                final_scores = {
                    t: float(self.hybrid_alpha * act_n[t] + (1.0 - self.hybrid_alpha) * lat_n[t])
                    for t in prev_tasks
                }
                best_task = min(final_scores, key=final_scores.get)
                best_score = final_scores[best_task]
                create_new = best_score > self.hybrid_reuse_threshold
            details = {
                "best_action": float(action_scores[best_task]),
                "best_latent": float(latent_scores[best_task]),
                "best_score": float(best_score),
            }
        else:
            raise ValueError(f"Unsupported reuse_score_mode: {self.reuse_score_mode}")

        if create_new:
            if self.max_model_copies is not None and len(self.copy_states) >= self.max_model_copies:
                copy_id = self.task_to_copy[int(best_task)]
                return int(copy_id), int(best_task), details, False
            new_copy = self._make_fresh_copy()
            self.copy_states.append(new_copy)
            return len(self.copy_states) - 1, int(best_task), details, True

        return int(self.task_to_copy[int(best_task)]), int(best_task), details, False

    # ------------------------------------------------------------------
    # Warm-start from selected source mask
    # ------------------------------------------------------------------
    def _warmstart_scores_from_source_mask(self, source_task: Optional[int]) -> None:
        if (not self.warmstart_source_scores) or (source_task is None):
            return

        source_copy_id = self.task_to_copy.get(int(source_task), None)
        if source_copy_id is None:
            return

        src_state = self.copy_states[int(source_copy_id)]
        src_masks = src_state.per_task_masks.get(int(source_task), None)
        if src_masks is None:
            return

        with torch.no_grad():
            for name, mod in _iter_tsn_modules(self.model):
                w_key = f"{name}.weight"
                src_w = src_masks.get(w_key, None)
                if src_w is not None:
                    src_w = src_w.to(device=mod.score.device, dtype=mod.score.dtype)
                    mod.score.normal_(mean=0.0, std=self.warmstart_noise_std)
                    mod.score.add_(self.warmstart_strength * src_w)

                if getattr(mod, "bias_score", None) is not None:
                    b_key = f"{name}.bias"
                    src_b = src_masks.get(b_key, None)
                    if src_b is not None:
                        src_b = src_b.to(device=mod.bias_score.device, dtype=mod.bias_score.dtype)
                        mod.bias_score.normal_(mean=0.0, std=self.warmstart_noise_std)
                        mod.bias_score.add_(self.warmstart_strength * src_b)

    # ------------------------------------------------------------------
    # training API override
    # ------------------------------------------------------------------
    def train_task(
        self,
        task_trajs: List[Trajectory],
        steps: int = 2000,
        batch_size: int = 64,
        *,
        active_action_dim_mask: Optional[List[int]] = None,
    ):
        if not task_trajs:
            raise ValueError("TSNImprovedReusePandaStrategy.train_task got empty task_trajs")

        task_trajs_pad = prepare_panda_trajs(task_trajs, obs_dim=self.obs_dim, act_dim=self.act_dim)

        raw_act_dim = _infer_raw_act_dim(task_trajs, self.act_dim)
        if active_action_dim_mask is None:
            active_action_dim_mask = [1] * raw_act_dim + [0] * max(0, self.act_dim - raw_act_dim)
        if len(active_action_dim_mask) != self.act_dim:
            raise ValueError(
                f"active_action_dim_mask has len={len(active_action_dim_mask)} but global act_dim={self.act_dim}"
            )
        raw_act_dim = int(sum(int(x) for x in active_action_dim_mask))
        self._last_task_action_dim = int(raw_act_dim)
        action_mask_t = torch.tensor(active_action_dim_mask, dtype=torch.float32, device=self.device)

        if self.store_task_obs_stats:
            mean_np, std_np = _compute_obs_stats(task_trajs_pad, self.obs_dim)
            self._last_task_obs_stats = (mean_np.copy(), std_np.copy())
        else:
            self._last_task_obs_stats = None

        task_memory = self._build_task_memory(task_trajs_pad)
        copy_id, src_task, score_details, created_new = self._select_copy_for_new_task_improved(
            task_memory,
            task_trajs_pad,
            action_mask_t,
        )
        self._activate_copy(copy_id)
        self.task_similarity[self.current_task_id] = {
            "source_task": None if src_task is None else int(src_task),
            "copy_id": int(copy_id),
            "best_action": score_details.get("best_action", None),
            "best_latent": score_details.get("best_latent", None),
            "best_score": score_details.get("best_score", None),
            "score_mode": self.reuse_score_mode,
            "created_new_copy": bool(created_new),
        }

        if self.store_task_obs_stats and self._last_task_obs_stats is not None:
            mean_np, std_np = self._last_task_obs_stats
            if hasattr(self.model, "obs_mean") and hasattr(self.model, "obs_std"):
                with torch.no_grad():
                    self.model.obs_mean.copy_(torch.as_tensor(mean_np, dtype=self.model.obs_mean.dtype, device=self.model.obs_mean.device))
                    self.model.obs_std.copy_(torch.as_tensor(std_np, dtype=self.model.obs_std.dtype, device=self.model.obs_std.device))

        self._prepare_current_task()
        if src_task is not None:
            if (not created_new) or self.warmstart_on_new_copy:
                self._warmstart_scores_from_source_mask(src_task)

        loader = make_panda_loader(
            task_trajs_pad,
            seq_len=self.seq_len,
            batch_size=batch_size,
            device=self.device,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
        )

        self.model.train()
        last_loss = None
        for it in range(int(steps)):
            obs, actions, rtg, ts, mask = unpack_batch_continuous(next(loader))
            pred = self.model(obs, actions, rtg, ts, attention_mask=mask)
            loss = _masked_mse_with_action_mask(pred, actions, mask, action_mask_t)

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self._zero_prev_task_param_grads()
            self._zero_non_maskable_grads()
            frozen_snapshot = self._snapshot_frozen_params()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()
            self._restore_frozen_params_(frozen_snapshot)

            last_loss = float(loss.detach().item())
            if it % 20000 == 0 or it == int(steps) - 1:
                meta = self.task_similarity[self.current_task_id]
                print(
                    f"[tsn-improved-reuse-panda] task={self.current_task_id} copy={meta['copy_id']} "
                    f"src={meta['source_task']} score_mode={meta['score_mode']} "
                    f"best_action={meta['best_action']} best_latent={meta['best_latent']} "
                    f"best_score={meta['best_score']} new_copy={int(meta['created_new_copy'])} "
                    f"it={it} bc={last_loss:.6e} keep_ratio={self.current_keep_ratio:.4f} "
                    f"active_action_dim_mask={active_action_dim_mask}"
                )

        self.task_memories[self.current_task_id] = task_memory.detach().cpu()
        return {
            "bc_loss": last_loss,
            "keep_ratio": float(self.current_keep_ratio),
        }

    def after_task(self, task_trajs: List[Trajectory]):
        super().after_task(task_trajs)
        finished_task = int(self.current_task_id - 1)
        self._store_task_latent_stats(finished_task)
