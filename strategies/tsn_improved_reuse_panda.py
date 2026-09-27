from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import copy

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
     to 0.0 after min-max normalization;
  5) optional probe-guided routing: before committing a route, the strategy
     can briefly train several reuse plans and select the one with the best
     held-out offline validation loss under copy/residual penalties;
  6) probe candidates are compared with deterministic RNG and train/validation
     splits, so source/copy decisions are less noisy;
  7) residual reuse can be forced to use a source mask plus new/free delta
     parameters, instead of accidentally adding arbitrary occupied weights.
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
        routing_policy: str = "affinity",
        copy_penalty: float = 0.0,
        occupancy_penalty: float = 0.0,
        reuse_margin: float = 0.0,
        residual_reuse: bool = False,
        residual_keep_ratio: float = 0.10,
        dynamic_occ_target: float = 0.80,
        dynamic_occ_open: float = 0.84,
        dynamic_occ_hard: float = 0.90,
        dynamic_occ_lambda: float = 30.0,
        dynamic_full_lambda: float = 20.0,
        dynamic_copy_margin: float = 0.25,
        dynamic_delay_new_until_task: int = 1,
        dynamic_rho_high: float = 0.50,
        dynamic_rho_mid: float = 0.25,
        dynamic_rho_low: float = 0.10,
        dynamic_low_occ: float = 0.60,
        dynamic_mid_occ: float = 0.80,
        anchor_first_reuse: bool = False,
        anchor_first_reuse_rho: float = 0.50,
        anchor_first_reuse_source_task: int = 0,
        probe_routing: bool = False,
        probe_steps: int = 0,
        probe_top_k: int = 2,
        probe_batch_size: int = 64,
        probe_val_batches: int = 4,
        probe_residual_grid: str = "0,0.5,0.75",
        probe_residual_penalty: float = 0.0,
        probe_include_new_copy: bool = True,
        probe_val_fraction: float = 0.10,
        probe_seed: int = 123000,
        probe_deterministic: bool = True,
        residual_delta_free_only: bool = True,
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
        self.routing_policy = str(routing_policy)
        self.copy_penalty = float(copy_penalty)
        self.occupancy_penalty = float(occupancy_penalty)
        self.reuse_margin = float(reuse_margin)
        self.residual_reuse = bool(residual_reuse)
        self.residual_keep_ratio = float(residual_keep_ratio)

        # Dynamic occupancy-aware routing.  This mirrors the Atari DynOcc
        # route selector but keeps the Panda-specific action masks and losses.
        self.dynamic_occ_target = float(dynamic_occ_target)
        self.dynamic_occ_open = float(dynamic_occ_open)
        self.dynamic_occ_hard = float(dynamic_occ_hard)
        self.dynamic_occ_lambda = float(dynamic_occ_lambda)
        self.dynamic_full_lambda = float(dynamic_full_lambda)
        self.dynamic_copy_margin = float(dynamic_copy_margin)
        self.dynamic_delay_new_until_task = int(dynamic_delay_new_until_task)
        self.dynamic_rho_high = float(dynamic_rho_high)
        self.dynamic_rho_mid = float(dynamic_rho_mid)
        self.dynamic_rho_low = float(dynamic_rho_low)
        self.dynamic_low_occ = float(dynamic_low_occ)
        self.dynamic_mid_occ = float(dynamic_mid_occ)
        self.anchor_first_reuse = bool(anchor_first_reuse)
        self.anchor_first_reuse_rho = float(anchor_first_reuse_rho)
        self.anchor_first_reuse_source_task = int(anchor_first_reuse_source_task)

        self.probe_routing = bool(probe_routing)
        self.probe_steps = int(probe_steps)
        self.probe_top_k = int(probe_top_k)
        self.probe_batch_size = int(probe_batch_size)
        self.probe_val_batches = int(probe_val_batches)
        self.probe_residual_grid = str(probe_residual_grid)
        self.probe_residual_penalty = float(probe_residual_penalty)
        self.probe_include_new_copy = bool(probe_include_new_copy)
        self.probe_val_fraction = float(probe_val_fraction)
        self.probe_seed = int(probe_seed)
        self.probe_deterministic = bool(probe_deterministic)
        self.residual_delta_free_only = bool(residual_delta_free_only)

        # If residual_delta_free_only=True we temporarily set TSN modules to
        # disallow generic occupied-weight reuse while a residual base is active.
        # This makes the candidate mask exactly: source_mask OR delta_over_free_weights.
        self._residual_old_allow_weight_reuse: Dict[str, bool] = {}

        self.task_latent_stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}


    # ------------------------------------------------------------------
    # Capacity-aware / residual-reuse helpers
    # ------------------------------------------------------------------
    def _copy_occupied_ratio(self, copy_id: int) -> float:
        try:
            st = self.copy_states[int(copy_id)]
            used = 0
            total = 0
            for key, mask in st.consolidated_masks.items():
                if mask is None or not key.endswith(".weight"):
                    continue
                used += int(mask.sum().item())
                total += int(mask.numel())
            return float(used / max(1, total))
        except Exception:
            return 0.0

    def _capacity_budget(self) -> int:
        if self.max_model_copies is not None:
            return max(1, int(self.max_model_copies))
        return max(1, len(self.copy_states))

    def _source_task_for_copy(self, copy_id: int, prev_tasks: List[int]) -> Optional[int]:
        for t in reversed(list(prev_tasks)):
            if int(self.task_to_copy.get(int(t), -1)) == int(copy_id):
                return int(t)
        return None

    def _select_copy_by_policy(self, prev_tasks: List[int]) -> Tuple[int, Optional[int], Dict[str, Optional[float]], bool]:
        policy = self.routing_policy
        budget = self._capacity_budget()
        details = {"best_action": None, "best_latent": None, "best_score": None, "best_kl": None}

        def _new_copy(src: Optional[int] = None):
            new_copy = self._make_fresh_copy()
            self.copy_states.append(new_copy)
            return len(self.copy_states) - 1, src, details, True

        if policy == "always_new":
            if self.max_model_copies is None or len(self.copy_states) < budget:
                return _new_copy(None)
            cid = int(self.current_task_id % len(self.copy_states))
            return cid, self._source_task_for_copy(cid, prev_tasks), details, False

        if policy == "round_robin":
            desired = int(self.current_task_id % budget)
            if desired >= len(self.copy_states):
                return _new_copy(None)
            return desired, self._source_task_for_copy(desired, prev_tasks), details, False

        if policy == "random":
            if len(self.copy_states) < budget:
                return _new_copy(None)
            rng = np.random.RandomState(12345 + int(self.current_task_id))
            cid = int(rng.randint(0, len(self.copy_states)))
            return cid, self._source_task_for_copy(cid, prev_tasks), details, False

        raise ValueError(f"Unsupported non-affinity routing_policy={policy!r}")

    def _should_create_new(self, best_task: int, score: float, threshold: float) -> bool:
        if self.routing_policy == "always_reuse":
            return False
        if self.routing_policy == "always_new":
            if self.max_model_copies is None:
                return True
            return len(self.copy_states) < int(self.max_model_copies)
        threshold = float(max(float(threshold), 1e-12))
        ratio = float(score) / threshold
        copy_id = int(self.task_to_copy.get(int(best_task), 0))
        occ = self._copy_occupied_ratio(copy_id)
        adjusted = ratio + self.occupancy_penalty * occ
        return bool(adjusted > (1.0 + self.copy_penalty + self.reuse_margin))

    # ------------------------------------------------------------------
    # Dynamic occupancy-aware routing (no probe)
    # ------------------------------------------------------------------
    def _threshold_for_current_mode(self) -> float:
        if self.reuse_score_mode == "action":
            return float(self.action_reuse_threshold)
        if self.reuse_score_mode == "latent":
            return float(self.latent_reuse_threshold)
        if self.reuse_score_mode == "hybrid":
            return float(self.hybrid_reuse_threshold)
        return float("inf")

    def _dynamic_residual_keep_ratio(self, copy_occ: float, score: float) -> float:
        """Choose a residual keep ratio from current copy occupancy.

        Low-occupancy copies can accept a larger residual delta.  Crowded copies
        receive a smaller delta to preserve capacity.  The score argument is kept
        for API symmetry with Atari and for future score-conditioned schedules.
        """
        _ = float(score)
        occ = float(max(0.0, min(1.0, copy_occ)))
        if occ <= float(self.dynamic_low_occ):
            return float(max(0.0, min(1.0, self.dynamic_rho_high)))
        if occ <= float(self.dynamic_mid_occ):
            return float(max(0.0, min(1.0, self.dynamic_rho_mid)))
        return float(max(0.0, min(1.0, self.dynamic_rho_low)))

    @staticmethod
    def _dynamic_projected_occupancy(copy_occ: float, rho: float) -> float:
        occ = float(max(0.0, min(1.0, copy_occ)))
        rho = float(max(0.0, min(1.0, rho)))
        return float(min(1.0, occ + rho * max(0.0, 1.0 - occ)))

    def _anchor_first_transfer_route(
        self,
        details: Dict[str, Optional[float]],
    ) -> Optional[Tuple[int, Optional[int], Dict[str, Optional[float]], bool]]:
        """Optional deterministic anchor for the first transfer step.

        This is disabled by default for Panda.  If enabled, task 1 reuses the
        configured source task copy and uses a high residual ratio.  It is useful
        only as a controlled diagnostic.
        """
        if not bool(self.anchor_first_reuse):
            return None
        if int(self.current_task_id) != 1:
            return None
        src = int(self.anchor_first_reuse_source_task)
        if src not in self.task_to_copy:
            return None
        copy_id = int(self.task_to_copy[src])
        self.residual_keep_ratio = float(self.anchor_first_reuse_rho)
        out_details: Dict[str, Optional[float]] = dict(details)
        out_details["dynamic_score"] = out_details.get("best_score", None)
        out_details["dynamic_occ"] = float(self._copy_occupied_ratio(copy_id))
        out_details["dynamic_projected_occ"] = float(
            self._dynamic_projected_occupancy(out_details["dynamic_occ"], self.residual_keep_ratio)  # type: ignore[arg-type]
        )
        out_details["dynamic_rho"] = float(self.residual_keep_ratio)
        out_details["dynamic_objective"] = out_details.get("best_score", None)
        out_details["dynamic_reason"] = None  # type: ignore[assignment]
        out_details["dynamic_reason"] = "anchor_first_reuse"  # type: ignore[assignment]
        print(
            "[tsn-dynamic-routing-panda] ANCHOR "
            f"task={self.current_task_id} copy={copy_id} src={src} "
            f"rho={float(self.residual_keep_ratio):.3f}",
            flush=True,
        )
        return int(copy_id), int(src), out_details, False

    def _dynamic_route_from_scores(
        self,
        final_scores: Dict[int, float],
        best_task: int,
        details: Dict[str, Optional[float]],
    ) -> Optional[Tuple[int, Optional[int], Dict[str, Optional[float]], bool]]:
        """Select source/copy/rho using affinity and copy-local occupancy.

        The objective follows the Atari DynOcc diagnostic: affinity score plus
        pressure for projected copy occupancy and for already crowded copies.
        It may open a fresh copy only if the fixed copy budget permits it.
        """
        if not final_scores:
            return None

        prev_tasks = sorted(int(t) for t in final_scores.keys())
        if not prev_tasks:
            return None

        anchor_route = self._anchor_first_transfer_route(details)
        if anchor_route is not None:
            return anchor_route

        tau_reuse = float(self._threshold_for_current_mode())
        occ_target = float(self.dynamic_occ_target)
        occ_open = float(self.dynamic_occ_open)
        occ_hard = float(self.dynamic_occ_hard)
        lam_occ = float(self.dynamic_occ_lambda)
        lam_full = float(self.dynamic_full_lambda)
        margin = float(self.dynamic_copy_margin)

        candidates: List[Dict[str, float]] = []
        for src_task in prev_tasks:
            score = float(final_scores[int(src_task)])
            copy_id = int(self.task_to_copy.get(int(src_task), 0))
            occ = float(self._copy_occupied_ratio(copy_id))
            rho = float(self._dynamic_residual_keep_ratio(occ, score))
            projected = float(self._dynamic_projected_occupancy(occ, rho))
            capacity_pressure = float(max(0.0, projected - occ_target))
            full_pressure = float(max(0.0, occ - occ_hard))
            obj = float(score + lam_occ * capacity_pressure + lam_full * full_pressure)
            candidates.append({
                "src_task": float(src_task),
                "copy_id": float(copy_id),
                "score": float(score),
                "occ": float(occ),
                "rho": float(rho),
                "projected_occ": float(projected),
                "objective": float(obj),
            })
            print(
                "[tsn-dynamic-routing-panda] CAND "
                f"task={self.current_task_id} src={int(src_task)} copy={copy_id} "
                f"score={score:.6f} occ={occ:.4f} rho={rho:.3f} "
                f"proj={projected:.4f} obj={obj:.6f}",
                flush=True,
            )

        best_by_score = min(candidates, key=lambda x: x["score"])
        best_by_obj = min(candidates, key=lambda x: x["objective"])
        if float(best_by_obj["score"]) <= float(best_by_score["score"]) + margin:
            chosen = best_by_obj
            chosen_reason = "objective"
        else:
            chosen = best_by_score
            chosen_reason = "score"

        chosen_src = int(chosen["src_task"])
        chosen_copy = int(chosen["copy_id"])
        chosen_score = float(chosen["score"])
        chosen_occ = float(chosen["occ"])
        chosen_rho = float(chosen["rho"])
        chosen_proj = float(chosen["projected_occ"])
        best_score = float(best_by_score["score"])

        budget = self._capacity_budget()
        can_create = (self.max_model_copies is None) or (len(self.copy_states) < int(budget))
        create_due_to_score = bool(chosen_score > tau_reuse and can_create)
        create_due_to_occ = bool(
            can_create
            and int(self.current_task_id) >= int(self.dynamic_delay_new_until_task)
            and chosen_occ >= occ_open
        )
        create_new = bool(create_due_to_score or create_due_to_occ)

        out_details: Dict[str, Optional[float]] = dict(details)
        out_details["threshold"] = float(tau_reuse)
        out_details["dynamic_score"] = float(chosen_score)
        out_details["dynamic_best_score"] = float(best_score)
        out_details["dynamic_occ"] = float(chosen_occ)
        out_details["dynamic_projected_occ"] = float(chosen_proj)
        out_details["dynamic_rho"] = float(chosen_rho)
        out_details["dynamic_objective"] = float(chosen["objective"])
        out_details["dynamic_reason"] = None  # type: ignore[assignment]
        out_details["dynamic_reason"] = chosen_reason  # type: ignore[assignment]

        self.residual_keep_ratio = float(chosen_rho)

        if create_new:
            new_copy = self._make_fresh_copy()
            self.copy_states.append(new_copy)
            copy_id = len(self.copy_states) - 1
            reason = "score" if create_due_to_score else "occupancy"
            print(
                "[tsn-dynamic-routing-panda] CHOSEN "
                f"task={self.current_task_id} action=new reason={reason} "
                f"copy={copy_id} src={chosen_src} score={chosen_score:.6f} "
                f"tau={tau_reuse:.6f} occ={chosen_occ:.4f} rho={chosen_rho:.3f} "
                f"proj={chosen_proj:.4f} obj={float(chosen['objective']):.6f}",
                flush=True,
            )
            return int(copy_id), int(chosen_src), out_details, True

        print(
            "[tsn-dynamic-routing-panda] CHOSEN "
            f"task={self.current_task_id} action=reuse reason={chosen_reason} "
            f"copy={chosen_copy} src={chosen_src} score={chosen_score:.6f} "
            f"tau={tau_reuse:.6f} occ={chosen_occ:.4f} rho={chosen_rho:.3f} "
            f"proj={chosen_proj:.4f} obj={float(chosen['objective']):.6f}",
            flush=True,
        )
        return int(chosen_copy), int(chosen_src), out_details, False

    def _clear_residual_base_masks(self) -> None:
        for name, mod in _iter_tsn_modules(self.model):
            if hasattr(mod, "clear_residual_base_masks"):
                mod.clear_residual_base_masks()
            if name in self._residual_old_allow_weight_reuse:
                try:
                    mod.allow_weight_reuse = bool(self._residual_old_allow_weight_reuse[name])
                except Exception:
                    pass
        self._residual_old_allow_weight_reuse.clear()

    def _source_masks_for_task(self, source_task: int) -> Optional[Dict[str, Optional[torch.Tensor]]]:
        source_copy_id = self.task_to_copy.get(int(source_task), None)
        if source_copy_id is None:
            return None
        if int(source_copy_id) != int(self.current_copy_id):
            return None
        src_state = self.copy_states[int(source_copy_id)]
        return src_state.per_task_masks.get(int(source_task), None)

    def _set_residual_base_from_source_mask(self, source_task: Optional[int]) -> None:
        if (not self.residual_reuse) or source_task is None:
            return
        if float(self.residual_keep_ratio) <= 0.0:
            return
        src_masks = self._source_masks_for_task(int(source_task))
        if src_masks is None:
            return
        with torch.no_grad():
            for name, mod in _iter_tsn_modules(self.model):
                w_key = f"{name}.weight"
                b_key = f"{name}.bias"
                src_w = src_masks.get(w_key, None)
                if src_w is not None:
                    mod.residual_base_weight_mask = src_w.detach().clone().to(device=mod.weight.device, dtype=torch.bool)
                    mod.residual_keep_ratio = float(self.residual_keep_ratio)
                    if self.residual_delta_free_only and hasattr(mod, "allow_weight_reuse"):
                        self._residual_old_allow_weight_reuse.setdefault(name, bool(mod.allow_weight_reuse))
                        # In residual mode, the base source mask provides frozen reuse.
                        # The delta should use free capacity, not arbitrary occupied weights.
                        mod.allow_weight_reuse = False
                if getattr(mod, "bias", None) is not None:
                    src_b = src_masks.get(b_key, None)
                    if src_b is not None:
                        mod.residual_base_bias_mask = src_b.detach().clone().to(device=mod.bias.device, dtype=torch.bool)
                        mod.residual_keep_ratio = float(self.residual_keep_ratio)
        print(f"[tsn-capacity] residual reuse enabled: task={self.current_task_id} source={source_task} delta_keep={self.residual_keep_ratio:g}")

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


    # ------------------------------------------------------------------
    # Probe-guided adaptive routing helpers
    # ------------------------------------------------------------------
    def _parse_probe_residual_grid(self) -> List[float]:
        vals: List[float] = []
        for x in str(self.probe_residual_grid).split(","):
            x = x.strip()
            if not x:
                continue
            try:
                vals.append(float(x))
            except ValueError:
                raise ValueError(f"Bad --tsn-probe-residual-grid entry: {x!r}")
        if not vals:
            vals = [0.0]
        # stable unique, clipped to [0, 1]
        out: List[float] = []
        seen = set()
        for v in vals:
            vv = float(max(0.0, min(1.0, v)))
            key = round(vv, 6)
            if key not in seen:
                out.append(vv)
                seen.add(key)
        return out

    def _rank_source_tasks_for_probe(
        self,
        prev_tasks: List[int],
        action_scores: Dict[int, float],
        latent_scores: Dict[int, float],
    ) -> Tuple[List[int], Dict[int, float]]:
        if not prev_tasks:
            return [], {}
        if self.reuse_score_mode == "action":
            scores = dict(action_scores)
        elif self.reuse_score_mode == "latent":
            scores = dict(latent_scores)
        elif self.reuse_score_mode == "hybrid":
            if len(prev_tasks) == 1:
                t0 = int(prev_tasks[0])
                scores = {
                    t0: self._single_source_hybrid_score(
                        float(action_scores[t0]),
                        float(latent_scores[t0]),
                    )
                }
            else:
                act_n = self._normalize_score_dict(action_scores) if self.normalize_similarity_scores else action_scores
                lat_n = self._normalize_score_dict(latent_scores) if self.normalize_similarity_scores else latent_scores
                scores = {
                    int(t): float(self.hybrid_alpha * act_n[int(t)] + (1.0 - self.hybrid_alpha) * lat_n[int(t)])
                    for t in prev_tasks
                }
        else:
            raise ValueError(f"Unsupported reuse_score_mode: {self.reuse_score_mode}")
        ranked = sorted([int(t) for t in prev_tasks], key=lambda t: float(scores.get(int(t), float("inf"))))
        return ranked, scores

    def _snapshot_copy_for_probe(self, copy_id: int) -> Dict[str, Any]:
        st = self.copy_states[int(copy_id)]
        return {
            "model_state": {k: v.detach().clone() for k, v in st.model.state_dict().items()},
            "opt_state": copy.deepcopy(st.opt.state_dict()),
            "task_keep_ratios": dict(self.task_keep_ratios),
            "st_task_keep_ratios": dict(getattr(st, "task_keep_ratios", {})),
            "current_keep_ratio": float(self.current_keep_ratio),
            "active_eval_task": None if self.active_eval_task is None else int(self.active_eval_task),
        }

    def _restore_copy_after_probe(self, copy_id: int, snap: Dict[str, Any]) -> None:
        st = self.copy_states[int(copy_id)]
        st.model.load_state_dict(snap["model_state"])
        try:
            st.opt.load_state_dict(snap["opt_state"])
        except Exception:
            # Optimizer structure may have been rebuilt during probing. In that case
            # restoring the model is the essential part; the real full training will
            # rebuild the optimizer again in _prepare_current_task().
            pass
        st.task_keep_ratios = dict(snap.get("st_task_keep_ratios", getattr(st, "task_keep_ratios", {})))
        if int(getattr(self, "current_copy_id", -1)) == int(copy_id):
            self.model = st.model
            self.opt = st.opt
        self.task_keep_ratios = dict(snap.get("task_keep_ratios", self.task_keep_ratios))
        self.current_keep_ratio = float(snap.get("current_keep_ratio", self.current_keep_ratio))
        self.active_eval_task = snap.get("active_eval_task", self.active_eval_task)

    def _split_probe_train_val(self, task_trajs_pad: List[Trajectory]) -> Tuple[List[Trajectory], List[Trajectory]]:
        """Deterministic task-level split used only for probe route selection."""
        n = len(task_trajs_pad)
        if n < 10:
            return task_trajs_pad, task_trajs_pad
        frac = float(max(0.0, min(0.5, self.probe_val_fraction)))
        if frac <= 0.0:
            return task_trajs_pad, task_trajs_pad
        rng = np.random.default_rng(int(self.probe_seed) + 1009 * int(self.current_task_id))
        idx = np.arange(n, dtype=np.int64)
        rng.shuffle(idx)
        n_val = max(1, int(round(frac * n)))
        val_set = set(int(i) for i in idx[:n_val].tolist())
        train = [tr for i, tr in enumerate(task_trajs_pad) if i not in val_set]
        val = [tr for i, tr in enumerate(task_trajs_pad) if i in val_set]
        if not train or not val:
            return task_trajs_pad, task_trajs_pad
        return train, val

    def _snapshot_rng_for_probe(self) -> Dict[str, Any]:
        snap: Dict[str, Any] = {
            "np_state": np.random.get_state(),
            "torch_state": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            try:
                snap["cuda_state"] = torch.cuda.get_rng_state_all()
            except Exception:
                snap["cuda_state"] = None
        return snap

    def _restore_rng_after_probe(self, snap: Dict[str, Any]) -> None:
        try:
            np.random.set_state(snap["np_state"])
        except Exception:
            pass
        try:
            torch.random.set_rng_state(snap["torch_state"])
        except Exception:
            pass
        cuda_state = snap.get("cuda_state", None)
        if cuda_state is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(cuda_state)
            except Exception:
                pass

    def _set_probe_rng(self) -> None:
        seed = int(self.probe_seed) + 7919 * int(self.current_task_id)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _probe_validation_loss(
        self,
        task_trajs_pad: List[Trajectory],
        action_mask_t: torch.Tensor,
    ) -> float:
        loader = make_panda_loader(
            task_trajs_pad,
            seq_len=self.seq_len,
            batch_size=max(1, int(self.probe_batch_size)),
            device=self.device,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
        )
        vals: List[float] = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(max(1, int(self.probe_val_batches))):
                obs, actions, rtg, ts, mask = unpack_batch_continuous(next(loader))
                pred = self.model(obs, actions, rtg, ts, attention_mask=mask)
                loss = _masked_mse_with_action_mask(pred, actions, mask, action_mask_t)
                vals.append(float(loss.detach().cpu().item()))
        return float(np.mean(vals)) if vals else float("inf")

    def _probe_train_candidate(
        self,
        probe_train_trajs_pad: List[Trajectory],
        probe_val_trajs_pad: List[Trajectory],
        action_mask_t: torch.Tensor,
        *,
        source_task: Optional[int],
        residual_ratio: float,
        created_new_copy: bool,
    ) -> float:
        self._prepare_current_task()
        self._clear_residual_base_masks()

        old_residual_keep = float(self.residual_keep_ratio)
        old_residual_reuse = bool(self.residual_reuse)
        try:
            self.residual_keep_ratio = float(residual_ratio)
            # residual_ratio == 0 means score warm-start only / no forced source mask.
            self.residual_reuse = bool(old_residual_reuse and residual_ratio > 0.0 and (not created_new_copy))
            if source_task is not None and self.residual_reuse:
                self._set_residual_base_from_source_mask(int(source_task))
            if source_task is not None:
                if (not created_new_copy) or self.warmstart_on_new_copy:
                    self._warmstart_scores_from_source_mask(int(source_task))

            if int(self.probe_steps) > 0:
                loader = make_panda_loader(
                    probe_train_trajs_pad,
                    seq_len=self.seq_len,
                    batch_size=max(1, int(self.probe_batch_size)),
                    device=self.device,
                    obs_dim=self.obs_dim,
                    act_dim=self.act_dim,
                )
                self.model.train()
                for _ in range(int(self.probe_steps)):
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
            return self._probe_validation_loss(probe_val_trajs_pad, action_mask_t)
        finally:
            self._clear_residual_base_masks()
            self.residual_keep_ratio = old_residual_keep
            self.residual_reuse = old_residual_reuse

    def _probe_select_route(
        self,
        prev_tasks: List[int],
        action_scores: Dict[int, float],
        latent_scores: Dict[int, float],
        final_scores: Dict[int, float],
        task_trajs_pad: List[Trajectory],
        action_mask_t: torch.Tensor,
    ) -> Tuple[int, Optional[int], Dict[str, Optional[float]], bool]:
        ranked_sources = sorted([int(t) for t in prev_tasks], key=lambda t: float(final_scores.get(int(t), float("inf"))))
        ranked_sources = ranked_sources[: max(1, int(self.probe_top_k))]
        if not ranked_sources:
            return 0, None, {"best_action": None, "best_latent": None, "best_score": None}, False

        residual_grid = self._parse_probe_residual_grid()
        candidates: List[Dict[str, Any]] = []
        for src in ranked_sources:
            copy_id = int(self.task_to_copy[int(src)])
            for rr in residual_grid:
                candidates.append({
                    "kind": "reuse",
                    "copy_id": copy_id,
                    "source_task": int(src),
                    "residual_ratio": float(rr),
                    "created_new_copy": False,
                })

        can_make_new = self.max_model_copies is None or len(self.copy_states) < int(self.max_model_copies)
        if self.probe_include_new_copy and can_make_new:
            # Use the best source only as an optional score warm-start source for the fresh copy.
            candidates.append({
                "kind": "new_copy",
                "copy_id": None,
                "source_task": int(ranked_sources[0]),
                "residual_ratio": 0.0,
                "created_new_copy": True,
            })

        probe_train_trajs, probe_val_trajs = self._split_probe_train_val(task_trajs_pad)
        print(
            f"[tsn-probe-routing] task={self.current_task_id} split="
            f"train{len(probe_train_trajs)}/val{len(probe_val_trajs)} "
            f"det={int(self.probe_deterministic)}"
        )

        original_copy_id = int(self.current_copy_id)
        original_n_copies = len(self.copy_states)
        candidate_results: List[Dict[str, Any]] = []

        for cand in candidates:
            rng_snap = self._snapshot_rng_for_probe() if self.probe_deterministic else None
            if self.probe_deterministic:
                self._set_probe_rng()

            temp_copy_id: Optional[int] = None
            cid: Optional[int] = None
            snap: Optional[Dict[str, Any]] = None
            try:
                if cand["created_new_copy"]:
                    temp_state = self._make_fresh_copy()
                    self.copy_states.append(temp_state)
                    temp_copy_id = len(self.copy_states) - 1
                    cid = temp_copy_id
                else:
                    cid = int(cand["copy_id"])

                snap = self._snapshot_copy_for_probe(cid)
                self._activate_copy(cid)
                val_loss = self._probe_train_candidate(
                    probe_train_trajs,
                    probe_val_trajs,
                    action_mask_t,
                    source_task=cand["source_task"],
                    residual_ratio=float(cand["residual_ratio"]),
                    created_new_copy=bool(cand["created_new_copy"]),
                )
                occ = 0.0 if cand["created_new_copy"] else self._copy_occupied_ratio(cid)
                objective = (
                    float(val_loss)
                    + float(self.copy_penalty) * float(bool(cand["created_new_copy"]))
                    + float(self.occupancy_penalty) * float(occ)
                    + float(self.probe_residual_penalty) * float(cand["residual_ratio"])
                )
                candidate_results.append({
                    **cand,
                    "candidate_index": int(len(candidate_results)),
                    "probe_val_loss": float(val_loss),
                    "probe_objective": float(objective),
                    "occupied_ratio": float(occ),
                })
            finally:
                if cid is not None and snap is not None:
                    self._restore_copy_after_probe(int(cid), snap)
                self._clear_residual_base_masks()
                if temp_copy_id is not None:
                    # Remove the temporary probe copy after restoring global refs below.
                    pass
                # Restore active copy before possibly popping temp copy, so _activate_copy
                # does not try to sync a soon-to-be-deleted public model.
                if original_copy_id < len(self.copy_states):
                    self._activate_copy(original_copy_id)
                if temp_copy_id is not None:
                    self.copy_states.pop()
                    if self.current_copy_id >= len(self.copy_states):
                        self.current_copy_id = min(original_copy_id, len(self.copy_states) - 1)
                if rng_snap is not None:
                    self._restore_rng_after_probe(rng_snap)

            # Defensive cleanup if an exception skipped the pop above.
            while len(self.copy_states) > original_n_copies:
                self.copy_states.pop()

        if original_copy_id < len(self.copy_states):
            self._activate_copy(original_copy_id)

        best = min(candidate_results, key=lambda x: float(x["probe_objective"]))
        src = int(best["source_task"]) if best.get("source_task") is not None else None
        best_score = float(final_scores.get(src, float("nan"))) if src is not None else None
        details: Dict[str, Any] = {
            "best_action": None if src is None else float(action_scores.get(src, float("nan"))),
            "best_latent": None if src is None or src not in latent_scores else float(latent_scores.get(src, float("nan"))),
            "best_score": best_score,
            "probe_routing": True,
            "probe_plan": str(best["kind"]),
            "probe_selected_residual_keep_ratio": float(best["residual_ratio"]),
            "probe_val_loss": float(best["probe_val_loss"]),
            "probe_objective": float(best["probe_objective"]),
            "probe_train_size": int(len(probe_train_trajs)),
            "probe_val_size": int(len(probe_val_trajs)),
            "probe_deterministic": bool(self.probe_deterministic),
            "probe_val_fraction": float(self.probe_val_fraction),
            "residual_delta_free_only": bool(self.residual_delta_free_only),
            "probe_candidates": [
                {
                    "candidate_index": int(r.get("candidate_index", -1)),
                    "kind": str(r["kind"]),
                    "copy_id": None if r.get("copy_id") is None else int(r["copy_id"]),
                    "source_task": None if r.get("source_task") is None else int(r["source_task"]),
                    "residual_ratio": float(r["residual_ratio"]),
                    "probe_val_loss": float(r["probe_val_loss"]),
                    "probe_objective": float(r["probe_objective"]),
                    "occupied_ratio": float(r["occupied_ratio"]),
                }
                for r in candidate_results
            ],
        }

        for r in sorted(candidate_results, key=lambda x: float(x["probe_objective"]))[: min(6, len(candidate_results))]:
            print(
                f"[tsn-probe-candidate] task={self.current_task_id} kind={r['kind']} "
                f"src={r.get('source_task')} resid={float(r['residual_ratio']):.3f} "
                f"new={int(bool(r['created_new_copy']))} val={float(r['probe_val_loss']):.6e} "
                f"obj={float(r['probe_objective']):.6e} occ={float(r['occupied_ratio']):.4f}"
            )

        print(
            f"[tsn-probe-routing] task={self.current_task_id} selected={details['probe_plan']} "
            f"src={src} resid={details['probe_selected_residual_keep_ratio']:.3f} "
            f"val={details['probe_val_loss']:.6e} obj={details['probe_objective']:.6e} "
            f"candidates={len(candidate_results)}"
        )

        if bool(best["created_new_copy"]):
            if self.max_model_copies is not None and len(self.copy_states) >= int(self.max_model_copies):
                # Should not happen because we filter can_make_new, but keep safe fallback.
                copy_id = int(self.task_to_copy[int(src)]) if src is not None else 0
                return copy_id, src, details, False
            new_copy = self._make_fresh_copy()
            self.copy_states.append(new_copy)
            return len(self.copy_states) - 1, src, details, True

        copy_id = int(best["copy_id"])
        return copy_id, src, details, False

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

        prev_tasks = sorted(int(t) for t in self.task_to_copy.keys())
        if self.routing_policy in ("random", "round_robin", "always_new"):
            return self._select_copy_by_policy([int(t) for t in prev_tasks])

        for t in prev_tasks:
            action_scores[int(t)] = self._estimate_action_compatibility(int(t), task_trajs_pad, action_mask_t)
            if self.reuse_score_mode in ("latent", "hybrid"):
                latent_scores[int(t)] = self._estimate_latent_similarity(int(t), task_memory_obs)

        ranked_tasks, final_scores = self._rank_source_tasks_for_probe(prev_tasks, action_scores, latent_scores)
        best_task = int(ranked_tasks[0])
        best_score = float(final_scores[best_task])

        if self.reuse_score_mode == "action":
            threshold = self.action_reuse_threshold
            details = {
                "best_action": float(action_scores[best_task]),
                "best_latent": None,
                "best_score": float(best_score),
            }
        elif self.reuse_score_mode == "latent":
            threshold = self.latent_reuse_threshold
            details = {
                "best_action": float(action_scores.get(best_task, float("nan"))),
                "best_latent": float(latent_scores[best_task]),
                "best_score": float(best_score),
            }
        elif self.reuse_score_mode == "hybrid":
            threshold = 1.0 if len(prev_tasks) == 1 else self.hybrid_reuse_threshold
            details = {
                "best_action": float(action_scores[best_task]),
                "best_latent": float(latent_scores[best_task]),
                "best_score": float(best_score),
            }
        else:
            raise ValueError(f"Unsupported reuse_score_mode: {self.reuse_score_mode}")

        if self.routing_policy == "dynamic_occ":
            dynamic_route = self._dynamic_route_from_scores(final_scores, int(best_task), details)
            if dynamic_route is not None:
                return dynamic_route

        if self.probe_routing and int(self.probe_steps) >= 0:
            return self._probe_select_route(
                prev_tasks,
                action_scores,
                latent_scores,
                final_scores,
                task_trajs_pad,
                action_mask_t,
            )

        create_new = self._should_create_new(int(best_task), float(best_score), float(threshold))
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
        selected_residual_keep_ratio = score_details.get("probe_selected_residual_keep_ratio", None)
        if selected_residual_keep_ratio is None:
            selected_residual_keep_ratio = float(self.residual_keep_ratio)

        self.task_similarity[self.current_task_id] = {
            "source_task": None if src_task is None else int(src_task),
            "copy_id": int(copy_id),
            "best_action": score_details.get("best_action", None),
            "best_latent": score_details.get("best_latent", None),
            "best_score": score_details.get("best_score", None),
            "score_mode": self.reuse_score_mode,
            "created_new_copy": bool(created_new),
            "routing_policy": self.routing_policy,
            "copy_penalty": float(self.copy_penalty),
            "occupancy_penalty": float(self.occupancy_penalty),
            "residual_reuse": bool(self.residual_reuse),
            "residual_keep_ratio": float(self.residual_keep_ratio),
            "selected_residual_keep_ratio": float(selected_residual_keep_ratio),
            "dynamic_occ": score_details.get("dynamic_occ", None),
            "dynamic_projected_occ": score_details.get("dynamic_projected_occ", None),
            "dynamic_rho": score_details.get("dynamic_rho", None),
            "dynamic_score": score_details.get("dynamic_score", None),
            "dynamic_objective": score_details.get("dynamic_objective", None),
            "dynamic_reason": score_details.get("dynamic_reason", None),
            "probe_routing": bool(score_details.get("probe_routing", False)),
            "probe_plan": score_details.get("probe_plan", None),
            "probe_val_loss": score_details.get("probe_val_loss", None),
            "probe_objective": score_details.get("probe_objective", None),
            "probe_train_size": score_details.get("probe_train_size", None),
            "probe_val_size": score_details.get("probe_val_size", None),
            "probe_val_fraction": score_details.get("probe_val_fraction", None),
            "probe_deterministic": score_details.get("probe_deterministic", None),
            "residual_delta_free_only": score_details.get("residual_delta_free_only", None),
            "probe_candidates": score_details.get("probe_candidates", None),
        }

        if self.store_task_obs_stats and self._last_task_obs_stats is not None:
            mean_np, std_np = self._last_task_obs_stats
            if hasattr(self.model, "obs_mean") and hasattr(self.model, "obs_std"):
                with torch.no_grad():
                    self.model.obs_mean.copy_(torch.as_tensor(mean_np, dtype=self.model.obs_mean.dtype, device=self.model.obs_mean.device))
                    self.model.obs_std.copy_(torch.as_tensor(std_np, dtype=self.model.obs_std.dtype, device=self.model.obs_std.device))

        self._prepare_current_task()
        self._clear_residual_base_masks()
        old_residual_keep_ratio = float(self.residual_keep_ratio)
        self.residual_keep_ratio = float(selected_residual_keep_ratio)
        if src_task is not None and (not created_new) and self.residual_reuse and float(self.residual_keep_ratio) > 0.0:
            self._set_residual_base_from_source_mask(src_task)
        if src_task is not None:
            if (not created_new) or self.warmstart_on_new_copy:
                self._warmstart_scores_from_source_mask(src_task)
        self.residual_keep_ratio = old_residual_keep_ratio

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
                    f"probe={int(bool(meta.get('probe_routing', False)))} plan={meta.get('probe_plan', None)} "
                    f"resid={meta.get('selected_residual_keep_ratio', meta.get('residual_keep_ratio', None))} "
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