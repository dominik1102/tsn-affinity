from __future__ import annotations

import copy
import re

try:
    from .reuse_accounting import compute_mask_reuse_stats, print_mask_reuse_stats
except ImportError:  # compatibility with older reuse_accounting.py
    from .reuse_accounting import compute_mask_reuse_stats, short_reuse_line

    def print_mask_reuse_stats(prefix: str, stats: dict) -> None:
        task_id = stats.get("task_id", "?")
        print(short_reuse_line(prefix, task_id, stats), flush=True)

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from dt.dataset import Trajectory, make_minibatches

from .tsn_common import _iter_tsn_modules
from .tsn_original_reuse_atari import TSNOriginalReuseStrategy
from .utils import _unpack_batch


class TSNImprovedReuseAtariStrategy(TSNOriginalReuseStrategy):
    """
    RL-oriented improved reuse for Atari DT.

    Compared to TSNOriginalReuseStrategy (origin reuse), this version adds:
      1) routing by ACTION COMPATIBILITY on offline expert demonstrations,
      2) optional routing by LATENT similarity (symmetric KL between diagonal
         Gaussians fitted to encoded observation latents),
      3) optional HYBRID routing = alpha * normalized(action_score)
                                  + (1-alpha) * normalized(latent_score),
      4) warm-start of mask scores from the selected source-task mask.

    Important:
      - task 0 does NOT call parent train_task/after_task;
      - task 0 is handled in the same improved pipeline,
        only with source_task=None and copy_id=0.
    """

    def __init__(
        self,
        *args,
        reuse_score_mode: str = "action",   # action | latent | hybrid
        routing_n_batches: int = 4,
        routing_batch_size: int = 64,
        action_reuse_threshold: float = 12.0,
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
        probe_routing: bool = False,
        probe_steps: int = 0,
        probe_top_k: int = 2,
        probe_batch_size: int = 64,
        probe_val_batches: int = 4,
        probe_residual_grid: str = "0.10,0.25,0.50",
        probe_new_copy_penalty: float = 0.02,
        probe_residual_penalty: float = 0.0,
        probe_occupancy_penalty: float = 0.0,
        probe_include_new_copy: bool = True,
        probe_seed: int = 123000,
        probe_deterministic: bool = True,
        manual_route_plan: str = "",
        manual_action_thresholds: str = "",
        dynamic_occ_target: float = 0.80,
        dynamic_occ_open: float = 0.84,
        dynamic_occ_hard: float = 0.90,
        dynamic_occ_lambda: float = 30.0,
        dynamic_full_lambda: float = 20.0,
        dynamic_copy_margin: float = 6.0,
        dynamic_delay_new_until_task: int = 2,
        dynamic_rho_high: float = 0.50,
        dynamic_rho_mid: float = 0.25,
        dynamic_rho_low: float = 0.10,
        dynamic_low_occ: float = 0.60,
        dynamic_mid_occ: float = 0.80,
        anchor_first_reuse: bool = False,
        anchor_first_reuse_rho: float = 0.50,
        anchor_first_reuse_source_task: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.reuse_score_mode = str(reuse_score_mode)
        self.routing_n_batches = int(routing_n_batches)
        self.routing_batch_size = int(routing_batch_size)

        # lower is better
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

        # Probe-Adaptive Routing (PAR). When enabled, routing is selected by
        # a short offline inner-loop probe instead of a fixed hard threshold.
        self.probe_routing = bool(probe_routing)
        self.probe_steps = int(probe_steps)
        self.probe_top_k = int(probe_top_k)
        self.probe_batch_size = int(probe_batch_size)
        self.probe_val_batches = int(probe_val_batches)
        self.probe_residual_grid = str(probe_residual_grid)
        self.probe_new_copy_penalty = float(probe_new_copy_penalty)
        self.probe_residual_penalty = float(probe_residual_penalty)
        self.probe_occupancy_penalty = float(probe_occupancy_penalty)
        self.probe_include_new_copy = bool(probe_include_new_copy)
        self.probe_seed = int(probe_seed)
        self.probe_deterministic = bool(probe_deterministic)

        # per-task latent stats under task's own copy/mask
        self.task_latent_stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.task_weight_reuse_stats: Dict[int, Dict[str, object]] = {}

        # Manual/oracle routing debugger.  This is intentionally explicit and
        # deterministic: it lets us test route plans such as
        #   "1:reuse:0:rho=0.50,2:reuse:0:rho=0.50,3:new:1,4:reuse:3:rho=0.50".
        # It is useful both as an oracle upper bound and as a debugger for future
        # dynamic routing rules.  If no plan is provided, normal routing is used.
        self.manual_route_plan_raw = str(manual_route_plan or "")
        self.manual_route_entries: Dict[int, Dict[str, Any]] = self._parse_manual_route_plan(
            self.manual_route_plan_raw
        )
        self.manual_action_thresholds: Dict[int, float] = self._parse_task_thresholds(
            str(manual_action_thresholds or "")
        )
        for _task_id, _entry in self.manual_route_entries.items():
            if _entry.get("threshold") is not None and _task_id not in self.manual_action_thresholds:
                self.manual_action_thresholds[int(_task_id)] = float(_entry["threshold"])

        # Dynamic occupancy-aware routing (no probe).  This route selector uses
        # action/latent affinity, current copy occupancy and projected occupancy
        # to choose source/copy and a task-specific residual keep ratio.
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

        # Anchor the first transfer step (task_id=1).  This is not a manual
        # oracle over the whole benchmark; it encodes a stable Atari-specific
        # bootstrap rule observed in the logs: the first incoming task should
        # reuse the task-0 copy and use a high residual budget.
        self.anchor_first_reuse = bool(anchor_first_reuse)
        self.anchor_first_reuse_rho = float(anchor_first_reuse_rho)
        self.anchor_first_reuse_source_task = int(anchor_first_reuse_source_task)

    def _restore_active_copy_without_sync(self, copy_id: int) -> None:
        """
        Restore public self.model/self.opt from copy_states[copy_id] WITHOUT first
        syncing the current public model back into the copy bank.

        This is required inside probe routing. The normal _activate_copy() calls
        _sync_public_state_to_active_copy() before switching, which can overwrite
        a restored clean copy with a temporary probe-trained model.
        """
        self.current_copy_id = int(copy_id)
        st = self.copy_states[self.current_copy_id]
        self.model = st.model
        self.opt = st.opt
        self._refresh_name_sets()

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

    # ------------------------------------------------------------------
    # Manual/oracle routing debugger
    # ------------------------------------------------------------------
    @staticmethod
    def _split_plan_entries(text: str) -> List[str]:
        raw = str(text or "").strip()
        if not raw:
            return []
        return [x.strip() for x in re.split(r"[;,]", raw) if x.strip()]

    @staticmethod
    def _safe_int(x: object, default: Optional[int] = None) -> Optional[int]:
        try:
            return int(str(x).strip())
        except Exception:
            return default

    @staticmethod
    def _safe_float(x: object, default: Optional[float] = None) -> Optional[float]:
        try:
            return float(str(x).strip())
        except Exception:
            return default

    def _parse_task_thresholds(self, text: str) -> Dict[int, float]:
        """
        Parse per-task threshold overrides.

        Examples:
            "1:25,4:10"
            "1=25;4=10"

        Task ids are zero-based continual ids, so in the Atari-5 sequence:
            1 -> Alien, 2 -> Atlantis, 3 -> Boxing, 4 -> Centipede.
        """
        out: Dict[int, float] = {}
        for entry in self._split_plan_entries(text):
            if "=" in entry and ":" not in entry.split("=", 1)[0]:
                entry = entry.replace("=", ":", 1)
            parts = [part.strip() for part in entry.split(":") if part.strip()]
            if len(parts) < 2:
                continue
            task_id = self._safe_int(parts[0], None)
            value = self._safe_float(parts[1], None)
            if task_id is None or value is None:
                continue
            out[int(task_id)] = float(value)
        return out

    def _parse_manual_route_plan(self, text: str) -> Dict[int, Dict[str, Any]]:
        """
        Parse a manual/oracle route plan.

        Supported forms, separated by ',' or ';':
            "1:reuse:0"
            "1:reuse:0:rho=0.50"
            "2:new:0"
            "3:copy:1:src=2:rho=0.25"
            "4:threshold:25"  # only overrides threshold; normal routing continues

        Meaning:
            reuse:S     -> use the copy that currently stores source task S.
            new:S       -> create a fresh copy, with S kept as source metadata.
            copy:C      -> use existing copy C; source can be set with src=S.
            threshold:X -> use normal affinity routing with threshold X for this task.

        Task ids are zero-based continual task ids.
        """
        plan: Dict[int, Dict[str, Any]] = {}
        for raw in self._split_plan_entries(text):
            entry = raw
            if "=" in entry and entry.split("=", 1)[0].strip().isdigit():
                entry = entry.replace("=", ":", 1)
            parts = [part.strip() for part in entry.split(":") if part.strip()]
            if len(parts) < 2:
                continue

            task_id = self._safe_int(parts[0], None)
            if task_id is None:
                continue

            action = parts[1].lower().replace("-", "_")
            cfg: Dict[str, Any] = {
                "action": action,
                "source_task": None,
                "copy_id": None,
                "rho": None,
                "threshold": None,
                "raw": raw,
            }

            positional: List[str] = []
            for tok in parts[2:]:
                if "=" in tok:
                    key, value = tok.split("=", 1)
                    key = key.strip().lower().replace("-", "_")
                    value = value.strip()
                    if key in ("src", "source", "source_task", "task"):
                        cfg["source_task"] = self._safe_int(value, None)
                    elif key in ("copy", "copy_id"):
                        cfg["copy_id"] = self._safe_int(value, None)
                    elif key in ("rho", "residual", "residual_keep_ratio", "delta", "delta_keep"):
                        cfg["rho"] = self._safe_float(value, None)
                    elif key in ("thr", "threshold", "tau"):
                        cfg["threshold"] = self._safe_float(value, None)
                else:
                    positional.append(tok)

            if action in ("reuse", "source", "src", "always_reuse"):
                if positional and cfg["source_task"] is None:
                    cfg["source_task"] = self._safe_int(positional[0], None)
            elif action in ("new", "spawn", "always_new"):
                if positional and cfg["source_task"] is None:
                    cfg["source_task"] = self._safe_int(positional[0], None)
            elif action in ("copy", "use_copy", "copy_id"):
                if positional and cfg["copy_id"] is None:
                    cfg["copy_id"] = self._safe_int(positional[0], None)
                if len(positional) >= 2 and cfg["source_task"] is None:
                    cfg["source_task"] = self._safe_int(positional[1], None)
            elif action in ("threshold", "thr", "tau", "auto"):
                if positional and cfg["threshold"] is None:
                    cfg["threshold"] = self._safe_float(positional[0], None)
            else:
                print(f"[tsn-manual-routing-atari] ignoring unsupported entry: {raw!r}", flush=True)
                continue

            if cfg["rho"] is not None:
                cfg["rho"] = float(max(0.0, min(1.0, float(cfg["rho"]))))
            plan[int(task_id)] = cfg
        return plan

    def _task_threshold_override(self, default_threshold: float) -> float:
        task_id = int(self.current_task_id)
        if task_id in self.manual_action_thresholds:
            return float(self.manual_action_thresholds[task_id])
        entry = self.manual_route_entries.get(task_id, None)
        if entry is not None and entry.get("threshold") is not None:
            return float(entry["threshold"])
        return float(default_threshold)

    def _manual_select_route(
        self,
        final_scores: Dict[int, float],
        best_task: int,
        details: Dict[str, Optional[float]],
    ) -> Optional[Tuple[int, Optional[int], Dict[str, Optional[float]], bool]]:
        """Return an explicitly requested route for the current task, if any."""
        task_id = int(self.current_task_id)
        entry = self.manual_route_entries.get(task_id, None)
        if entry is None:
            return None

        action = str(entry.get("action", "")).lower().replace("-", "_")
        if action in ("threshold", "thr", "tau", "auto"):
            return None

        out_details: Dict[str, Optional[float]] = dict(details)
        out_details["manual_route"] = str(entry.get("raw", ""))  # type: ignore[assignment]
        out_details["manual_action"] = action  # type: ignore[assignment]

        if entry.get("rho") is not None:
            self.residual_keep_ratio = float(entry["rho"])
            out_details["manual_residual_keep_ratio"] = float(entry["rho"])

        source_task = entry.get("source_task", None)
        if source_task is None:
            source_task = int(best_task)
        else:
            source_task = int(source_task)

        def _fallback_source_copy(reason: str) -> Tuple[int, Optional[int], Dict[str, Optional[float]], bool]:
            src = None if source_task is None else int(source_task)
            if src is not None and src in self.task_to_copy:
                cid = int(self.task_to_copy[src])
            elif self.copy_states:
                cid = min(int(self.current_copy_id), len(self.copy_states) - 1)
                src = self._source_task_for_copy(cid, sorted(self.task_to_copy.keys()))
            else:
                cid = 0
            out_details["manual_fallback"] = reason  # type: ignore[assignment]
            print(
                "[tsn-manual-routing-atari] "
                f"task={task_id} fallback={reason} copy={cid} src={src} "
                f"rho={self.residual_keep_ratio:g}",
                flush=True,
            )
            return int(cid), None if src is None else int(src), out_details, False

        created_new = False
        if action in ("reuse", "source", "src", "always_reuse"):
            if source_task not in self.task_to_copy:
                return _fallback_source_copy("missing_source_for_reuse")
            copy_id = int(self.task_to_copy[int(source_task)])

        elif action in ("copy", "use_copy", "copy_id"):
            copy_id = entry.get("copy_id", None)
            if copy_id is None:
                return _fallback_source_copy("missing_copy_id")
            copy_id = int(copy_id)
            if copy_id < 0 or copy_id >= len(self.copy_states):
                return _fallback_source_copy("copy_id_out_of_range")
            if source_task not in self.task_to_copy or int(self.task_to_copy.get(int(source_task), -1)) != int(copy_id):
                inferred = self._source_task_for_copy(int(copy_id), sorted(self.task_to_copy.keys()))
                source_task = inferred if inferred is not None else source_task

        elif action in ("new", "spawn", "always_new"):
            budget = self._capacity_budget()
            can_create = (self.max_model_copies is None) or (len(self.copy_states) < int(budget))
            if not can_create:
                return _fallback_source_copy("manual_new_capacity_full")
            new_copy = self._make_fresh_copy()
            self.copy_states.append(new_copy)
            copy_id = len(self.copy_states) - 1
            created_new = True

        else:
            return None

        print(
            "[tsn-manual-routing-atari] CHOSEN "
            f"task={task_id} action={action} copy={int(copy_id)} "
            f"src={None if source_task is None else int(source_task)} "
            f"rho={self.residual_keep_ratio:g} new_copy={int(created_new)} "
            f"raw={entry.get('raw', '')!r}",
            flush=True,
        )
        return int(copy_id), None if source_task is None else int(source_task), out_details, bool(created_new)

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

    def _clear_residual_base_masks(self) -> None:
        for _, mod in _iter_tsn_modules(self.model):
            if hasattr(mod, "clear_residual_base_masks"):
                mod.clear_residual_base_masks()

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
                if getattr(mod, "bias", None) is not None:
                    src_b = src_masks.get(b_key, None)
                    if src_b is not None:
                        mod.residual_base_bias_mask = src_b.detach().clone().to(device=mod.bias.device, dtype=torch.bool)
                        mod.residual_keep_ratio = float(self.residual_keep_ratio)
        print(f"[tsn-capacity] residual reuse enabled: task={self.current_task_id} source={source_task} delta_keep={self.residual_keep_ratio:g}", flush=True)

    # ------------------------------------------------------------------
    # Latent helpers
    # ------------------------------------------------------------------
    def _extract_obs_latents(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Extract observation latents from DecisionTransformer.obs_enc.

        obs: [B,L,C,H,W] or [B,L,D]
        returns: [B,L,d_model]
        """
        x = obs.to(self.device)
        orig_dtype = x.dtype
        x = x.to(dtype=torch.float32)

        if orig_dtype == torch.uint8:
            x = x / 255.0
        else:
            if x.numel() > 0 and float(x.max().item()) > 1.5:
                x = x / 255.0

        if x.dim() == 5:
            B, L, C, H, W = x.shape
            z = self.model.obs_enc(x.reshape(B * L, C, H, W)).reshape(B, L, -1)
        elif x.dim() == 3:
            B, L, D = x.shape
            z = self.model.obs_enc(x.reshape(B * L, D)).reshape(B, L, -1)
        else:
            raise ValueError(f"Unexpected obs shape for latent extraction: {tuple(x.shape)}")

        return z

    @staticmethod
    def _diag_stats(z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fit diagonal Gaussian to latent vectors.
        """
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
        """
        Symmetric KL between diagonal Gaussians.
        """
        kl_ab = 0.5 * torch.sum(torch.log(var_b / var_a) + (var_a + (mu_a - mu_b).pow(2)) / var_b - 1.0)
        kl_ba = 0.5 * torch.sum(torch.log(var_a / var_b) + (var_b + (mu_b - mu_a).pow(2)) / var_a - 1.0)
        return 0.5 * (kl_ab + kl_ba)

    def _compute_current_memory_latent_stats(self, task_memory_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            obs = task_memory_obs.to(self.device, dtype=torch.float32).unsqueeze(0)  # [1,N,...]
            z = self._extract_obs_latents(obs)  # [1,N,H]
            mu, var = self._diag_stats(z)
            return mu.detach().cpu(), var.detach().cpu()

    def _store_task_latent_stats(self, task_id: int) -> None:
        """
        Store latent stats of one finished task under its own copy/mask.
        """
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
    def _make_routing_loader(self, task_trajs: List[Trajectory]):
        return make_minibatches(task_trajs, self.seq_len, self.routing_batch_size, self.device)

    def _estimate_action_compatibility(self, source_task_id: int, task_trajs: List[Trajectory]) -> float:
        """
        How well does source-task policy explain expert actions of the NEW task?
        Lower is better.
        """
        self.set_eval_task(int(source_task_id))
        self.model.eval()

        loader = self._make_routing_loader(task_trajs)
        vals: List[float] = []

        with torch.no_grad():
            for _ in range(max(1, self.routing_n_batches)):
                obs, actions, rtg, ts, mask = _unpack_batch(next(loader))
                logits = self.model(obs, actions, rtg, ts, attention_mask=mask)
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    actions.reshape(-1),
                    ignore_index=-1,
                )
                vals.append(float(ce.detach().cpu().item()))

        self.clear_eval_task()
        return float(np.mean(vals)) if vals else float("inf")

    def _estimate_latent_similarity(self, source_task_id: int, task_memory_obs: torch.Tensor) -> float:
        """
        Compare NEW task latents (under source copy/mask) with stored source-task
        latent distribution. Lower is better.
        """
        if int(source_task_id) not in self.task_latent_stats:
            return float("inf")

        self.set_eval_task(int(source_task_id))
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

    def _anchor_first_transfer_route(
        self,
        details: Dict[str, Optional[float]],
    ) -> Optional[Tuple[int, Optional[int], Dict[str, Optional[float]], bool]]:
        """Hard anchor for the first transfer after task 0.

        This fires only for current_task_id == 1.  It forces the first incoming
        task to reuse source task 0 in its existing copy and sets a high
        residual keep ratio.  It does NOT hardcode any return and it does NOT
        affect task 0 training/evaluation.  It only stabilizes the first
        routing decision before the dynamic occupancy policy takes over for
        later tasks.
        """
        if not bool(getattr(self, "anchor_first_reuse", False)):
            return None
        if int(self.current_task_id) != 1:
            return None

        src_task = int(getattr(self, "anchor_first_reuse_source_task", 0))
        if src_task not in self.task_to_copy:
            print(
                "[tsn-anchor-first-atari] SKIP "
                f"task={self.current_task_id} src={src_task} not in task_to_copy",
                flush=True,
            )
            return None

        copy_id = int(self.task_to_copy[src_task])
        rho = float(getattr(self, "anchor_first_reuse_rho", self.residual_keep_ratio))
        self.residual_keep_ratio = rho

        out_details: Dict[str, Optional[float]] = dict(details)
        out_details["anchor_first_reuse"] = 1.0
        out_details["anchor_first_reuse_rho"] = float(rho)
        out_details["anchor_first_reuse_source_task"] = float(src_task)
        out_details["dynamic_rho"] = float(rho)

        print(
            "[tsn-anchor-first-atari] CHOSEN "
            f"task={self.current_task_id} action=reuse copy={copy_id} "
            f"src={src_task} rho={rho:.3f} new_copy=0",
            flush=True,
        )
        return int(copy_id), int(src_task), out_details, False


    def _select_copy_for_new_task_improved(
        self,
        task_memory_obs: torch.Tensor,
        task_trajs: List[Trajectory],
    ) -> Tuple[int, Optional[int], Dict[str, Optional[float]], bool]:
        """
        RL-aware routing.

        Modes:
          - action: choose task with minimum action CE on new-task expert data
          - latent: choose task with minimum latent symmetric-KL
          - hybrid: alpha * normalized(action) + (1-alpha) * normalized(latent)

        Returns:
            (copy_id, source_task_id, score_details, created_new_copy)
        """
        if self.current_task_id == 0 or not self.task_to_copy:
            return 0, None, {"best_action": None, "best_latent": None, "best_score": None, "best_kl": None}, False

        action_scores: Dict[int, float] = {}
        latent_scores: Dict[int, float] = {}
        prev_tasks = sorted(self.task_to_copy.keys())

        for t in prev_tasks:
            action_scores[int(t)] = self._estimate_action_compatibility(int(t), task_trajs)
            if self.reuse_score_mode in ("latent", "hybrid"):
                latent_scores[int(t)] = self._estimate_latent_similarity(int(t), task_memory_obs)

        if self.reuse_score_mode == "action":
            final_scores = dict(action_scores)
            best_task = min(final_scores, key=final_scores.get)
            best_score = final_scores[best_task]
            threshold = self._task_threshold_override(self.action_reuse_threshold)
            create_new = self._should_create_new(int(best_task), float(best_score), threshold)
            details = {
                "threshold": float(threshold),
                "best_action": float(action_scores[best_task]),
                "best_latent": None,
                "best_score": float(best_score),
                "best_kl": None,
            }

        elif self.reuse_score_mode == "latent":
            final_scores = dict(latent_scores)
            best_task = min(final_scores, key=final_scores.get)
            best_score = final_scores[best_task]
            threshold = self._task_threshold_override(self.latent_reuse_threshold)
            create_new = self._should_create_new(int(best_task), float(best_score), threshold)
            details = {
                "threshold": float(threshold),
                "best_action": float(action_scores.get(best_task, float("nan"))),
                "best_latent": float(latent_scores[best_task]),
                "best_score": float(best_score),
                "best_kl": None,
            }

        elif self.reuse_score_mode == "hybrid":
            act_n = self._normalize_score_dict(action_scores) if self.normalize_similarity_scores else action_scores
            lat_n = self._normalize_score_dict(latent_scores) if self.normalize_similarity_scores else latent_scores
            final_scores = {
                t: float(self.hybrid_alpha * act_n[t] + (1.0 - self.hybrid_alpha) * lat_n[t])
                for t in prev_tasks
            }
            best_task = min(final_scores, key=final_scores.get)
            best_score = final_scores[best_task]
            threshold = self._task_threshold_override(self.hybrid_reuse_threshold)
            create_new = self._should_create_new(int(best_task), float(best_score), threshold)
            details = {
                "threshold": float(threshold),
                "best_action": float(action_scores[best_task]),
                "best_latent": float(latent_scores[best_task]),
                "best_score": float(best_score),
                "best_kl": None,
            }

        else:
            raise ValueError(f"Unsupported reuse_score_mode: {self.reuse_score_mode}")

        anchor_route = self._anchor_first_transfer_route(details)
        if anchor_route is not None:
            return anchor_route

        manual_route = self._manual_select_route(final_scores, int(best_task), details)
        if manual_route is not None:
            return manual_route

        if self.routing_policy == "dynamic_occ":
            dynamic_route = self._dynamic_route_from_scores(final_scores, int(best_task), details)
            if dynamic_route is not None:
                return dynamic_route

        if self.probe_routing and int(self.probe_steps) > 0:
            probed = self._probe_select_route(final_scores, int(best_task), details, task_trajs)
            if probed is not None:
                return probed

        if create_new:
            if self.max_model_copies is not None and len(self.copy_states) >= self.max_model_copies:
                copy_id = self.task_to_copy[int(best_task)]
                return int(copy_id), int(best_task), details, False

            new_copy = self._make_fresh_copy()
            self.copy_states.append(new_copy)
            return len(self.copy_states) - 1, int(best_task), details, True

        return int(self.task_to_copy[int(best_task)]), int(best_task), details, False


    # ------------------------------------------------------------------
    # Dynamic occupancy-aware routing (no probe)
    # ------------------------------------------------------------------
    def _dynamic_residual_keep_ratio(self, copy_occ: float, score: float) -> float:
        """Choose residual keep ratio from current copy occupancy and affinity.

        The schedule intentionally mirrors the empirical Atari behavior observed
        in the grid: large residuals help early positive transfer, whereas small
        residuals reduce over-packing when the selected copy is already occupied.
        """
        occ = float(copy_occ)
        # score is currently not thresholded here beyond the outer route check;
        # keep it in the signature so the rule can be extended without changing
        # call sites.
        _ = float(score)

        if occ <= float(self.dynamic_low_occ):
            return float(max(0.0, min(1.0, self.dynamic_rho_high)))
        if occ <= float(self.dynamic_mid_occ):
            return float(max(0.0, min(1.0, self.dynamic_rho_mid)))
        return float(max(0.0, min(1.0, self.dynamic_rho_low)))

    @staticmethod
    def _dynamic_projected_occupancy(copy_occ: float, rho: float) -> float:
        """Approximate copy occupancy after adding a residual mask."""
        occ = float(max(0.0, min(1.0, copy_occ)))
        rho = float(max(0.0, min(1.0, rho)))
        return float(min(1.0, occ + rho * max(0.0, 1.0 - occ)))

    def _dynamic_route_from_scores(
        self,
        final_scores: Dict[int, float],
        best_task: int,
        details: Dict[str, Optional[float]],
    ) -> Optional[Tuple[int, Optional[int], Dict[str, Optional[float]], bool]]:
        """Select a route using affinity plus copy-occupancy pressure.

        This is deliberately simpler than probe/PAR: it does not train temporary
        candidate models and therefore cannot corrupt copy states.  It ranks
        previous task sources by

            affinity_score
            + lambda_occ  * max(0, projected_occ - occ_target)
            + lambda_full * max(0, occ - occ_hard)

        and it may open a new copy only after a configurable number of tasks and
        only if the selected source copy is already too occupied.
        """
        if not final_scores:
            return None

        prev_tasks = sorted(int(t) for t in final_scores.keys())
        if not prev_tasks:
            return None

        # Reuse threshold is mode-dependent.  Lower score is better.
        if self.reuse_score_mode == "action":
            tau_reuse = float(self._task_threshold_override(self.action_reuse_threshold))
        elif self.reuse_score_mode == "latent":
            tau_reuse = float(self._task_threshold_override(self.latent_reuse_threshold))
        else:
            tau_reuse = float(self._task_threshold_override(self.hybrid_reuse_threshold))

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
                "capacity_pressure": float(capacity_pressure),
                "full_pressure": float(full_pressure),
                "objective": float(obj),
            })

            print(
                "[tsn-dynamic-routing-atari] CAND "
                f"task={self.current_task_id} src={int(src_task)} copy={copy_id} "
                f"score={score:.4f} occ={occ:.4f} rho={rho:.3f} "
                f"proj={projected:.4f} obj={obj:.4f}",
                flush=True,
            )

        best_by_score = min(candidates, key=lambda x: x["score"])
        best_by_obj = min(candidates, key=lambda x: x["objective"])

        # Prefer the occupancy-aware candidate only if it is not too much worse
        # in affinity.  This prevents capacity pressure from selecting a clearly
        # unrelated copy.
        if float(best_by_obj["score"]) <= float(best_by_score["score"]) + margin:
            chosen = best_by_obj
            chosen_reason = "objective"
        else:
            chosen = best_by_score
            chosen_reason = "score"

        best_score = float(best_by_score["score"])
        chosen_score = float(chosen["score"])
        chosen_occ = float(chosen["occ"])
        chosen_rho = float(chosen["rho"])
        chosen_proj = float(chosen["projected_occ"])
        chosen_src = int(chosen["src_task"])
        chosen_copy = int(chosen["copy_id"])

        budget = self._capacity_budget()
        can_create = (self.max_model_copies is None) or (len(self.copy_states) < int(budget))

        # Dynamic copy opening.  Task 1 is deliberately protected from spawning:
        # Atari Alien needs early transfer from Breakout in our setting.  Later,
        # if the best selected copy is already too occupied, opening a new copy
        # is allowed under the same K budget.
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
                "[tsn-dynamic-routing-atari] CHOSEN "
                f"task={self.current_task_id} action=new reason={reason} "
                f"copy={copy_id} src={chosen_src} score={chosen_score:.4f} "
                f"tau={tau_reuse:.4f} occ={chosen_occ:.4f} "
                f"rho={chosen_rho:.3f} proj={chosen_proj:.4f} "
                f"obj={float(chosen['objective']):.4f}",
                flush=True,
            )
            return int(copy_id), int(chosen_src), out_details, True

        print(
            "[tsn-dynamic-routing-atari] CHOSEN "
            f"task={self.current_task_id} action=reuse reason={chosen_reason} "
            f"copy={chosen_copy} src={chosen_src} score={chosen_score:.4f} "
            f"tau={tau_reuse:.4f} occ={chosen_occ:.4f} "
            f"rho={chosen_rho:.3f} proj={chosen_proj:.4f} "
            f"obj={float(chosen['objective']):.4f}",
            flush=True,
        )
        return int(chosen_copy), int(chosen_src), out_details, False

    # ------------------------------------------------------------------
    # Probe-Adaptive Routing (PAR)
    # ------------------------------------------------------------------
    def _parse_probe_residual_grid(self) -> List[float]:
        """Parse comma-separated residual keep-ratio candidates."""
        vals: List[float] = []
        for tok in str(self.probe_residual_grid).split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                vals.append(float(tok))
            except ValueError:
                continue

        vals = [float(max(0.0, min(1.0, x))) for x in vals]
        if not vals:
            vals = [float(self.residual_keep_ratio)]
        return sorted(set(vals))

    def _snapshot_rng_for_probe(self) -> Dict[str, object]:
        snap: Dict[str, object] = {
            "torch_cpu": torch.random.get_rng_state(),
            "numpy": np.random.get_state(),
        }
        if torch.cuda.is_available():
            snap["torch_cuda"] = torch.cuda.get_rng_state_all()
        return snap

    def _restore_rng_after_probe(self, snap: Dict[str, object]) -> None:
        torch.random.set_rng_state(snap["torch_cpu"])
        np.random.set_state(snap["numpy"])
        if torch.cuda.is_available() and "torch_cuda" in snap:
            torch.cuda.set_rng_state_all(snap["torch_cuda"])

    def _set_probe_rng(self) -> None:
        seed = int(self.probe_seed) + int(self.current_task_id)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _probe_eval_ce_current_task(
        self,
        task_trajs: List[Trajectory],
        *,
        batch_size: int,
        n_batches: int,
    ) -> float:
        """Validation CE on the current task for a temporary candidate route."""
        loader = make_minibatches(task_trajs, self.seq_len, int(batch_size), self.device)
        vals: List[float] = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(max(1, int(n_batches))):
                obs, actions, rtg, ts, mask = _unpack_batch(next(loader))
                logits = self.model(obs, actions, rtg, ts, attention_mask=mask)
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    actions.reshape(-1),
                    ignore_index=-1,
                )
                vals.append(float(ce.detach().cpu().item()))
        return float(np.mean(vals)) if vals else float("inf")

    def _probe_train_current_task(
        self,
        task_trajs: List[Trajectory],
        *,
        steps: int,
        batch_size: int,
    ) -> None:
        """Short temporary probe training loop using the same protected-TSN rules."""
        loader = make_minibatches(task_trajs, self.seq_len, int(batch_size), self.device)
        self.model.train()
        for _ in range(max(1, int(steps))):
            obs, actions, rtg, ts, mask = _unpack_batch(next(loader))
            logits = self.model(obs, actions, rtg, ts, attention_mask=mask)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                actions.reshape(-1),
                ignore_index=-1,
            )

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self._zero_prev_task_param_grads()
            self._zero_non_maskable_grads()
            frozen_snapshot = self._snapshot_frozen_params()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()
            self._restore_frozen_params_(frozen_snapshot)

    def _probe_one_plan(
            self,
            *,
            task_trajs: List[Trajectory],
            copy_id: int,
            source_task: Optional[int],
            created_new: bool,
            residual_keep_ratio: float,
    ) -> float:
        """
        Score one candidate route without committing it.

        Important:
        This function must not use _activate_copy() during restoration, because
        _activate_copy() first syncs the current public model into copy_states.
        During probing, the public model may be a temporary probe-trained model,
        so using _activate_copy() in finally can corrupt the restored copy bank.
        """
        saved_states = copy.deepcopy(self.copy_states)
        saved_copy_id = int(self.current_copy_id)
        saved_residual_keep_ratio = float(self.residual_keep_ratio)
        saved_current_keep_ratio = float(getattr(self, "current_keep_ratio", self.residual_keep_ratio))
        saved_active_eval_task = getattr(self, "active_eval_task", None)
        saved_task_keep_ratios = copy.deepcopy(getattr(self, "task_keep_ratios", {}))

        rng_snapshot = self._snapshot_rng_for_probe() if self.probe_deterministic else None
        if self.probe_deterministic:
            self._set_probe_rng()

        try:
            # Work only on a scratch copy bank.
            self.copy_states = copy.deepcopy(saved_states)
            self.current_copy_id = min(saved_copy_id, len(self.copy_states) - 1)
            self._restore_active_copy_without_sync(self.current_copy_id)

            if bool(created_new):
                tmp_copy = self._make_fresh_copy()
                self.copy_states.append(tmp_copy)
                probe_copy_id = len(self.copy_states) - 1
            else:
                probe_copy_id = int(copy_id)

            self.residual_keep_ratio = float(residual_keep_ratio)
            self._restore_active_copy_without_sync(probe_copy_id)

            self._prepare_current_task()
            self._clear_residual_base_masks()

            if source_task is not None and (not created_new) and self.residual_reuse:
                self._set_residual_base_from_source_mask(int(source_task))

            if source_task is not None:
                if (not created_new) or self.warmstart_on_new_copy:
                    self._warmstart_scores_from_source_mask(int(source_task))

            self._probe_train_current_task(
                task_trajs,
                steps=int(self.probe_steps),
                batch_size=int(self.probe_batch_size),
            )

            val_ce = self._probe_eval_ce_current_task(
                task_trajs,
                batch_size=int(self.probe_batch_size),
                n_batches=int(self.probe_val_batches),
            )
            return float(val_ce)

        finally:
            # Restore the exact clean copy bank and public model/optimizer.
            self.copy_states = saved_states
            self.current_copy_id = min(saved_copy_id, len(self.copy_states) - 1)
            self.residual_keep_ratio = saved_residual_keep_ratio
            self.current_keep_ratio = saved_current_keep_ratio
            self.active_eval_task = saved_active_eval_task
            self.task_keep_ratios = saved_task_keep_ratios

            self._restore_active_copy_without_sync(self.current_copy_id)
            self._clear_residual_base_masks()

            if rng_snapshot is not None:
                self._restore_rng_after_probe(rng_snapshot)

    def _probe_select_route(
            self,
            final_scores: Dict[int, float],
            best_task: int,
            details: Dict[str, Optional[float]],
            task_trajs: List[Trajectory],
    ) -> Optional[Tuple[int, Optional[int], Dict[str, Optional[float]], bool]]:
        """
        Select route by short offline probe.

        Candidate plans are:
          - reuse top-k affinity sources with several residual keep ratios,
          - optionally create a fresh copy.

        Objective:
          validation_CE
          + new-copy penalty
          + residual penalty
          + projected occupancy pressure.

        The projected occupancy term is a hinge penalty. It starts to matter only
        when the selected copy would exceed probe_occupancy_target. This allows
        useful reuse early in the sequence, while discouraging over-packing later.
        """
        if (not self.probe_routing) or int(self.probe_steps) <= 0:
            return None
        if not final_scores:
            return None

        ranked_sources = sorted(final_scores.keys(), key=lambda t: float(final_scores[t]))
        ranked_sources = ranked_sources[: max(1, int(self.probe_top_k))]
        residual_grid = self._parse_probe_residual_grid()

        occupancy_target = float(getattr(self, "probe_occupancy_target", 0.75))

        candidates: List[Dict[str, object]] = []

        for src in ranked_sources:
            src_i = int(src)
            cid = int(self.task_to_copy.get(src_i, 0))

            for rho in residual_grid:
                candidates.append(
                    {
                        "kind": "reuse",
                        "copy_id": cid,
                        "source_task": src_i,
                        "created_new": False,
                        "rho": float(rho),
                    }
                )

        budget = self._capacity_budget()
        can_create = (self.max_model_copies is None) or (len(self.copy_states) < int(budget))

        if bool(self.probe_include_new_copy) and can_create:
            candidates.append(
                {
                    "kind": "new",
                    "copy_id": len(self.copy_states),
                    "source_task": int(best_task),
                    "created_new": True,
                    "rho": float(getattr(self, "base_keep_ratio", self.residual_keep_ratio)),
                }
            )

        best: Optional[Dict[str, object]] = None

        for cand in candidates:
            created_new = bool(cand["created_new"])
            copy_id = int(cand["copy_id"])
            src_task = None if cand["source_task"] is None else int(cand["source_task"])
            rho = float(cand["rho"])

            val_ce = self._probe_one_plan(
                task_trajs=task_trajs,
                copy_id=copy_id,
                source_task=src_task,
                created_new=created_new,
                residual_keep_ratio=rho,
            )

            if created_new:
                occ = 0.0
                projected_occ = float(getattr(self, "base_keep_ratio", self.current_keep_ratio))
            else:
                occ = float(self._copy_occupied_ratio(copy_id))

                if self.residual_reuse:
                    # Approximate: residual selects rho of currently free capacity.
                    projected_occ = float(min(1.0, occ + rho * max(0.0, 1.0 - occ)))
                else:
                    projected_occ = float(max(occ, getattr(self, "current_keep_ratio", rho)))

            capacity_pressure = float(max(0.0, projected_occ - occupancy_target))

            objective = (
                    float(val_ce)
                    + float(self.probe_new_copy_penalty) * float(created_new)
                    + float(self.probe_residual_penalty) * rho
                    + float(self.probe_occupancy_penalty) * capacity_pressure
            )

            cand["val_ce"] = float(val_ce)
            cand["objective"] = float(objective)
            cand["occupancy"] = float(occ)
            cand["projected_occupancy"] = float(projected_occ)
            cand["capacity_pressure"] = float(capacity_pressure)

            print(
                "[tsn-probe-routing-atari] "
                f"task={self.current_task_id} kind={cand['kind']} "
                f"copy={copy_id} src={src_task} rho={rho:.3f} "
                f"val_ce={float(val_ce):.6e} "
                f"occ={occ:.4f} proj_occ={projected_occ:.4f} "
                f"cap={capacity_pressure:.4f} obj={objective:.6e}",
                flush=True,
            )

            if best is None or float(cand["objective"]) < float(best["objective"]):
                best = cand

        if best is None:
            return None

        chosen_rho = float(best["rho"])
        self.residual_keep_ratio = chosen_rho

        out_details: Dict[str, Optional[float]] = dict(details)
        out_details["probe_val_ce"] = float(best["val_ce"])
        out_details["probe_objective"] = float(best["objective"])
        out_details["probe_residual_keep_ratio"] = chosen_rho

        if bool(best["created_new"]):
            new_copy = self._make_fresh_copy()
            self.copy_states.append(new_copy)
            chosen_copy = len(self.copy_states) - 1

            print(
                "[tsn-probe-routing-atari] CHOSEN "
                f"task={self.current_task_id} kind=new copy={chosen_copy} "
                f"src={int(best['source_task'])} rho={chosen_rho:.3f} "
                f"obj={float(best['objective']):.6e}",
                flush=True,
            )

            return int(chosen_copy), int(best["source_task"]), out_details, True

        print(
            "[tsn-probe-routing-atari] CHOSEN "
            f"task={self.current_task_id} kind=reuse copy={int(best['copy_id'])} "
            f"src={int(best['source_task'])} rho={chosen_rho:.3f} "
            f"obj={float(best['objective']):.6e}",
            flush=True,
        )

        return int(best["copy_id"]), int(best["source_task"]), out_details, False

    # ------------------------------------------------------------------
    # Warm-start from selected source mask
    # ------------------------------------------------------------------
    def _warmstart_scores_from_source_mask(self, source_task: Optional[int]) -> None:
        if (not self.warmstart_source_scores) or (source_task is None):
            return

        st = self._active_state()
        if int(source_task) not in st.per_task_masks:
            return

        src_masks = st.per_task_masks[int(source_task)]

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
    def train_task(self, task_trajs: List[Trajectory], steps: int = 2000, batch_size: int = 64):
        task_memory = self._build_task_memory(task_trajs)

        # task 0 also goes through THIS class, not parent train_task
        if self.current_task_id == 0:
            copy_id = 0
            src_task = None
            score_details = {
                "best_action": None,
                "best_latent": None,
                "best_score": None,
                "best_kl": None,
            }
            created_new = False
        else:
            copy_id, src_task, score_details, created_new = self._select_copy_for_new_task_improved(
                task_memory,
                task_trajs,
            )

        self._activate_copy(copy_id)

        self.task_similarity[self.current_task_id] = {
            "source_task": None if src_task is None else int(src_task),
            "copy_id": int(copy_id),
            "best_action": score_details.get("best_action", None),
            "best_latent": score_details.get("best_latent", None),
            "best_score": score_details.get("best_score", None),
            "best_kl": score_details.get("best_kl", None),
            "score_mode": self.reuse_score_mode,
            "created_new_copy": bool(created_new),
            "routing_policy": self.routing_policy,
            "copy_penalty": float(self.copy_penalty),
            "occupancy_penalty": float(self.occupancy_penalty),
            "residual_reuse": bool(self.residual_reuse),
            "residual_keep_ratio": float(self.residual_keep_ratio),
            "probe_routing": bool(self.probe_routing),
            "probe_steps": int(self.probe_steps),
            "probe_val_ce": score_details.get("probe_val_ce", None),
            "probe_objective": score_details.get("probe_objective", None),
            "probe_residual_keep_ratio": score_details.get("probe_residual_keep_ratio", None),
            "probe_seed": int(self.probe_seed),
            "probe_deterministic": bool(self.probe_deterministic),
            "threshold": score_details.get("threshold", None),
            "manual_route": score_details.get("manual_route", None),
            "manual_action": score_details.get("manual_action", None),
            "manual_residual_keep_ratio": score_details.get("manual_residual_keep_ratio", None),
            "manual_fallback": score_details.get("manual_fallback", None),
            "dynamic_score": score_details.get("dynamic_score", None),
            "dynamic_best_score": score_details.get("dynamic_best_score", None),
            "dynamic_occ": score_details.get("dynamic_occ", None),
            "dynamic_projected_occ": score_details.get("dynamic_projected_occ", None),
            "dynamic_rho": score_details.get("dynamic_rho", None),
            "dynamic_objective": score_details.get("dynamic_objective", None),
            "dynamic_reason": score_details.get("dynamic_reason", None),
        }

        self._prepare_current_task()
        self._clear_residual_base_masks()
        if src_task is not None and (not created_new) and self.residual_reuse:
            self._set_residual_base_from_source_mask(src_task)

        if src_task is not None:
            if (not created_new) or self.warmstart_on_new_copy:
                self._warmstart_scores_from_source_mask(src_task)

        loader = make_minibatches(task_trajs, self.seq_len, batch_size, self.device)

        self.model.train()
        last_loss = None
        for it in range(int(steps)):
            obs, actions, rtg, ts, mask = _unpack_batch(next(loader))
            logits = self.model(obs, actions, rtg, ts, attention_mask=mask)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                actions.reshape(-1),
                ignore_index=-1,
            )

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
                    f"[tsn-improved-reuse-atari] task={self.current_task_id} copy={meta['copy_id']} "
                    f"src={meta['source_task']} score_mode={meta['score_mode']} "
                    f"best_action={meta['best_action']} best_latent={meta['best_latent']} "
                    f"best_score={meta['best_score']} new_copy={int(meta['created_new_copy'])} "
                    f"it={it} ce={last_loss:.6e} keep_ratio={self.current_keep_ratio:.4f}"
                )

        self.task_memories[self.current_task_id] = task_memory.detach().cpu()
        self._sync_public_state_to_active_copy()

        return {"loss": last_loss, "keep_ratio": float(self.current_keep_ratio)}

    def after_task(self, task_trajs: List[Trajectory]):
        self._sync_public_state_to_active_copy()

        task_id = int(self.current_task_id)
        st = self._active_state()

        task_masks = self._collect_current_task_masks()

        # Reuse accounting for improved routing. Compute BEFORE storing the
        # current mask into per_task_masks/consolidated_masks.
        meta = self.task_similarity.get(task_id, {})
        src_task = meta.get("source_task", None)
        prev_masks_same_copy = dict(st.per_task_masks)
        reuse_stats = compute_mask_reuse_stats(
            task_id=task_id,
            copy_id=int(self.current_copy_id),
            current_task_masks=task_masks,
            previous_task_masks=prev_masks_same_copy,
            source_task=None if src_task is None else int(src_task),
            weight_only=True,
        )
        if not hasattr(self, "task_weight_reuse_stats") or self.task_weight_reuse_stats is None:
            self.task_weight_reuse_stats = {}
        self.task_weight_reuse_stats[task_id] = reuse_stats
        print_mask_reuse_stats("[reuse-accounting-improved-atari]", reuse_stats)

        st.per_task_masks[task_id] = task_masks
        st.task_codebooks[task_id] = self._quantize_new_weights_for_current_task(task_masks)
        self._update_consolidated_masks(task_masks)

        self.task_to_copy[task_id] = int(self.current_copy_id)

        if self.reuse_score_mode in ("latent", "hybrid"):
            self._store_task_latent_stats(task_id)

        used = 0
        total = 0
        for key, mask in st.consolidated_masks.items():
            if mask is None or not key.endswith(".weight"):
                continue
            used += int(mask.sum().item())
            total += int(mask.numel())
        ratio = float(used / max(1, total))

        meta = self.task_similarity.get(task_id, {})
        print(
            f"[tsn-improved-reuse-atari] after task {task_id}: copy={self.current_copy_id} "
            f"occupied_ratio={ratio:.4f} source_task={meta.get('source_task', None)} "
            f"best_action={meta.get('best_action', None)} "
            f"best_latent={meta.get('best_latent', None)} "
            f"best_score={meta.get('best_score', None)} "
            f"created_new_copy={meta.get('created_new_copy', None)}"
        )

        self.set_eval_task(task_id)
        self.current_task_id += 1