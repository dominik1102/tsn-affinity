from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dt.dataset import Trajectory
from dt.panda_dt import PandaDecisionTransformer

from .tsn_common import (
    TSNConversionConfig,
    _convert_module_inplace,
    _iter_tsn_modules,
    _kmeans_quantize_selected_,
    _rebuild_optimizer_like,
)
from .utils import (
    prepare_panda_trajs,
    make_panda_loader,
    unpack_batch_continuous,
)


"""
Patched Panda old-reuse TSN.

Fixes in this version:
  1) task-memory sampling uses a LOCAL RNG instead of consuming the global
     NumPy RNG used later by loaders;
  2) stored task masks are always recomputed from the FINAL score tensors,
     instead of reusing potentially stale cached masks from before the last
     optimizer step.
"""


def _compute_obs_stats(trajs: List[Trajectory], obs_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    if not trajs:
        return np.zeros(obs_dim, dtype=np.float32), np.ones(obs_dim, dtype=np.float32)
    xs = [np.asarray(t.obs, dtype=np.float32).reshape(-1, obs_dim) for t in trajs]
    x = np.concatenate(xs, axis=0)
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std = np.clip(std, 1e-6, None)
    return mean, std


def _infer_raw_act_dim(task_trajs: List[Trajectory], act_dim_global: int) -> int:
    max_abs = np.zeros((act_dim_global,), dtype=np.float32)
    for tr in task_trajs[: min(len(task_trajs), 2048)]:
        a = np.asarray(tr.actions, dtype=np.float32)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        cur = min(a.shape[-1], act_dim_global)
        max_abs[:cur] = np.maximum(max_abs[:cur], np.max(np.abs(a[:, :cur]), axis=0))
    active = (max_abs > 1e-8).astype(np.int64)
    n = int(active.sum())
    return max(1, n)


def _masked_mse_with_action_mask(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    action_dim_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if action_dim_mask is not None:
        adm = action_dim_mask.to(device=pred.device, dtype=pred.dtype).view(1, 1, -1)
        sq = ((pred - target) ** 2) * adm
        denom_d = adm.sum(dim=-1).clamp(min=1.0)
        mse_per_step = sq.sum(dim=-1) / denom_d
    else:
        mse_per_step = ((pred - target) ** 2).mean(dim=-1)

    m = mask.float()
    denom = m.sum().clamp(min=1.0)
    return (mse_per_step * m).sum() / denom


@dataclass
class _ReuseCopyStatePanda:
    model: nn.Module
    opt: torch.optim.Optimizer
    per_task_masks: Dict[int, Dict[str, Optional[torch.Tensor]]]
    consolidated_masks: Dict[str, Optional[torch.Tensor]]
    task_codebooks: Dict[int, Dict[str, np.ndarray]]
    task_keep_ratios: Dict[int, float]
    per_task_obs_stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
    per_task_action_dims: Dict[int, int]


class TSNOriginalReusePandaStrategy:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        seq_len: int,
        device: str,
        *,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 1,
        p_drop: float = 0.1,
        max_ep_len: int = 50,
        act_tanh: bool = False,
        rtg_scale: float = 1000.0,
        obs_mean: Optional[np.ndarray] = None,
        obs_std: Optional[np.ndarray] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        grad_clip: float = 0.25,
        keep_ratio: float = 0.5,
        include_embeddings: bool = False,
        quantize_after_task: bool = True,
        quant_clusters: int = 16,
        freeze_non_mask_params_after_first: bool = True,
        skip_module_names: Tuple[str, ...] = ("dt.te",),
        expected_num_tasks: Optional[int] = None,
        keep_ratio_schedule: str = "constant",
        min_keep_ratio: float = 1e-3,
        reuse_memory_size: int = 256,
        reuse_kl_threshold: float = 0.25,
        max_model_copies: Optional[int] = None,
        store_task_obs_stats: bool = True,
        patch_model_act: bool = True,
    ):
        self.device = torch.device(device)
        self.seq_len = int(seq_len)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        self.base_keep_ratio = float(keep_ratio)
        self.current_keep_ratio = float(keep_ratio)
        self.include_embeddings = bool(include_embeddings)
        self.quantize_after_task = bool(quantize_after_task)
        self.quant_clusters = int(quant_clusters)
        self.allow_weight_reuse = True
        self.freeze_non_mask_params_after_first = bool(freeze_non_mask_params_after_first)
        self.skip_module_names = tuple(skip_module_names)
        self.expected_num_tasks = None if expected_num_tasks is None else int(expected_num_tasks)
        self.keep_ratio_schedule = str(keep_ratio_schedule)
        self.min_keep_ratio = float(min_keep_ratio)
        self.grad_clip = float(grad_clip)

        self.reuse_memory_size = int(reuse_memory_size)
        self.reuse_kl_threshold = float(reuse_kl_threshold)
        self.max_model_copies = None if max_model_copies is None else int(max_model_copies)

        self.store_task_obs_stats = bool(store_task_obs_stats)
        self.patch_model_act = bool(patch_model_act)

        self.model_hparams = {
            "obs_dim": int(obs_dim),
            "act_dim": int(act_dim),
            "seq_len": int(seq_len),
            "d_model": int(d_model),
            "n_layers": int(n_layers),
            "n_heads": int(n_heads),
            "p_drop": float(p_drop),
            "max_ep_len": int(max_ep_len),
            "act_tanh": bool(act_tanh),
            "rtg_scale": float(rtg_scale),
            "obs_mean": None if obs_mean is None else np.asarray(obs_mean, dtype=np.float32).copy(),
            "obs_std": None if obs_std is None else np.asarray(obs_std, dtype=np.float32).copy(),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "grad_clip": float(grad_clip),
        }

        self.current_task_id: int = 0
        self.current_copy_id: int = 0
        self.active_eval_task: Optional[int] = None

        self.task_to_copy: Dict[int, int] = {}
        self.task_memories: Dict[int, torch.Tensor] = {}
        self.task_keep_ratios: Dict[int, float] = {}
        self.task_similarity: Dict[int, Dict[str, Optional[float]]] = {}

        self._last_task_obs_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._last_task_action_dim: Optional[int] = None

        first_copy = self._make_fresh_copy()
        self.copy_states: List[_ReuseCopyStatePanda] = [first_copy]
        self._activate_copy(0)

    def _patch_model_act_for_action_truncation(self, model: nn.Module) -> None:
        if not self.patch_model_act:
            return
        if not hasattr(model, "act"):
            return
        if getattr(model, "_tsn_act_patched", False):
            return

        old_act = model.act

        def _wrapped_act(*args, **kwargs):
            out = old_act(*args, **kwargs)
            active_dim = getattr(model, "_tsn_active_action_dim", None)
            if active_dim is None:
                return out
            active_dim = int(active_dim)
            out_np = np.asarray(out, dtype=np.float32).reshape(-1)
            return out_np[:active_dim]

        model.act = _wrapped_act  # type: ignore[assignment]
        model._tsn_act_patched = True
        model._tsn_active_action_dim = None

    def _make_fresh_copy(self) -> _ReuseCopyStatePanda:
        hp = self.model_hparams
        model = PandaDecisionTransformer(
            obs_dim=int(hp["obs_dim"]),
            act_dim=int(hp["act_dim"]),
            d_model=int(hp["d_model"]),
            n_layers=int(hp["n_layers"]),
            n_heads=int(hp["n_heads"]),
            seq_len=int(hp["seq_len"]),
            p_drop=float(hp["p_drop"]),
            max_ep_len=int(hp["max_ep_len"]),
            act_tanh=bool(hp["act_tanh"]),
            obs_mean=hp["obs_mean"],
            obs_std=hp["obs_std"],
            rtg_scale=float(hp["rtg_scale"]),
        ).to(self.device)

        cfg = TSNConversionConfig(
            keep_ratio=float(self.base_keep_ratio),
            include_embeddings=bool(self.include_embeddings),
            allow_weight_reuse=True,
            skip_module_names=tuple(self.skip_module_names),
        )
        _convert_module_inplace(model, cfg)
        model.to(self.device)

        self._patch_model_act_for_action_truncation(model)

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(hp["lr"]),
            weight_decay=float(hp["weight_decay"]),
        )
        opt = _rebuild_optimizer_like(opt, model.parameters())

        return _ReuseCopyStatePanda(
            model=model,
            opt=opt,
            per_task_masks={},
            consolidated_masks={},
            task_codebooks={},
            task_keep_ratios={},
            per_task_obs_stats={},
            per_task_action_dims={},
        )

    def _activate_copy(self, copy_id: int) -> None:
        self.current_copy_id = int(copy_id)
        st = self.copy_states[self.current_copy_id]
        self.model = st.model
        self.opt = st.opt
        self._refresh_name_sets()

    def _build_task_memory(self, task_trajs: List[Trajectory]) -> torch.Tensor:
        obs_chunks: List[np.ndarray] = []
        total = 0
        for tr in task_trajs:
            obs = np.asarray(tr.obs, dtype=np.float32)
            if obs.ndim < 2:
                continue
            obs_chunks.append(obs.reshape(-1, self.obs_dim))
            total += int(obs.shape[0])
            if total >= self.reuse_memory_size * 4:
                break

        if not obs_chunks:
            raise ValueError("TSNOriginalReusePandaStrategy: cannot build task memory from empty observations")

        obs_all = np.concatenate(obs_chunks, axis=0)
        n = min(int(obs_all.shape[0]), int(self.reuse_memory_size))

        rng = np.random.default_rng(10_000 + int(self.current_task_id))
        idx = rng.choice(obs_all.shape[0], size=n, replace=False)

        mem = torch.as_tensor(obs_all[idx], dtype=torch.float32)
        return mem

    def _memory_kl(self, mem_a: torch.Tensor, mem_b: torch.Tensor) -> float:
        a = mem_a.reshape(mem_a.shape[0], -1)
        b = mem_b.reshape(mem_b.shape[0], -1)
        m = min(int(a.shape[0]), int(b.shape[0]))
        a = a[:m]
        b = b[:m]
        a_logp = F.log_softmax(a, dim=-1)
        b_p = F.softmax(b, dim=-1)
        kl = F.kl_div(a_logp, b_p, reduction="batchmean")
        return float(kl.detach().cpu().item())

    def _select_copy_for_new_task(self, task_memory: torch.Tensor) -> Tuple[int, Optional[int], Optional[float], bool]:
        if self.current_task_id == 0 or not self.task_memories:
            return 0, None, None, False

        best_task: Optional[int] = None
        best_kl: Optional[float] = None
        for t, mem in self.task_memories.items():
            kl = self._memory_kl(task_memory, mem)
            if (best_kl is None) or (kl < best_kl):
                best_kl = kl
                best_task = int(t)

        create_new = False
        if best_kl is None:
            create_new = True
        elif best_kl > self.reuse_kl_threshold:
            create_new = True

        if create_new:
            if self.max_model_copies is not None and len(self.copy_states) >= self.max_model_copies:
                copy_id = self.task_to_copy[best_task] if best_task is not None else 0
                return int(copy_id), best_task, best_kl, False
            new_copy = self._make_fresh_copy()
            self.copy_states.append(new_copy)
            return len(self.copy_states) - 1, best_task, best_kl, True

        assert best_task is not None
        return int(self.task_to_copy[best_task]), best_task, best_kl, False

    def _refresh_name_sets(self) -> None:
        self._maskable_param_names = set()
        self._score_param_names = set()
        for mod_name, mod in _iter_tsn_modules(self.model):
            self._maskable_param_names.add(f"{mod_name}.weight")
            self._score_param_names.add(f"{mod_name}.score")
            if getattr(mod, "bias", None) is not None:
                self._maskable_param_names.add(f"{mod_name}.bias")
            if getattr(mod, "bias_score", None) is not None:
                self._score_param_names.add(f"{mod_name}.bias_score")

    def _active_state(self) -> _ReuseCopyStatePanda:
        return self.copy_states[self.current_copy_id]

    def _reset_all_scores(self) -> None:
        for _, mod in _iter_tsn_modules(self.model):
            mod.reset_scores()
            mod.clear_active_masks()

    def _sync_occupied_masks_into_modules(self) -> None:
        st = self._active_state()
        for name, mod in _iter_tsn_modules(self.model):
            mod.clear_occupied_masks()
            w_key = f"{name}.weight"
            b_key = f"{name}.bias"
            if w_key in st.consolidated_masks and st.consolidated_masks[w_key] is not None:
                mod.occupied_weight_mask = st.consolidated_masks[w_key].detach().clone()
            if b_key in st.consolidated_masks and st.consolidated_masks[b_key] is not None:
                mod.occupied_bias_mask = st.consolidated_masks[b_key].detach().clone()

    @staticmethod
    def _recompute_module_task_masks(mod) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            if hasattr(mod, "weight_mask"):
                mod.weight_mask = None
            if hasattr(mod, "bias_mask"):
                mod.bias_mask = None
            w_mask = mod._current_weight_mask()
            b_mask = mod._current_bias_mask() if getattr(mod, "bias", None) is not None else None
        return w_mask, b_mask

    def _collect_current_task_masks(self) -> Dict[str, Optional[torch.Tensor]]:
        out: Dict[str, Optional[torch.Tensor]] = {}
        for name, mod in _iter_tsn_modules(self.model):
            w_mask, b_mask = self._recompute_module_task_masks(mod)
            out[f"{name}.weight"] = None if w_mask is None else w_mask.detach().cpu().clone()
            out[f"{name}.bias"] = None if b_mask is None else b_mask.detach().cpu().clone()
        return out

    def _update_consolidated_masks(self, task_masks: Dict[str, Optional[torch.Tensor]]) -> None:
        st = self._active_state()
        if not st.consolidated_masks:
            st.consolidated_masks = {k: (None if v is None else v.clone()) for k, v in task_masks.items()}
            return
        for key, mask in task_masks.items():
            if mask is None:
                st.consolidated_masks.setdefault(key, None)
                continue
            if st.consolidated_masks.get(key) is None:
                st.consolidated_masks[key] = mask.clone()
            else:
                st.consolidated_masks[key] = torch.logical_or(
                    st.consolidated_masks[key].bool(),
                    mask.bool(),
                ).to(torch.uint8)

    def _apply_task_obs_stats(self, task_id: Optional[int]) -> None:
        if not self.store_task_obs_stats:
            return
        if not hasattr(self.model, "obs_mean") or not hasattr(self.model, "obs_std"):
            return
        if task_id is None:
            return
        copy_id = self.task_to_copy.get(int(task_id), None)
        if copy_id is None:
            return
        st = self.copy_states[copy_id]
        if int(task_id) not in st.per_task_obs_stats:
            return
        mean_t, std_t = st.per_task_obs_stats[int(task_id)]
        with torch.no_grad():
            self.model.obs_mean.copy_(mean_t.to(device=self.model.obs_mean.device, dtype=self.model.obs_mean.dtype))
            self.model.obs_std.copy_(std_t.to(device=self.model.obs_std.device, dtype=self.model.obs_std.dtype))

    def _apply_eval_masks(self, task_id: Optional[int]) -> None:
        if task_id is None:
            for _, mod in _iter_tsn_modules(self.model):
                mod.clear_active_masks()
            return

        copy_id = self.task_to_copy.get(int(task_id), None)
        if copy_id is None:
            for _, mod in _iter_tsn_modules(self.model):
                mod.clear_active_masks()
            return

        self._activate_copy(copy_id)
        st = self._active_state()
        task_masks = st.per_task_masks.get(int(task_id), None)
        for name, mod in _iter_tsn_modules(self.model):
            if task_masks is None:
                mod.clear_active_masks()
                continue
            mod.active_weight_mask = task_masks.get(f"{name}.weight", None)
            mod.active_bias_mask = task_masks.get(f"{name}.bias", None)

    def has_task_mask(self, task_id: int) -> bool:
        copy_id = self.task_to_copy.get(int(task_id), None)
        if copy_id is None:
            return False
        return int(task_id) in self.copy_states[copy_id].per_task_masks

    def set_eval_task(self, task_id: int) -> None:
        task_id = int(task_id)
        self.active_eval_task = task_id if task_id in self.task_to_copy else None
        self._apply_eval_masks(self.active_eval_task)
        self._apply_task_obs_stats(self.active_eval_task)
        if hasattr(self.model, "_tsn_active_action_dim"):
            copy_id = self.task_to_copy.get(task_id, None)
            if copy_id is None:
                self.model._tsn_active_action_dim = None
            else:
                self.model._tsn_active_action_dim = self.copy_states[copy_id].per_task_action_dims.get(task_id, None)

    def clear_eval_task(self) -> None:
        self.active_eval_task = None
        self._apply_eval_masks(None)
        if hasattr(self.model, "_tsn_active_action_dim"):
            self.model._tsn_active_action_dim = None

    def _compute_task_keep_ratio(self) -> float:
        if self.keep_ratio_schedule == "constant" or self.expected_num_tasks is None:
            return float(max(self.min_keep_ratio, min(1.0, self.base_keep_ratio)))
        if self.keep_ratio_schedule == "equal_remaining":
            remaining = max(1, int(self.expected_num_tasks) - int(self.current_task_id))
            value = 1.0 / float(remaining)
            return float(max(self.min_keep_ratio, min(1.0, value)))
        raise ValueError(f"Unsupported keep_ratio_schedule: {self.keep_ratio_schedule}")

    def _apply_keep_ratio_to_modules(self, keep_ratio: float) -> None:
        self.current_keep_ratio = float(keep_ratio)
        for _, mod in _iter_tsn_modules(self.model):
            mod.keep_ratio = float(keep_ratio)

    def _zero_prev_task_param_grads(self) -> None:
        st = self._active_state()
        if not st.consolidated_masks:
            return
        name_to_module = dict(self.model.named_modules())
        for key, old_mask in st.consolidated_masks.items():
            if old_mask is None:
                continue
            module_name, attr = key.rsplit(".", 1)
            module = name_to_module.get(module_name, None)
            if module is None:
                continue
            param = getattr(module, attr, None)
            if param is None or param.grad is None:
                continue
            mask = old_mask.to(device=param.grad.device, dtype=torch.bool)
            param.grad.masked_fill_(mask, 0.0)

    def _zero_non_maskable_grads(self) -> None:
        if not self.freeze_non_mask_params_after_first:
            return
        st = self._active_state()
        if not st.consolidated_masks:
            return
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if name in self._maskable_param_names or name in self._score_param_names:
                continue
            param.grad.zero_()

    def _snapshot_frozen_params(self) -> Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        snap: Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]] = {}
        st = self._active_state()
        if not st.consolidated_masks:
            return snap
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            old_mask = st.consolidated_masks.get(name, None)
            if old_mask is not None:
                snap[name] = (
                    param.detach().clone(),
                    old_mask.to(device=param.device, dtype=torch.bool),
                )
                continue
            if self.freeze_non_mask_params_after_first:
                if name not in self._maskable_param_names and name not in self._score_param_names:
                    snap[name] = (param.detach().clone(), None)
        return snap

    def _restore_frozen_params_(self, snapshot: Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]]) -> None:
        if not snapshot:
            return
        name_to_param = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, (old_value, restore_mask) in snapshot.items():
                param = name_to_param.get(name, None)
                if param is None:
                    continue
                if restore_mask is None:
                    param.copy_(old_value)
                else:
                    param.copy_(torch.where(restore_mask, old_value, param))

    def _prepare_current_task(self) -> None:
        self.clear_eval_task()
        self._sync_occupied_masks_into_modules()
        keep_ratio = self._compute_task_keep_ratio()
        self._apply_keep_ratio_to_modules(keep_ratio)
        self.task_keep_ratios[self.current_task_id] = float(keep_ratio)
        st = self._active_state()
        st.task_keep_ratios[self.current_task_id] = float(keep_ratio)
        self._reset_all_scores()
        st.opt = _rebuild_optimizer_like(st.opt, self.model.parameters())
        self.opt = st.opt

    def _quantize_new_weights_for_current_task(self, task_masks: Dict[str, Optional[torch.Tensor]]) -> Dict[str, np.ndarray]:
        codebooks: Dict[str, np.ndarray] = {}
        if not self.quantize_after_task:
            return codebooks

        st = self._active_state()
        name_to_module = dict(self.model.named_modules())
        for key, mask in task_masks.items():
            if mask is None or not key.endswith(".weight"):
                continue
            module_name, _ = key.rsplit(".", 1)
            module = name_to_module.get(module_name, None)
            if module is None:
                continue
            prev_mask = st.consolidated_masks.get(key, None)
            if prev_mask is None:
                new_mask = mask.bool()
            else:
                new_mask = torch.logical_and(mask.bool(), ~prev_mask.bool())
            if int(new_mask.sum().item()) == 0:
                continue
            centers = _kmeans_quantize_selected_(module.weight, new_mask, self.quant_clusters)
            if centers is not None:
                codebooks[key] = centers
        return codebooks

    def train_task(
        self,
        task_trajs: List[Trajectory],
        steps: int = 2000,
        batch_size: int = 64,
        *,
        active_action_dim_mask: Optional[List[int]] = None,
    ):
        if not task_trajs:
            raise ValueError("TSNOriginalReusePandaStrategy.train_task got empty task_trajs")

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
        copy_id, src_task, best_kl, created_new = self._select_copy_for_new_task(task_memory)
        self._activate_copy(copy_id)
        self.task_similarity[self.current_task_id] = {
            "source_task": None if src_task is None else int(src_task),
            "copy_id": int(copy_id),
            "best_kl": None if best_kl is None else float(best_kl),
            "created_new_copy": bool(created_new),
        }

        if self.store_task_obs_stats and self._last_task_obs_stats is not None:
            mean_np, std_np = self._last_task_obs_stats
            if hasattr(self.model, "obs_mean") and hasattr(self.model, "obs_std"):
                with torch.no_grad():
                    self.model.obs_mean.copy_(torch.as_tensor(mean_np, dtype=self.model.obs_mean.dtype, device=self.model.obs_mean.device))
                    self.model.obs_std.copy_(torch.as_tensor(std_np, dtype=self.model.obs_std.dtype, device=self.model.obs_std.device))

        self._prepare_current_task()

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
                    f"[tsn-old-reuse-panda] task={self.current_task_id} copy={meta['copy_id']} "
                    f"src={meta['source_task']} kl={meta['best_kl']} new_copy={int(meta['created_new_copy'])} "
                    f"it={it} bc={last_loss:.6e} keep_ratio={self.current_keep_ratio:.4f} "
                    f"active_action_dim_mask={active_action_dim_mask}"
                )

        self.task_memories[self.current_task_id] = task_memory.detach().cpu()
        return {"bc_loss": last_loss, "keep_ratio": float(self.current_keep_ratio)}

    def after_task(self, task_trajs: List[Trajectory]):
        task_id = int(self.current_task_id)
        st = self._active_state()

        task_masks = self._collect_current_task_masks()
        st.per_task_masks[task_id] = task_masks
        st.task_codebooks[task_id] = self._quantize_new_weights_for_current_task(task_masks)
        self._update_consolidated_masks(task_masks)

        if self.store_task_obs_stats and self._last_task_obs_stats is not None:
            mean_np, std_np = self._last_task_obs_stats
            st.per_task_obs_stats[task_id] = (
                torch.as_tensor(mean_np, dtype=torch.float32).cpu(),
                torch.as_tensor(std_np, dtype=torch.float32).cpu(),
            )

        if self._last_task_action_dim is not None:
            st.per_task_action_dims[task_id] = int(self._last_task_action_dim)

        self.task_to_copy[task_id] = int(self.current_copy_id)

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
            f"[tsn-old-reuse-panda] after task {task_id}: copy={self.current_copy_id} occupied_ratio={ratio:.4f} "
            f"source_task={meta.get('source_task', None)} best_kl={meta.get('best_kl', None)}"
        )

        self.set_eval_task(task_id)
        self.current_task_id += 1
