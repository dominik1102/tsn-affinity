from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from dt.dataset import Trajectory

# Flexible Panda model import ladder
_PANDA_MODEL_IMPORT_ERR = None
PandaDecisionTransformer = None
for _cand in (
    "dt.panda_model",
    "dt.model_panda",
    "dt.panda_dt",
    "dt.panda_transformer",
):
    try:
        mod = __import__(_cand, fromlist=["PandaDecisionTransformer"])
        PandaDecisionTransformer = getattr(mod, "PandaDecisionTransformer")
        break
    except Exception as e:
        _PANDA_MODEL_IMPORT_ERR = e

from .tsn_common import (
    TSNConversionConfig,
    _convert_module_inplace,
    _iter_tsn_modules,
    _kmeans_quantize_selected_,
)
from .utils import (
    compute_obs_stats,
    infer_act_dim,
    infer_obs_dim,
    make_panda_loader,
    masked_mse,
    prepare_panda_trajs,
    unpack_batch_continuous,
)


class TSNPandaStrategy:
    """
    TinySubNets-style Panda strategy for continuous-action DT.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        seq_len: int,
        device: str,
        *,
        model: Optional[nn.Module] = None,
        model_ctor: Optional[Callable[..., nn.Module]] = None,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        p_drop: float = 0.1,
        max_ep_len: int = 256,
        act_tanh: bool = False,
        rtg_scale: float = 1.0,
        obs_mean: Optional[np.ndarray] = None,
        obs_std: Optional[np.ndarray] = None,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        keep_ratio: float = 0.5,
        include_embeddings: bool = False,
        quantize_after_task: bool = True,
        quant_clusters: int = 16,
        allow_weight_reuse: bool = False,
        freeze_non_mask_params_after_first: bool = True,
        store_task_obs_stats: bool = True,
        patch_model_act: bool = True,
        skip_module_names: Tuple[str, ...] = ("dt.te",),
    ):
        self.device = torch.device(device)
        self.seq_len = int(seq_len)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        if model is not None:
            self.model = model
        else:
            ctor = model_ctor if model_ctor is not None else PandaDecisionTransformer
            if ctor is None:
                raise ImportError(
                    "Could not import PandaDecisionTransformer automatically. "
                    "Pass model=... or model_ctor=... to TSNPandaStrategy. "
                    f"Last import error: {_PANDA_MODEL_IMPORT_ERR!r}"
                )
            self.model = ctor(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                d_model=int(d_model),
                n_layers=int(n_layers),
                n_heads=int(n_heads),
                seq_len=self.seq_len,
                p_drop=float(p_drop),
                max_ep_len=int(max_ep_len),
                act_tanh=bool(act_tanh),
                obs_mean=obs_mean,
                obs_std=obs_std,
                rtg_scale=float(rtg_scale),
            )

        cfg = TSNConversionConfig(
            keep_ratio=float(keep_ratio),
            include_embeddings=bool(include_embeddings),
            allow_weight_reuse=bool(allow_weight_reuse),
            skip_module_names=tuple(skip_module_names),
        )
        _convert_module_inplace(self.model, cfg)
        self.model.to(self.device)

        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )

        self.keep_ratio = float(keep_ratio)
        self.grad_clip = float(grad_clip)
        self.include_embeddings = bool(include_embeddings)
        self.quantize_after_task = bool(quantize_after_task)
        self.quant_clusters = int(quant_clusters)
        self.allow_weight_reuse = bool(allow_weight_reuse)
        self.freeze_non_mask_params_after_first = bool(freeze_non_mask_params_after_first)
        self.store_task_obs_stats = bool(store_task_obs_stats)
        self.patch_model_act = bool(patch_model_act)
        self.skip_module_names = tuple(skip_module_names)

        self.current_task_id: int = 0
        self.per_task_masks: Dict[int, Dict[str, Optional[torch.Tensor]]] = {}
        self.consolidated_masks: Dict[str, Optional[torch.Tensor]] = {}
        self.task_codebooks: Dict[int, Dict[str, np.ndarray]] = {}
        self.per_task_obs_stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.per_task_action_dims: Dict[int, int] = {}
        self.active_eval_task: Optional[int] = None

        self._last_task_obs_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._last_task_action_dim: Optional[int] = None
        self._last_task_obs_dim: Optional[int] = None

        self._refresh_name_sets()
        if self.patch_model_act:
            self._patch_model_act_for_action_truncation()
        self.clear_eval_task()

    def _patch_model_act_for_action_truncation(self) -> None:
        if not hasattr(self.model, "act"):
            return
        if getattr(self.model, "_tsn_act_patched", False):
            return

        old_act = self.model.act

        def _wrapped_act(*args, **kwargs):
            out = old_act(*args, **kwargs)
            active_dim = getattr(self.model, "_tsn_active_action_dim", None)
            if active_dim is None:
                return out
            active_dim = int(active_dim)
            out_np = np.asarray(out, dtype=np.float32).reshape(-1)
            return out_np[:active_dim]

        self.model.act = _wrapped_act  # type: ignore[assignment]
        self.model._tsn_act_patched = True
        self.model._tsn_active_action_dim = None

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

    def _reset_all_scores(self) -> None:
        for _, mod in _iter_tsn_modules(self.model):
            mod.reset_scores()
            mod.clear_active_masks()

    def _sync_occupied_masks_into_modules(self) -> None:
        for name, mod in _iter_tsn_modules(self.model):
            mod.clear_occupied_masks()
            w_key = f"{name}.weight"
            b_key = f"{name}.bias"
            if w_key in self.consolidated_masks and self.consolidated_masks[w_key] is not None:
                mod.occupied_weight_mask = self.consolidated_masks[w_key].detach().clone()
            if b_key in self.consolidated_masks and self.consolidated_masks[b_key] is not None:
                mod.occupied_bias_mask = self.consolidated_masks[b_key].detach().clone()

    def _collect_current_task_masks(self) -> Dict[str, Optional[torch.Tensor]]:
        out: Dict[str, Optional[torch.Tensor]] = {}
        for name, mod in _iter_tsn_modules(self.model):
            if mod.weight_mask is None:
                with torch.no_grad():
                    _ = mod._current_weight_mask()
                    if getattr(mod, "bias", None) is not None:
                        _ = mod._current_bias_mask()
            out[f"{name}.weight"] = None if mod.weight_mask is None else mod.weight_mask.detach().cpu().clone()
            out[f"{name}.bias"] = None if mod.bias_mask is None else mod.bias_mask.detach().cpu().clone()
        return out

    def _update_consolidated_masks(self, task_masks: Dict[str, Optional[torch.Tensor]]) -> None:
        if not self.consolidated_masks:
            self.consolidated_masks = {k: (None if v is None else v.clone()) for k, v in task_masks.items()}
            return
        for key, mask in task_masks.items():
            if mask is None:
                self.consolidated_masks.setdefault(key, None)
                continue
            if self.consolidated_masks.get(key) is None:
                self.consolidated_masks[key] = mask.clone()
            else:
                self.consolidated_masks[key] = torch.logical_or(self.consolidated_masks[key].bool(), mask.bool()).to(torch.uint8)

    def _apply_eval_masks(self, task_id: Optional[int]) -> None:
        task_masks = self.per_task_masks.get(task_id, None) if task_id is not None else None
        for name, mod in _iter_tsn_modules(self.model):
            if task_masks is None:
                mod.clear_active_masks()
                continue
            mod.active_weight_mask = task_masks.get(f"{name}.weight", None)
            mod.active_bias_mask = task_masks.get(f"{name}.bias", None)

    def _apply_task_obs_stats(self, task_id: Optional[int]) -> None:
        if not self.store_task_obs_stats:
            return
        if not hasattr(self.model, "obs_mean") or not hasattr(self.model, "obs_std"):
            return
        if task_id is None or task_id not in self.per_task_obs_stats:
            return
        mean_t, std_t = self.per_task_obs_stats[task_id]
        with torch.no_grad():
            self.model.obs_mean.copy_(mean_t.to(device=self.model.obs_mean.device, dtype=self.model.obs_mean.dtype))
            self.model.obs_std.copy_(std_t.to(device=self.model.obs_std.device, dtype=self.model.obs_std.dtype))

    def has_task_mask(self, task_id: int) -> bool:
        return int(task_id) in self.per_task_masks

    def set_eval_task(self, task_id: int) -> None:
        task_id = int(task_id)
        self.active_eval_task = task_id if task_id in self.per_task_masks else None
        self._apply_eval_masks(self.active_eval_task)
        self._apply_task_obs_stats(self.active_eval_task)
        if hasattr(self.model, "_tsn_active_action_dim"):
            self.model._tsn_active_action_dim = self.per_task_action_dims.get(task_id, None)

    def clear_eval_task(self) -> None:
        self.active_eval_task = None
        self._apply_eval_masks(None)
        if hasattr(self.model, "_tsn_active_action_dim"):
            self.model._tsn_active_action_dim = None

    def _zero_prev_task_param_grads(self) -> None:
        if not self.consolidated_masks:
            return
        name_to_module = dict(self.model.named_modules())
        for key, old_mask in self.consolidated_masks.items():
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
        if not self.freeze_non_mask_params_after_first or self.current_task_id == 0:
            return
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if name in self._maskable_param_names or name in self._score_param_names:
                continue
            param.grad.zero_()

    def _prepare_current_task(self) -> None:
        self.clear_eval_task()
        self._sync_occupied_masks_into_modules()
        if self.current_task_id > 0:
            self._reset_all_scores()

    def _quantize_new_weights_for_current_task(self, task_masks: Dict[str, Optional[torch.Tensor]]) -> Dict[str, np.ndarray]:
        codebooks: Dict[str, np.ndarray] = {}
        if not self.quantize_after_task:
            return codebooks

        name_to_module = dict(self.model.named_modules())
        for key, mask in task_masks.items():
            if mask is None or not key.endswith(".weight"):
                continue
            module_name, _ = key.rsplit(".", 1)
            module = name_to_module.get(module_name, None)
            if module is None:
                continue
            prev_mask = self.consolidated_masks.get(key, None)
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
            raise ValueError("TSNPandaStrategy.train_task got empty task_trajs")

        raw_obs_dim = infer_obs_dim(task_trajs)
        raw_act_dim = infer_act_dim(task_trajs)
        self._last_task_obs_dim = int(raw_obs_dim)
        self._last_task_action_dim = int(raw_act_dim)

        task_trajs_pad = prepare_panda_trajs(task_trajs, obs_dim=self.obs_dim, act_dim=self.act_dim)

        if self.store_task_obs_stats:
            mean_np, std_np = compute_obs_stats(task_trajs_pad, self.obs_dim)
            self._last_task_obs_stats = (mean_np.copy(), std_np.copy())
            if hasattr(self.model, "obs_mean") and hasattr(self.model, "obs_std"):
                with torch.no_grad():
                    self.model.obs_mean.copy_(torch.as_tensor(mean_np, dtype=self.model.obs_mean.dtype, device=self.model.obs_mean.device))
                    self.model.obs_std.copy_(torch.as_tensor(std_np, dtype=self.model.obs_std.dtype, device=self.model.obs_std.device))

        self._prepare_current_task()

        if active_action_dim_mask is None:
            active_action_dim_mask = [1] * raw_act_dim + [0] * max(0, self.act_dim - raw_act_dim)
        if len(active_action_dim_mask) != self.act_dim:
            raise ValueError(
                f"active_action_dim_mask has len={len(active_action_dim_mask)} but global act_dim={self.act_dim}"
            )
        raw_act_dim = int(sum(int(x) for x in active_action_dim_mask))
        self._last_task_action_dim = int(raw_act_dim)
        action_mask_t = torch.tensor(active_action_dim_mask, dtype=torch.float32, device=self.device)

        loader = make_panda_loader(
            task_trajs_pad,
            seq_len=self.seq_len,
            batch_size=batch_size,
            device=str(self.device),
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
        )

        self.model.train()
        last_loss = None
        for it in range(int(steps)):
            obs, actions, rtg, ts, mask = unpack_batch_continuous(next(loader))
            pred = self.model(obs, actions, rtg, ts, attention_mask=mask)
            loss = masked_mse(pred, actions, mask, action_dim_mask=action_mask_t)

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self._zero_prev_task_param_grads()
            self._zero_non_maskable_grads()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()

            last_loss = float(loss.detach().item())
            if it % 20000 == 0 or it == int(steps) - 1:
                print(
                    f"[tsn-panda] task={self.current_task_id} it={it} "
                    f"bc={last_loss:.6e} keep_ratio={self.keep_ratio:.2f} "
                    f"reuse={int(self.allow_weight_reuse)} quant={int(self.quantize_after_task)} "
                    f"active_action_dim_mask={active_action_dim_mask}"
                )

        return {
            "bc_loss": last_loss,
            "task_obs_dim": raw_obs_dim,
            "task_act_dim": raw_act_dim,
        }

    def after_task(self, task_trajs: List[Trajectory]):
        task_id = int(self.current_task_id)

        task_masks = self._collect_current_task_masks()
        self.per_task_masks[task_id] = task_masks
        self.task_codebooks[task_id] = self._quantize_new_weights_for_current_task(task_masks)
        self._update_consolidated_masks(task_masks)

        if self.store_task_obs_stats and self._last_task_obs_stats is not None:
            mean_np, std_np = self._last_task_obs_stats
            self.per_task_obs_stats[task_id] = (
                torch.as_tensor(mean_np, dtype=torch.float32).cpu(),
                torch.as_tensor(std_np, dtype=torch.float32).cpu(),
            )

        if self._last_task_action_dim is not None:
            self.per_task_action_dims[task_id] = int(self._last_task_action_dim)

        used = 0
        total = 0
        for key, mask in self.consolidated_masks.items():
            if mask is None or not key.endswith(".weight"):
                continue
            used += int(mask.sum().item())
            total += int(mask.numel())
        ratio = float(used / max(1, total))
        print(f"[tsn-panda] after task {task_id}: occupied_ratio={ratio:.4f}")

        self.set_eval_task(task_id)
        self.current_task_id += 1
