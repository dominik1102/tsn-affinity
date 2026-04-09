from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from dt.dataset import Trajectory, make_minibatches
from .base import BaseStrategy
from .tsn_common import (
    TSNConversionConfig,
    _convert_module_inplace,
    _iter_tsn_modules,
    _kmeans_quantize_selected_,
    _rebuild_optimizer_like,
)
from .utils import _unpack_batch


class TSNStrategy(BaseStrategy):
    """
    TSN-core Atari strategy:
      - trainable score masks on Conv / Linear / optional Embedding layers,
      - one stored binary mask per task,
      - strict freeze of old-task weights,
      - optional KMeans quantization after task,
      - optional equal-share keep-ratio schedule across known tasks.
    """

    def __init__(
        self,
        *args,
        keep_ratio: float = 0.5,
        include_embeddings: bool = True,
        quantize_after_task: bool = True,
        quant_clusters: int = 16,
        allow_weight_reuse: bool = False,
        freeze_non_mask_params_after_first: bool = True,
        skip_module_names: Tuple[str, ...] = ("dt.te",),
        expected_num_tasks: Optional[int] = None,
        keep_ratio_schedule: str = "constant",
        min_keep_ratio: float = 1e-3,
        grad_clip: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        cfg = TSNConversionConfig(
            keep_ratio=float(keep_ratio),
            include_embeddings=bool(include_embeddings),
            allow_weight_reuse=bool(allow_weight_reuse),
            skip_module_names=tuple(skip_module_names),
        )
        _convert_module_inplace(self.model, cfg)
        self.model.to(self.device)
        self.opt = _rebuild_optimizer_like(self.opt, self.model.parameters())

        self.base_keep_ratio = float(keep_ratio)
        self.current_keep_ratio = float(keep_ratio)
        self.include_embeddings = bool(include_embeddings)
        self.quantize_after_task = bool(quantize_after_task)
        self.quant_clusters = int(quant_clusters)
        self.allow_weight_reuse = bool(allow_weight_reuse)
        self.freeze_non_mask_params_after_first = bool(freeze_non_mask_params_after_first)
        self.skip_module_names = tuple(skip_module_names)
        self.expected_num_tasks = None if expected_num_tasks is None else int(expected_num_tasks)
        self.keep_ratio_schedule = str(keep_ratio_schedule)
        self.min_keep_ratio = float(min_keep_ratio)
        self.grad_clip = float(grad_clip)

        self.current_task_id: int = 0
        self.per_task_masks: Dict[int, Dict[str, Optional[torch.Tensor]]] = {}
        self.consolidated_masks: Dict[str, Optional[torch.Tensor]] = {}
        self.task_codebooks: Dict[int, Dict[str, np.ndarray]] = {}
        self.active_eval_task: Optional[int] = None
        self.task_keep_ratios: Dict[int, float] = {}

        self._refresh_name_sets()
        self.clear_eval_task()

    def set_expected_num_tasks(self, n_tasks: int) -> None:
        self.expected_num_tasks = int(n_tasks)

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
                self.consolidated_masks[key] = torch.logical_or(
                    self.consolidated_masks[key].bool(),
                    mask.bool(),
                ).to(torch.uint8)

    def _apply_eval_masks(self, task_id: Optional[int]) -> None:
        task_masks = self.per_task_masks.get(task_id, None) if task_id is not None else None
        for name, mod in _iter_tsn_modules(self.model):
            if task_masks is None:
                mod.clear_active_masks()
                continue
            mod.active_weight_mask = task_masks.get(f"{name}.weight", None)
            mod.active_bias_mask = task_masks.get(f"{name}.bias", None)

    def has_task_mask(self, task_id: int) -> bool:
        return int(task_id) in self.per_task_masks

    def set_eval_task(self, task_id: int) -> None:
        task_id = int(task_id)
        self.active_eval_task = task_id if task_id in self.per_task_masks else None
        self._apply_eval_masks(self.active_eval_task)

    def clear_eval_task(self) -> None:
        self.active_eval_task = None
        self._apply_eval_masks(None)

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

    def _snapshot_frozen_params(self) -> Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        snap: Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]] = {}
        if self.current_task_id == 0:
            return snap

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            old_mask = self.consolidated_masks.get(name, None)
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

        if self.current_task_id > 0:
            self._reset_all_scores()
            self.opt = _rebuild_optimizer_like(self.opt, self.model.parameters())

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
    ):
        self._prepare_current_task()
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
                print(
                    f"[tsn] task={self.current_task_id} it={it} "
                    f"ce={last_loss:.6e} keep_ratio={self.current_keep_ratio:.4f} "
                    f"schedule={self.keep_ratio_schedule} "
                    f"reuse={int(self.allow_weight_reuse)} quant={int(self.quantize_after_task)}"
                )

        return {"loss": last_loss, "keep_ratio": float(self.current_keep_ratio)}

    def after_task(self, task_trajs: List[Trajectory]):
        task_id = int(self.current_task_id)

        task_masks = self._collect_current_task_masks()
        self.per_task_masks[task_id] = task_masks
        self.task_codebooks[task_id] = self._quantize_new_weights_for_current_task(task_masks)
        self._update_consolidated_masks(task_masks)

        used = 0
        total = 0
        for key, mask in self.consolidated_masks.items():
            if mask is None or not key.endswith(".weight"):
                continue
            used += int(mask.sum().item())
            total += int(mask.numel())
        ratio = float(used / max(1, total))
        print(
            f"[tsn] after task {task_id}: "
            f"occupied_ratio={ratio:.4f} keep_ratio={self.task_keep_ratios.get(task_id, self.current_keep_ratio):.4f}"
        )

        self.set_eval_task(task_id)
        self.current_task_id += 1
