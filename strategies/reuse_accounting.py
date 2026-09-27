from __future__ import annotations

from typing import Any, Dict, Optional

import torch

MaskDict = Dict[str, Optional[torch.Tensor]]


def _to_bool_cpu(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Detach a mask and move it to CPU bool without modifying model state."""
    if mask is None:
        return None
    return mask.detach().cpu().bool()


def compute_mask_reuse_stats(
    *,
    task_id: int,
    copy_id: int,
    current_task_masks: MaskDict,
    previous_task_masks: Dict[int, MaskDict],
    source_task: Optional[int] = None,
    weight_only: bool = True,
) -> Dict[str, Any]:
    """Compute read-only overlap between current and previous task masks.

    This function does not touch a model, does not call forward(), does not set
    active/eval masks, and does not mutate any mask tensor. It is intended for
    post-hoc accounting after all task training/evaluation has already finished.

    Definitions:
      selected_weights: number of active parameters in the current task mask M_t.
      reused_from_any_previous: active parameters in M_t that were already active
        in at least one previous task mask in the same model copy.
      reuse_ratio_any: reused_from_any_previous / selected_weights.
      new_weights: active parameters in M_t that were not active in any previous
        task mask in the same model copy.
      source_coverage: source_overlap / |M_source|.
      source_ratio_in_current: source_overlap / |M_t|.

    Important: for multi-copy methods, pass previous_task_masks only from the
    same copy. Overlap across different copies is not true parameter reuse.
    """
    task_id = int(task_id)
    copy_id = int(copy_id)

    selected_total = 0
    reused_any_total = 0
    new_total = 0

    per_prev_overlap: Dict[int, int] = {int(t): 0 for t in previous_task_masks.keys()}
    per_prev_size: Dict[int, int] = {int(t): 0 for t in previous_task_masks.keys()}
    layer_stats: Dict[str, Dict[str, Any]] = {}

    for prev_t, prev_masks in previous_task_masks.items():
        prev_t = int(prev_t)
        size = 0
        for key, old_mask in prev_masks.items():
            if old_mask is None:
                continue
            if weight_only and not key.endswith(".weight"):
                continue
            old_b = _to_bool_cpu(old_mask)
            if old_b is not None:
                size += int(old_b.sum().item())
        per_prev_size[prev_t] = int(size)

    for key, cur_mask in current_task_masks.items():
        if cur_mask is None:
            continue
        if weight_only and not key.endswith(".weight"):
            continue

        cur_b = _to_bool_cpu(cur_mask)
        if cur_b is None:
            continue

        selected = int(cur_b.sum().item())
        if selected == 0:
            continue

        prev_union = torch.zeros_like(cur_b, dtype=torch.bool)
        per_layer_prev: Dict[int, int] = {}

        for prev_t, prev_masks in previous_task_masks.items():
            prev_t = int(prev_t)
            old_mask = prev_masks.get(key, None)
            old_b = _to_bool_cpu(old_mask)
            if old_b is None or tuple(old_b.shape) != tuple(cur_b.shape):
                per_layer_prev[prev_t] = 0
                continue

            overlap = int(torch.logical_and(cur_b, old_b).sum().item())
            per_prev_overlap[prev_t] = int(per_prev_overlap.get(prev_t, 0) + overlap)
            per_layer_prev[prev_t] = overlap
            prev_union = torch.logical_or(prev_union, old_b)

        reused_any = int(torch.logical_and(cur_b, prev_union).sum().item())
        new_selected = int(selected - reused_any)

        selected_total += selected
        reused_any_total += reused_any
        new_total += new_selected

        layer_stats[key] = {
            "selected": int(selected),
            "reused_from_any_previous": int(reused_any),
            "new": int(new_selected),
            "reuse_ratio_any": float(reused_any / max(1, selected)),
            "per_prev_overlap": {str(k): int(v) for k, v in per_layer_prev.items()},
        }

    best_prev_task = None
    best_prev_overlap = 0
    for prev_t, overlap in per_prev_overlap.items():
        if int(overlap) > int(best_prev_overlap):
            best_prev_task = int(prev_t)
            best_prev_overlap = int(overlap)

    source_overlap = None
    source_mask_size = None
    source_coverage = None
    source_ratio_in_current = None

    if source_task is not None:
        source_task = int(source_task)
        source_overlap = int(per_prev_overlap.get(source_task, 0))
        source_mask_size = int(per_prev_size.get(source_task, 0))
        source_coverage = float(source_overlap / max(1, source_mask_size))
        source_ratio_in_current = float(source_overlap / max(1, selected_total))

    return {
        "task_id": int(task_id),
        "copy_id": int(copy_id),
        "source_task": None if source_task is None else int(source_task),
        "selected_weights": int(selected_total),
        "reused_from_any_previous": int(reused_any_total),
        "new_weights": int(new_total),
        "reuse_ratio_any": float(reused_any_total / max(1, selected_total)),
        "new_ratio": float(new_total / max(1, selected_total)),
        "best_prev_task_by_overlap": best_prev_task,
        "best_prev_overlap": int(best_prev_overlap),
        "best_prev_ratio_in_current": float(best_prev_overlap / max(1, selected_total)),
        "source_overlap": source_overlap,
        "source_mask_size": source_mask_size,
        "source_coverage": source_coverage,
        "source_ratio_in_current": source_ratio_in_current,
        "per_prev_overlap": {str(k): int(v) for k, v in per_prev_overlap.items()},
        "per_prev_size": {str(k): int(v) for k, v in per_prev_size.items()},
        "layers": layer_stats,
    }


def short_reuse_line(prefix: str, task_id: str | int, stats: Dict[str, Any]) -> str:
    def fmt(x: Any) -> str:
        if x is None:
            return "None"
        if isinstance(x, float):
            return f"{x:.4f}"
        return str(x)

    return (
        f"{prefix} task={task_id} copy={stats.get('copy_id')} "
        f"src={stats.get('source_task')} selected={stats.get('selected_weights')} "
        f"reuse_any={fmt(stats.get('reuse_ratio_any'))} "
        f"new={fmt(stats.get('new_ratio'))} "
        f"src_cov={fmt(stats.get('source_coverage'))} "
        f"src_in_cur={fmt(stats.get('source_ratio_in_current'))} "
        f"best_prev={stats.get('best_prev_task_by_overlap')} "
        f"best_prev_in_cur={fmt(stats.get('best_prev_ratio_in_current'))}"
    )

def print_mask_reuse_stats(prefix: str, stats: Dict[str, Any]) -> None:
    """Print one compact, stable reuse-accounting line.

    This wrapper exists because TSN strategy files call print_mask_reuse_stats(),
    while older reuse_accounting.py versions exposed only short_reuse_line().
    It is intentionally read-only: it formats the already computed stats and
    does not touch any model state or mask tensors.
    """
    task_id = stats.get("task_id", "?")
    print(short_reuse_line(str(prefix), task_id, stats), flush=True)