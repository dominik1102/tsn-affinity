
from __future__ import annotations
import numpy as np
from typing import List, Dict, Any

def acc_per_step(P: np.ndarray) -> List[float]:
    return [float(np.mean(P[i, :i+1])) for i in range(P.shape[0])]

def bwt_per_step(P: np.ndarray) -> List[float]:
    n = P.shape[0]; out = []
    for i in range(n):
        if i == 0: out.append(0.0)
        else:
            diffs = [P[i, j] - P[j, j] for j in range(i)]
            out.append(float(np.mean(diffs)) if diffs else 0.0)
    return out

def forgetting_per_step(P: np.ndarray) -> List[float]:
    n = P.shape[0]; out = []
    for i in range(n):
        if i == 0: out.append(0.0)
        else:
            vals = []
            for j in range(i):
                best = np.max(P[:i+1, j])
                vals.append(best - P[i, j])
            out.append(float(np.mean(vals)) if vals else 0.0)
    return out

def per_step_report(task_names: List[str], P: np.ndarray) -> List[Dict[str, Any]]:
    acc_i = acc_per_step(P); bwt_i = bwt_per_step(P); fgt_i = forgetting_per_step(P)
    rep = []
    for i, name in enumerate(task_names):
        rep.append({
            "step": i+1,
            "task_name": name,
            "acc_i": acc_i[i],
            "bwt_i": bwt_i[i],
            "forgetting_i": fgt_i[i],
            "diag": float(P[i, i]),
            "mean_seen": float(np.mean(P[i, :i+1])),
        })
    return rep

def per_step_detailed(task_names: List[str], P: np.ndarray) -> List[Dict[str, Any]]:
    n = P.shape[0]
    out: List[Dict[str, Any]] = []
    for i in range(n):
        per_task = []
        for j in range(i+1):
            current = float(P[i, j])
            diag_base = float(P[j, j])
            best_so_far = float(np.max(P[:i+1, j]))
            per_task.append({
                "task": task_names[j],
                "j": j,
                "current": current,
                "diag_baseline": diag_base,
                "delta_vs_diag": current - diag_base,
                "best_so_far": best_so_far,
                "forgetting": best_so_far - current,
            })
        out.append({"step": i+1, "task_name": task_names[i], "per_task": per_task})
    return out
