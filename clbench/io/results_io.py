
from __future__ import annotations
from typing import List, Dict, Any
import json, csv, numpy as np

def save_perf_json(path: str, name: str, task_names: List[str], perf_matrix: np.ndarray):
    payload = { "name": name, "task_names": task_names, "perf_matrix": perf_matrix.tolist() }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def save_metrics_json(path: str, metrics: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

def save_perf_csv(path: str, task_names: List[str], perf_matrix: np.ndarray):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["after_task \ on_task"] + task_names)
        for i, row in enumerate(perf_matrix):
            w.writerow([task_names[i]] + [f"{v:.6f}" for v in row])

def save_metrics_csv(path: str, metrics: Dict[str, Any]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["metric", "value"])
        for k, v in metrics.items(): w.writerow([k, v])
