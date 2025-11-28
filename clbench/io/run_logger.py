
from __future__ import annotations
import os, json, csv, datetime
from typing import Dict, Any, List
import numpy as np

def timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def bench_short(bench: str) -> str:
    bench = (bench or "").lower()
    if bench.startswith("cartpole"): return "cp"
    if bench.startswith("atari"): return "at"
    return bench[:3] or "run"

def build_run_dir(root: str, benchmark: str, strategy: str | None, tag: str = "") -> str:
    ts = timestamp()
    leaf = strategy if strategy else "baseline"
    safe_tag = (tag or "").replace("/", "_").replace(" ", "_")
    run_dir = os.path.join(root, ts, benchmark, leaf, safe_tag)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "gen"), exist_ok=True)
    return run_dir

def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def save_matrix_csv(path: str, task_names: List[str], P: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["after_task \\ on_task"] + task_names)
        for i, row in enumerate(P):
            w.writerow([task_names[i]] + [f"{v:.6f}" for v in row])

def save_task_gen_json(run_dir: str, i: int, payload: Dict[str, Any]) -> str:
    path = os.path.join(run_dir, "gen", f"task_{i}.json")
    save_json(path, payload)
    return path
