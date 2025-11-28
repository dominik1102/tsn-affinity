
from __future__ import annotations
from typing import Dict, List, Any
import numpy as np
from ..core.task_spec import TaskSpec
from ..core.registry import TaskRegistry

def make_tasks(benchmark: str, specs: List[TaskSpec]) -> Dict[str, Any]:
    adapter = TaskRegistry.get(benchmark)
    return {s.name: adapter.create_env(s) for s in specs}

def describe_tasks(envs: Dict[str, Any], benchmark: str) -> str:
    adapter = TaskRegistry.get(benchmark)
    lines = []
    for name, env in envs.items():
        info = adapter.describe(env)
        parts = ", ".join(f"{k}={v}" for k, v in info.items())
        lines.append(f"{name}: {parts}")
    return "\n".join(lines)

class BenchmarkResults:
    def __init__(self, name: str, task_names: List[str], perf_matrix: np.ndarray, seed: int | None = None):
        self.name = name
        self.task_names = task_names
        self.perf_matrix = perf_matrix.astype(np.float32)
        self.seed = seed
