
from __future__ import annotations
from typing import List
import json
from ..core.task_spec import TaskSpec

def save_task_specs(path: str, specs: List[TaskSpec]):
    payload = [dict(name=s.name, env_id=s.env_id, seed=s.seed, params=s.params) for s in specs]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def load_task_specs(path: str) -> List[TaskSpec]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [TaskSpec(**d) for d in data]
