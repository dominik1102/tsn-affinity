
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class TaskSpec:
    name: str
    env_id: Optional[str] = None
    seed: Optional[int] = None
    params: Optional[Dict[str, Any]] = None
