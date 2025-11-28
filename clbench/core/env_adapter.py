
from __future__ import annotations
from typing import Protocol, Dict, Any

class EnvAdapter(Protocol):
    def create_env(self, spec) -> Any: ...
    def describe(self, env) -> Dict[str, Any]: ...
