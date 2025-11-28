
from __future__ import annotations
from typing import Dict
from .env_adapter import EnvAdapter

class TaskRegistry:
    _adapters: Dict[str, EnvAdapter] = {}
    @classmethod
    def register(cls, name: str, adapter: EnvAdapter):
        cls._adapters[name] = adapter
    @classmethod
    def get(cls, name: str) -> EnvAdapter:
        if name not in cls._adapters:
            raise KeyError(f"No adapter registered for benchmark '{name}'")
        return cls._adapters[name]
