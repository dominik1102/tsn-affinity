
from __future__ import annotations
import warnings
from typing import Any, Dict
from ..core.task_spec import TaskSpec

try:
    import gymnasium as gym
except Exception:
    import gym  # type: ignore

class CartPoleAdapter:
    ENV_ID = "CartPole-v1"
    TUNEABLES = ["gravity", "length", "masscart", "masspole", "force_mag", "tau"]

    def create_env(self, spec: TaskSpec):
        env = gym.make(spec.env_id or self.ENV_ID)
        try:
            env.reset(seed=spec.seed)
        except TypeError:
            if spec.seed is not None:
                env.seed(spec.seed)
        params = spec.params or {}
        un = getattr(env, "unwrapped", env)
        for k, v in params.items():
            if k in self.TUNEABLES and hasattr(un, k):
                setattr(un, k, v)
            elif k in self.TUNEABLES:
                warnings.warn(f"CartPole has no attribute '{k}', cannot set {k}={v}")
        return env

    def describe(self, env) -> Dict[str, Any]:
        un = getattr(env, "unwrapped", env)
        return {k: getattr(un, k) for k in self.TUNEABLES if hasattr(un, k)}
