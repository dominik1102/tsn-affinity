from __future__ import annotations

from typing import Any, Tuple, Optional

import numpy as np
import torch
import gymnasium as gym


def _env_reset(env) -> Tuple[Any, dict]:
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
    return out, {}


def _env_step(env, action):
    out = env.step(action)
    # gymnasium: (obs, reward, terminated, truncated, info)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, float(reward), bool(terminated), bool(truncated), info
    # old gym: (obs, reward, done, info)
    obs, reward, done, info = out
    return obs, float(reward), bool(done), False, info


def _get_fire_action(env) -> Optional[int]:
    """Return index of FIRE action if present, else None."""
    try:
        if not isinstance(env.action_space, gym.spaces.Discrete):
            return None
        meanings = env.unwrapped.get_action_meanings()
        if isinstance(meanings, (list, tuple)) and "FIRE" in meanings:
            return int(meanings.index("FIRE"))
    except Exception:
        return None
    return None


@torch.no_grad()
def evaluate_dt(
    strategy,
    env,
    episodes: int,
    device: str,
    max_steps: int = 1000,
    target_return: float = 1.0,
    fire_reset: bool = False,
) -> float:
    """
    Evaluation for Decision Transformer.

    - start RTG = target_return
    - each step: rtg_remaining -= reward

    If fire_reset=True and FIRE exists:
      we execute FIRE as first action (t=0) and PRIME DT history (obs0, FIRE, rtg0, t=0).
    """
    model = strategy.model
    model.eval()

    returns = []

    n_actions = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else None
    fire_action = _get_fire_action(env) if fire_reset else None

    for _ in range(int(episodes)):
        obs, _info = _env_reset(env)
        model.reset_history()

        ep_ret = 0.0
        rtg_remaining = float(target_return)
        prev_a = 0
        t0 = 0

        if fire_action is not None:
            if hasattr(model, "prime_history"):
                model.prime_history(obs, action=fire_action, rtg_scalar=rtg_remaining, t=0)

            obs, r0, done, truncated, _ = _env_step(env, fire_action)
            ep_ret += float(r0)
            rtg_remaining -= float(r0)
            prev_a = fire_action
            t0 = 1

            if done or truncated:
                returns.append(ep_ret)
                continue

        for t in range(t0, int(max_steps)):
            if isinstance(env.action_space, gym.spaces.Discrete):
                a = model.act(
                    obs,
                    rtg_scalar=rtg_remaining,
                    t=t,
                    prev_action=prev_a,
                    device=device,
                    n_actions=n_actions,
                )
                a = int(a)
                assert env.action_space.contains(a), f"Invalid action {a} for {env.action_space}"
            else:
                a = model.act(
                    obs,
                    rtg_scalar=rtg_remaining,
                    t=t,
                    prev_action=prev_a,
                    device=device,
                )

            obs, r, done, truncated, _ = _env_step(env, a)
            ep_ret += float(r)
            rtg_remaining -= float(r)
            prev_a = a

            if done or truncated:
                break

        returns.append(ep_ret)

    return float(np.mean(returns)) if returns else 0.0


def replay_actions(env, actions: np.ndarray, max_steps: Optional[int] = None) -> float:
    """
    Debug: reset env and execute the given action sequence.
    This lets you verify env<->dataset match.
    """
    _obs, _ = _env_reset(env)
    ep_ret = 0.0
    steps = int(len(actions) if max_steps is None else min(len(actions), max_steps))

    for i in range(steps):
        a = int(actions[i])
        _obs, r, done, truncated, _ = _env_step(env, a)
        ep_ret += float(r)
        if done or truncated:
            break

    return float(ep_ret)
