from __future__ import annotations
from typing import List

import numpy as np
import torch
import gymnasium as gym

from .dataset import Trajectory, discount_cumsum
from dt.dataset_panda import flatten_panda_step


# ==========================
#  Common helpers
# ==========================

def _flatten_obs_for_dt(obs):
    """
    Convert an environment observation into the flat vector expected by the
    Decision Transformer.

    - For Panda: obs is a dict -> flatten_panda_step(...)
    - For CartPole/Atari: obs is already a numpy array and is returned as-is.
    """
    if isinstance(obs, dict):
        return flatten_panda_step(obs)
    return obs


def _to_chw(arr: np.ndarray) -> np.ndarray:
    """
    Convert observations from HWC (or HW) to CHW format.

    - HWC  -> CHW
    - HW   -> [1, H, W]
    - CHW  -> returned unchanged
    """
    if arr is None:
        return arr
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[-1] in (1, 2, 3, 4, 12):  # HWC
        return np.transpose(a, (2, 0, 1))
    if a.ndim == 2:  # HW
        return a[None, ...]
    return a  # already CHW or something unexpected


# ==========================
#  Evaluation (generic DT)
# ==========================

@torch.no_grad()
def evaluate_dt(
    strategy,
    env,
    episodes: int,
    device: str,
    max_steps: int = 1000,
) -> float:
    """
    Generic evaluation for Decision Transformer policies.

    Works with:
      - discrete-action DT (CartPole / Atari; model.act returns an int),
      - continuous-action DT (Panda; model.act returns a float vector).

    Requirements:
      - strategy.model exposes .act(obs, rtg_scalar, t, prev_action, device)
    """
    rets = []
    model = strategy.model
    model.eval()

    for _ in range(episodes):
        obs, info = env.reset()
        total = 0.0
        prev_a = 0  # For continuous DT this can be ignored in .act()

        for t in range(max_steps):
            obs_in = _flatten_obs_for_dt(obs)

            if isinstance(env.action_space, gym.spaces.Discrete):
                n_actions = env.action_space.n
                a = model.act(
                    obs_in,
                    rtg_scalar=1.0,
                    t=t,
                    prev_action=prev_a,
                    device=device,
                    n_actions=n_actions,  # 👈
                )

                assert env.action_space.contains(a), \
                    f"Invalid action {a} for {env.action_space}"
            else:
                # np. Panda (continuous)
                a = model.act(
                    obs_in,
                    rtg_scalar=1.0,
                    t=t,
                    prev_action=prev_a,
                    device=device,
                )
            # For discrete envs, `a` is an int.
            # For Panda (continuous), `a` is a numpy vector.
            obs, r, done, truncated, _ = env.step(a)

            total += float(r)
            prev_a = a
            if done or truncated:
                break

        rets.append(total)

    return float(np.mean(rets)) if rets else 0.0


# Backwards-compatible alias for Panda-specific evaluation
@torch.no_grad()
def evaluate_dt_panda(strategy, env, episodes: int, device: str) -> float:
    """
    Convenience alias for evaluating Panda DT models.
    Simply calls evaluate_dt with a smaller default max_steps.
    """
    return evaluate_dt(strategy, env, episodes, device=device, max_steps=1000)


# ==========================
#  Trajectory collection (discrete envs)
# ==========================

def collect_trajectories(
    env,
    policy,
    n_episodes: int,
    max_len: int,
    target_return: float,
    device: str,
) -> List[Trajectory]:
    """
    Collect on-policy trajectories for discrete-action environments
    (e.g. CartPole, Atari).

    Observations can be images (HWC) or vectors; images are converted to CHW.
    The collected data is returned as a list of Trajectory objects.
    """
    trajs: List[Trajectory] = []

    for _ in range(n_episodes):
        o, _ = env.reset()
        o = _to_chw(o)
        obs, actions, rewards, timesteps = [], [], [], []
        prev_a = 0

        for t in range(max_len):
            obs.append(o)

            # 🔑 ważne: ograniczamy liczbę akcji do tego env
            a = policy.act(
                o,
                rtg_scalar=target_return,
                t=t,
                prev_action=prev_a,
                device=device,
                n_actions=env.action_space.n,
            )
            a = int(a)

            # opcjonalny safety-check (możesz usunąć po debugowaniu)
            if not env.action_space.contains(a):
                raise ValueError(
                    f"collect_trajectories: invalid action {a} for "
                    f"action_space={env.action_space}"
                )

            o, r, done, truncated, _ = env.step(a)
            o = _to_chw(o)

            actions.append(a)
            rewards.append(r)
            timesteps.append(t)
            prev_a = a

            if done or truncated:
                break

        # Stack observations into [T, C, H, W] (or [T, D] for vector obs)
        obs_arr = np.stack(obs).astype(np.float32)
        rtg = discount_cumsum(np.array(rewards, dtype=np.float32), gamma=1.0)

        trajs.append(
            Trajectory(
                obs=obs_arr,
                actions=np.array(actions, dtype=np.int64),
                rewards=np.array(rewards, dtype=np.float32),
                timesteps=np.array(timesteps, dtype=np.int64),
                returns_to_go=rtg,
            )
        )

    return trajs
