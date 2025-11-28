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
        strategy.model.reset_history()
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


import numpy as np
import torch


@torch.no_grad()
def evaluate_dt_panda(strategy, env, episodes: int, device: str, max_steps: int = 1000):
    """
    Evaluate a PandaDecisionTransformer-based strategy on a Panda environment.

    The function:
      - resets the model's internal history at the start of each episode,
      - flattens observations using _flatten_panda_obs so that they match
        the offline dataset representation,
      - uses model.act(...) to select continuous actions,
      - returns the average return across `episodes` episodes.
    """
    model = strategy.model
    returns: list[float] = []

    for ep in range(episodes):
        # Important: clear the DT history at the start of every episode
        if hasattr(model, "reset_history"):
            model.reset_history()

        obs, _ = env.reset()
        total_reward = 0.0
        prev_action = None

        for t in range(max_steps):
            # Flatten observation to the same vector used in the offline dataset
            obs_vec = _flatten_panda_obs(
                obs,
                expected_dim=getattr(model, "obs_dim", None),
            )

            # Select action using the PandaDecisionTransformer
            action = model.act(
                obs_vec,
                rtg_scalar=1.0,
                t=t,
                prev_action=prev_action,
                device=device,
            )

            action = np.asarray(action, dtype=np.float32).ravel()
            act_dim_env = int(np.prod(env.action_space.shape))

            if action.shape[0] < act_dim_env:
                # Pad with zeros if the model outputs fewer dimensions than the env expects
                pad = np.zeros(act_dim_env - action.shape[0], dtype=np.float32)
                action = np.concatenate([action, pad], axis=0)
            elif action.shape[0] > act_dim_env:
                # Truncate if the model outputs more than the env expects
                action = action[:act_dim_env]

            obs, reward, done, truncated, _ = env.step(action)
            total_reward += float(reward)
            prev_action = action

            if done or truncated:
                break

        returns.append(total_reward)

    return float(np.mean(returns)) if returns else 0.0


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


def _flatten_panda_obs(obs, expected_dim: int | None = None) -> np.ndarray:
    """
    Convert a Panda environment observation to a flat vector compatible
    with PandaDecisionTransformer.

    Typical patterns:
      - Goal-based envs: obs is a dict with keys like "observation",
        "achieved_goal", "desired_goal".
      - Non-goal envs: obs is already a flat array/tensor.

    This function:
      1) flattens the observation to 1D,
      2) if expected_dim is not None:
           - pads with zeros if dim < expected_dim,
           - truncates if dim > expected_dim.
    """

    # If it's a torch.Tensor, move to CPU/NumPy first
    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu().numpy()

    # Dict case (e.g., Panda goal envs)
    if isinstance(obs, dict):
        if "observation" in obs and "desired_goal" in obs:
            base = np.asarray(obs["observation"], dtype=np.float32).ravel()
            goal = np.asarray(obs["desired_goal"], dtype=np.float32).ravel()
            flat = np.concatenate([base, goal], axis=0)
        elif "observation" in obs:
            # Only 'observation' is available
            flat = np.asarray(obs["observation"], dtype=np.float32).ravel()
        else:
            # Generic fallback: concatenate all values in sorted key order
            parts = [
                np.asarray(v, dtype=np.float32).ravel()
                for k, v in sorted(obs.items())
            ]
            flat = np.concatenate(parts, axis=0)
    else:
        # Already a plain vector / array
        flat = np.asarray(obs, dtype=np.float32).ravel()

    # Adjust dimension to match what the model expects
    if expected_dim is not None:
        d = flat.shape[-1]
        if d < expected_dim:
            pad = np.zeros(expected_dim - d, dtype=np.float32)
            flat = np.concatenate([flat, pad], axis=0)
        elif d > expected_dim:
            flat = flat[:expected_dim]

    return flat
