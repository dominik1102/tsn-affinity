#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from collections import deque
from typing import Iterable, List, Dict, Any, Tuple

import numpy as np

# Lazy imports so the script can be inspected without TF installed.
import tensorflow as tf  # type: ignore
import tensorflow_datasets as tfds  # type: ignore


def _stack_episode_frames(frames: np.ndarray, stack_size: int = 4) -> np.ndarray:
    """Convert [T,H,W,1] uint8 frames into [T,stack,H,W] float32 in [0,1].

    Uses standard Atari-style frame stacking with repetition of the first frame
    to fill the initial context.
    """
    if frames.ndim != 4 or frames.shape[-1] != 1:
        raise ValueError(f"Expected frames [T,H,W,1], got {frames.shape}")

    T, H, W, _ = frames.shape
    q: deque[np.ndarray] = deque(maxlen=stack_size)
    first = frames[0, :, :, 0]
    for _ in range(stack_size):
        q.append(first)

    out = np.empty((T, stack_size, H, W), dtype=np.float32)
    for t in range(T):
        cur = frames[t, :, :, 0]
        q.append(cur)
        out[t] = np.stack(list(q), axis=0).astype(np.float32) / 255.0
    return out


def _episode_steps_to_arrays(episode: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Handle common TFDS/RLDS episode layouts.

    Returns:
        observations [T,H,W,1] uint8,
        actions [T] int64,
        rewards [T] float32,
        is_last [T] bool
    """
    steps = episode["steps"]

    # Case 1: tfds.as_numpy already materialized the step fields as arrays.
    if isinstance(steps, dict):
        obs = np.asarray(steps["observation"])
        act = np.asarray(steps["action"])
        rew = np.asarray(steps["reward"], dtype=np.float32)
        is_last = np.asarray(steps["is_last"], dtype=bool)
        return obs, act, rew, is_last

    # Case 2: steps is an iterable of per-step dicts.
    obs_list: List[np.ndarray] = []
    act_list: List[int] = []
    rew_list: List[float] = []
    last_list: List[bool] = []
    for step in steps:
        obs_list.append(np.asarray(step["observation"]))
        act_list.append(int(step["action"]))
        rew_list.append(float(step["reward"]))
        last_list.append(bool(step["is_last"]))
    obs = np.stack(obs_list, axis=0)
    act = np.asarray(act_list, dtype=np.int64)
    rew = np.asarray(rew_list, dtype=np.float32)
    is_last = np.asarray(last_list, dtype=bool)
    return obs, act, rew, is_last


def convert_one_config(
    config_name: str,
    output_npz: str,
    *,
    data_dir: str | None,
    max_episodes: int | None,
    stack_size: int,
) -> Dict[str, Any]:
    ds = tfds.load(config_name, split="train", data_dir=data_dir, shuffle_files=False)
    ds_np = tfds.as_numpy(ds)

    obs_eps: List[np.ndarray] = []
    act_eps: List[np.ndarray] = []
    rew_eps: List[np.ndarray] = []
    lengths: List[int] = []
    returns: List[float] = []
    checkpoint_ids: List[int] = []
    episode_ids: List[int] = []

    count = 0
    for ep in ds_np:
        if max_episodes is not None and count >= max_episodes:
            break

        obs_raw, act, rew, is_last = _episode_steps_to_arrays(ep)
        if obs_raw.shape[0] == 0:
            continue

        # Keep only real decision steps; RLDS episodes already respect boundaries.
        T = int(act.shape[0])
        obs_stacked = _stack_episode_frames(obs_raw, stack_size=stack_size)
        dones = np.zeros((T,), dtype=bool)
        if T > 0:
            dones[-1] = True
            # Prefer the dataset boundary if present.
            if is_last.shape[0] == T:
                dones[:] = False
                dones[np.where(is_last)[0]] = True
                if not dones.any():
                    dones[-1] = True

        obs_eps.append(obs_stacked)
        act_eps.append(act.astype(np.int64))
        rew_eps.append(rew.astype(np.float32))
        lengths.append(T)
        returns.append(float(np.sum(rew)))
        checkpoint_ids.append(int(ep.get("checkpoint_id", -1)))
        episode_ids.append(int(ep.get("episode_id", count)))
        count += 1

    if not obs_eps:
        raise RuntimeError(f"No episodes found in config {config_name}")

    observations = np.concatenate(obs_eps, axis=0).astype(np.float32)
    actions = np.concatenate(act_eps, axis=0).astype(np.int64)
    rewards = np.concatenate(rew_eps, axis=0).astype(np.float32)
    dones = np.zeros_like(rewards, dtype=bool)
    idx = 0
    for L in lengths:
        dones[idx + L - 1] = True
        idx += L

    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    np.savez_compressed(
        output_npz,
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
        episode_lengths=np.asarray(lengths, dtype=np.int64),
        episode_returns=np.asarray(returns, dtype=np.float32),
        checkpoint_ids=np.asarray(checkpoint_ids, dtype=np.int64),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        source_config=np.asarray(config_name),
    )

    return {
        "config": config_name,
        "episodes": len(lengths),
        "transitions": int(observations.shape[0]),
        "avg_return": float(np.mean(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "output": output_npz,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert RL Unplugged Atari TFDS episodes to the npz format used by this codebase.")
    ap.add_argument("--games", nargs="+", required=True, help="Games to convert, e.g. Breakout Alien Atlantis Boxing Centipede")
    ap.add_argument("--runs", nargs="+", type=int, default=[1], help="TFDS run IDs, e.g. 1 2")
    ap.add_argument("--dataset", choices=["rlu_atari", "rlu_atari_checkpoints_ordered"], default="rlu_atari")
    ap.add_argument("--output-root", required=True, help="Where to write converted .npz files")
    ap.add_argument("--data-dir", default=None, help="Optional TFDS cache/download directory")
    ap.add_argument("--max-episodes", type=int, default=None, help="Optional cap per config for quick experiments")
    ap.add_argument("--stack-size", type=int, default=4)
    args = ap.parse_args()

    summaries: List[Dict[str, Any]] = []
    for game in args.games:
        for run_id in args.runs:
            config_name = f"{args.dataset}/{game}_run_{run_id}"
            out_dir = os.path.join(args.output_root, game)
            out_npz = os.path.join(out_dir, f"{args.dataset}_{game}_run_{run_id}.npz")
            print(f"[convert] {config_name} -> {out_npz}")
            stats = convert_one_config(
                config_name,
                out_npz,
                data_dir=args.data_dir,
                max_episodes=args.max_episodes,
                stack_size=args.stack_size,
            )
            summaries.append(stats)
            print(
                f"[done] {game} run={run_id} episodes={stats['episodes']} transitions={stats['transitions']} "
                f"avg_return={stats['avg_return']:.3f} min={stats['min_return']:.3f} max={stats['max_return']:.3f}"
            )

    print("\n=== Summary ===")
    for s in summaries:
        print(s)


if __name__ == "__main__":
    main()
