#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
export_minari_atari_expert_from_json.py

Reads a JSON task list like:

[
  {"name":"A_Pong","env_id":null,"seed":0,"params":{"game":"ALE/Pong-v5", ...}},
  ...
]

Then downloads Minari Atari expert datasets and exports DT-ready trajectories
in the same concatenated .npz format you used before DQN->DT:

Keys in each output .npz:
  - observations:     [sum_T, C, H, W]
  - actions:          [sum_T]
  - rewards:          [sum_T]
  - dones:            [sum_T]
  - episode_lengths:  [n_episodes]

Output structure (default):
resources/atari_expert/<TASK_NAME>/expert_minari_dqn.npz
resources/atari_expert/<TASK_NAME>/expert_minari_dqn_stats.json
resources/atari_expert/<TASK_NAME>/task_config.json
resources/atari_expert/manifest.json
resources/atari_expert/tasks_with_env_id.json

Preprocess:
  - Default: "dqn" (gray84 + stack4), CHW
  - Observations stored by default as float32 in [0,1]
    (you can switch to float16 or uint8 to reduce disk).

Notes:
  - Minari EpisodeData.observations is length T+1 (includes initial & final obs),
    we export obs[0:T] to align with actions/rewards/dones (length T).
"""

from __future__ import annotations

import argparse
import gc
import json
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# Image preprocessing
# =============================================================================

def _to_gray_uint8(frame_rgb: np.ndarray) -> np.ndarray:
    """HWC uint8 RGB -> HW uint8 grayscale (luma)."""
    if frame_rgb.ndim != 3 or frame_rgb.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB frame, got shape={frame_rgb.shape}")
    if frame_rgb.dtype != np.uint8:
        frame_rgb = frame_rgb.astype(np.uint8, copy=False)

    gray = (
        0.299 * frame_rgb[..., 0].astype(np.float32)
        + 0.587 * frame_rgb[..., 1].astype(np.float32)
        + 0.114 * frame_rgb[..., 2].astype(np.float32)
    )
    return np.clip(gray, 0.0, 255.0).astype(np.uint8)


def _resize_hw_uint8(img_hw: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """
    Resize a 2D uint8 image to out_hw using Pillow if available, else OpenCV, else nearest neighbor.
    out_hw is (H, W).
    """
    if img_hw.ndim != 2:
        raise ValueError(f"Expected HW image, got shape={img_hw.shape}")
    oh, ow = int(out_hw[0]), int(out_hw[1])

    # Pillow (preferred)
    try:
        from PIL import Image  # type: ignore
        pil = Image.fromarray(img_hw, mode="L")
        pil = pil.resize((ow, oh), resample=Image.BILINEAR)
        return np.asarray(pil, dtype=np.uint8)
    except Exception:
        pass

    # OpenCV fallback
    try:
        import cv2  # type: ignore
        return cv2.resize(img_hw, (ow, oh), interpolation=cv2.INTER_AREA).astype(np.uint8)
    except Exception:
        pass

    # Nearest-neighbor fallback (no extra deps)
    ys = (np.linspace(0, img_hw.shape[0] - 1, oh)).astype(np.int32)
    xs = (np.linspace(0, img_hw.shape[1] - 1, ow)).astype(np.int32)
    return img_hw[ys][:, xs].astype(np.uint8)


def _preprocess_frame_dqn_hw01(frame_rgb: np.ndarray, dqn_size: int = 84) -> np.ndarray:
    """One frame -> HW float32 in [0,1] (DQN style gray84)."""
    gray = _to_gray_uint8(frame_rgb)
    gray = _resize_hw_uint8(gray, (dqn_size, dqn_size))
    return gray.astype(np.float32) / 255.0  # HW


def _stack_frames_chw(frames_hw01: Sequence[np.ndarray]) -> np.ndarray:
    """Stack K frames HW -> CHW float32."""
    return np.stack(frames_hw01, axis=0).astype(np.float32)


def _convert_obs_dtype(x_chw01: np.ndarray, obs_dtype: str) -> np.ndarray:
    """
    Convert observation array to requested storage dtype:
      - float32: keep [0,1]
      - float16: keep [0,1]
      - uint8  : store [0,255] uint8 (still CHW)
    """
    obs_dtype = obs_dtype.lower()
    if obs_dtype == "float32":
        return x_chw01.astype(np.float32, copy=False)
    if obs_dtype == "float16":
        return x_chw01.astype(np.float16, copy=False)
    if obs_dtype == "uint8":
        x = np.clip(x_chw01 * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
        return x
    raise ValueError(f"Unsupported obs_dtype={obs_dtype}. Choose: float32, float16, uint8")


# =============================================================================
# Minari helpers
# =============================================================================

def derive_minari_dataset_id(task: Dict[str, Any], dataset_name: str = "expert-v0") -> str:
    """
    If task['env_id'] is not null -> use it as Minari dataset_id.
    Else derive from params['game'] like: "ALE/Pong-v5" -> "atari/pong/expert-v0".
    """
    env_id = task.get("env_id", None)
    if isinstance(env_id, str) and env_id.strip():
        return env_id.strip()

    params = task.get("params", {})
    game = params.get("game", None)
    if not isinstance(game, str):
        raise ValueError(f"Task {task.get('name')} has no params.game string to derive dataset_id.")

    m = re.match(r"^ALE/([A-Za-z0-9_]+)-v\d+$", game.strip())
    if not m:
        raise ValueError(
            f"params.game must look like 'ALE/Pong-v5', got: {game!r}. "
            "Either fix JSON or set env_id to Minari dataset_id explicitly."
        )

    game_name = m.group(1).lower()
    return f"atari/{game_name}/{dataset_name}"


def first_done_length(terminations: np.ndarray, truncations: np.ndarray, max_len: int) -> Tuple[int, bool]:
    """
    Given term/trunc arrays of length T, compute exported episode length L:
      - If any done within first max_len steps -> L = first_done_index + 1
      - Else L = min(T, max_len)
    Returns (L, hit_limit_without_done)
    """
    T = int(len(terminations))
    limit = min(T, int(max_len))

    done_mask = (terminations[:limit].astype(bool) | truncations[:limit].astype(bool))
    if done_mask.any():
        first_idx = int(np.argmax(done_mask))  # first True
        return first_idx + 1, False

    # no done before limit
    hit_limit = (limit >= int(max_len)) and (T >= int(max_len))
    return limit, hit_limit


@dataclass
class ExportStats:
    name: str
    dataset_id: str
    seed: int
    preprocess: str
    obs_dtype: str
    dqn_size: int
    frame_stack: int
    clip_rewards: bool
    n_episodes: int
    max_len: int
    total_steps_written: int
    episode_lengths: List[int]
    episode_returns: List[float]
    hit_max_len_without_done: int


# =============================================================================
# Core export: two-pass (plan lengths -> fill memmaps -> save npz)
# =============================================================================

def export_one_task_minari_atari(
    task: Dict[str, Any],
    out_root: Path,
    n_episodes: Optional[int],
    max_len: int,
    dataset_name: str,
    obs_dtype: str,
    dqn_size_default: int,
    preprocess_override: Optional[str] = None,
) -> Tuple[Path, Path, ExportStats]:
    """
    Exports a single task into resources structure.

    Returns:
      (npz_path, stats_json_path, stats)
    """
    try:
        import minari  # type: ignore
    except ImportError as e:
        raise ImportError("minari is required. Install with: pip install minari") from e

    name = str(task.get("name", "UNKNOWN"))
    seed = int(task.get("seed", 0))
    params = task.get("params", {}) if isinstance(task.get("params", {}), dict) else {}

    dataset_id = derive_minari_dataset_id(task, dataset_name=dataset_name)

    # Decide preprocess from JSON unless overridden.
    # We implement only the DQN-style path here because your JSON is for Atari preprocessing.
    preprocess = (preprocess_override or "dqn").lower()
    if preprocess != "dqn":
        raise ValueError("This exporter currently supports preprocess='dqn' only (gray84 + stack4).")

    frame_stack = int(params.get("frame_stack", 4))
    dqn_size = int(params.get("dqn_size", dqn_size_default))  # not in your JSON, default 84
    clip_rewards = bool(params.get("clip_rewards", True))

    # Load dataset (auto-download)
    dataset = minari.load_dataset(dataset_id, download=True)

    total_eps = int(getattr(dataset, "total_episodes"))
    use_eps = total_eps if n_episodes is None else min(int(n_episodes), total_eps)

    episode_indices = list(range(use_eps))  # deterministic order

    # Output dirs
    task_dir = out_root / name
    task_dir.mkdir(parents=True, exist_ok=True)

    # Save the task config snapshot (for reproducibility)
    (task_dir / "task_config.json").write_text(json.dumps(task, indent=2), encoding="utf-8")

    # -------- Pass 1: compute episode lengths (without decoding images) --------
    ep_lengths: List[int] = []
    hit_limit_count = 0

    for ep_i, episode in enumerate(dataset.iterate_episodes(episode_indices=episode_indices)):
        term = np.asarray(episode.terminations).reshape(-1)
        trunc = np.asarray(episode.truncations).reshape(-1)

        L, hit_limit = first_done_length(term, trunc, max_len=max_len)
        ep_lengths.append(int(L))
        if hit_limit:
            hit_limit_count += 1

        print(f"[plan] {name} ep {ep_i+1}/{use_eps} id={episode.id} planned_len={L} hit_limit={hit_limit}")

    sum_T = int(np.sum(ep_lengths))
    if sum_T <= 0:
        raise ValueError(f"{name}: sum_T computed as {sum_T}, nothing to export.")

    # Observation shape for DQN preprocess
    C, H, W = frame_stack, dqn_size, dqn_size

    # -------- Allocate memmaps (avoid huge RAM usage) --------
    tmp_dir = task_dir / ".tmp_memmap"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    obs_store_dtype = np.dtype(obs_dtype.lower())
    if obs_dtype.lower() not in ("float32", "float16", "uint8"):
        raise ValueError("--obs-dtype must be one of: float32, float16, uint8")

    # Use .npy memmaps (have headers) so numpy saves cleanly later
    obs_mm = np.lib.format.open_memmap(
        str(tmp_dir / "observations.npy"), mode="w+",
        dtype=obs_store_dtype, shape=(sum_T, C, H, W)
    )
    act_mm = np.lib.format.open_memmap(
        str(tmp_dir / "actions.npy"), mode="w+",
        dtype=np.int64, shape=(sum_T,)
    )
    rew_mm = np.lib.format.open_memmap(
        str(tmp_dir / "rewards.npy"), mode="w+",
        dtype=np.float32, shape=(sum_T,)
    )
    done_mm = np.lib.format.open_memmap(
        str(tmp_dir / "dones.npy"), mode="w+",
        dtype=np.bool_, shape=(sum_T,)
    )
    ep_len_arr = np.asarray(ep_lengths, dtype=np.int32)

    # -------- Pass 2: fill data --------
    write_pos = 0
    ep_returns: List[float] = []

    for ep_i, episode in enumerate(dataset.iterate_episodes(episode_indices=episode_indices)):
        L = ep_lengths[ep_i]

        obs_all = np.asarray(episode.observations)  # length T+1
        actions = np.asarray(episode.actions).reshape(-1)
        rewards = np.asarray(episode.rewards).reshape(-1)
        term = np.asarray(episode.terminations).reshape(-1)
        trunc = np.asarray(episode.truncations).reshape(-1)

        T = int(len(actions))
        if len(obs_all) != T + 1:
            raise ValueError(
                f"{name} episode {episode.id}: expected len(observations)=T+1, got {len(obs_all)} vs T={T}"
            )

        # Init frame stack with first obs
        dq: Deque[np.ndarray] = deque(maxlen=frame_stack)
        first_hw01 = _preprocess_frame_dqn_hw01(obs_all[0], dqn_size=dqn_size)
        for _ in range(frame_stack):
            dq.append(first_hw01.copy())

        ep_ret = 0.0

        for t in range(L):
            # obs CHW in [0,1]
            obs_chw01 = _stack_frames_chw(list(dq))  # float32 [C,H,W] in [0,1]
            obs_store = _convert_obs_dtype(obs_chw01, obs_dtype=obs_dtype)
            obs_mm[write_pos] = obs_store

            a = int(actions[t])
            r = float(rewards[t])
            if clip_rewards:
                r = float(np.clip(r, -1.0, 1.0))
            d = bool(term[t] or trunc[t])

            act_mm[write_pos] = a
            rew_mm[write_pos] = r
            done_mm[write_pos] = d

            ep_ret += r
            write_pos += 1

            # Update stack with next observation (t+1 exists because obs_all is T+1)
            next_hw01 = _preprocess_frame_dqn_hw01(obs_all[t + 1], dqn_size=dqn_size)
            dq.append(next_hw01)

        ep_returns.append(float(ep_ret))
        print(f"[fill] {name} ep {ep_i+1}/{use_eps} id={episode.id} return={ep_ret:.2f} L={L}")

    if write_pos != sum_T:
        raise RuntimeError(f"{name}: write_pos={write_pos} != planned sum_T={sum_T}")

    # Flush memmaps
    obs_mm.flush()
    act_mm.flush()
    rew_mm.flush()
    done_mm.flush()

    # -------- Save final NPZ in the exact concatenated format --------
    npz_path = task_dir / f"expert_minari_{preprocess}.npz"
    np.savez_compressed(
        str(npz_path),
        observations=np.asarray(obs_mm),
        actions=np.asarray(act_mm),
        rewards=np.asarray(rew_mm),
        dones=np.asarray(done_mm),
        episode_lengths=ep_len_arr,
    )
    print(f"[save] wrote: {npz_path}")

    # -------- Stats JSON (helpful for sanity checks) --------
    stats = ExportStats(
        name=name,
        dataset_id=dataset_id,
        seed=seed,
        preprocess=preprocess,
        obs_dtype=obs_dtype.lower(),
        dqn_size=dqn_size,
        frame_stack=frame_stack,
        clip_rewards=clip_rewards,
        n_episodes=use_eps,
        max_len=int(max_len),
        total_steps_written=int(sum_T),
        episode_lengths=[int(x) for x in ep_lengths],
        episode_returns=[float(x) for x in ep_returns],
        hit_max_len_without_done=int(hit_limit_count),
    )
    stats_json_path = task_dir / f"expert_minari_{preprocess}_stats.json"
    stats_json_path.write_text(json.dumps(stats.__dict__, indent=2), encoding="utf-8")
    print(f"[save] wrote: {stats_json_path}")

    # Cleanup temp memmaps (optional but default yes)
    del obs_mm, act_mm, rew_mm, done_mm
    gc.collect()

    # remove tmp directory content
    for p in tmp_dir.glob("*"):
        try:
            p.unlink()
        except Exception:
            pass
    try:
        tmp_dir.rmdir()
    except Exception:
        pass

    return npz_path, stats_json_path, stats


# =============================================================================
# Top-level: export from JSON into resources/atari_expert
# =============================================================================

def load_tasks_json(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Config JSON must be a list of task objects.")
    out: List[Dict[str, Any]] = []
    for i, x in enumerate(data):
        if not isinstance(x, dict):
            raise ValueError(f"Task #{i} must be an object/dict.")
        if "name" not in x:
            raise ValueError(f"Task #{i} missing 'name'.")
        out.append(x)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export Minari Atari expert trajectories from your JSON task list into resources/atari_expert/..."
    )
    ap.add_argument("--config", type=str, required=True, help="path to your tasks JSON (the one you pasted)")
    ap.add_argument(
        "--out-root", type=str, default="resources/atari_expert",
        help="output root directory (default: resources/atari_expert)"
    )
    ap.add_argument(
        "--dataset-name", type=str, default="expert-v0",
        help="Minari dataset name part (default: expert-v0)"
    )
    ap.add_argument(
        "--n-episodes", type=int, default=None,
        help="limit episodes per task (default: all episodes in dataset)"
    )
    ap.add_argument(
        "--max-len", type=int, default=50000,
        help="truncate episodes longer than this (default: 50000)"
    )
    ap.add_argument(
        "--obs-dtype", type=str, default="float32",
        choices=["float32", "float16", "uint8"],
        help="storage dtype for observations (float32 biggest; uint8 smallest)"
    )
    ap.add_argument("--dqn-size", type=int, default=84, help="resize for DQN gray84 (default 84)")
    ap.add_argument(
        "--preprocess", type=str, default="dqn",
        choices=["dqn"],
        help="currently only 'dqn' (gray84 + stack4) is supported"
    )
    args = ap.parse_args()

    config_path = Path(args.config)
    out_root = Path(args.out_root)

    tasks = load_tasks_json(config_path)

    out_root.mkdir(parents=True, exist_ok=True)
    # Save a copy of the original config next to outputs
    (out_root / "tasks_original.json").write_text(json.dumps(tasks, indent=2), encoding="utf-8")

    manifest: List[Dict[str, Any]] = []
    tasks_with_env_id: List[Dict[str, Any]] = []

    for task in tasks:
        name = str(task.get("name"))
        dataset_id = derive_minari_dataset_id(task, dataset_name=args.dataset_name)

        print("\n" + "=" * 90)
        print(f"[task] {name}  ->  {dataset_id}")
        print("=" * 90)

        npz_path, stats_path, stats = export_one_task_minari_atari(
            task=task,
            out_root=out_root,
            n_episodes=args.n_episodes,
            max_len=args.max_len,
            dataset_name=args.dataset_name,
            obs_dtype=args.obs_dtype,
            dqn_size_default=args.dqn_size,
            preprocess_override=args.preprocess,
        )

        manifest.append(
            {
                "name": name,
                "minari_dataset_id": dataset_id,
                "npz_path": str(npz_path.as_posix()),
                "stats_path": str(stats_path.as_posix()),
                "total_steps": stats.total_steps_written,
                "n_episodes": stats.n_episodes,
                "obs_dtype": stats.obs_dtype,
                "preprocess": stats.preprocess,
            }
        )

        # Write a "filled" version of the task json with env_id set to the exported npz path
        task_filled = dict(task)
        task_filled["env_id"] = str(npz_path.as_posix())
        task_filled["minari_dataset_id"] = dataset_id
        tasks_with_env_id.append(task_filled)

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_root / "tasks_with_env_id.json").write_text(json.dumps(tasks_with_env_id, indent=2), encoding="utf-8")

    print("\n[OK] Done.")
    print(f" - manifest: {out_root / 'manifest.json'}")
    print(f" - tasks_with_env_id: {out_root / 'tasks_with_env_id.json'}")


if __name__ == "__main__":
    main()
