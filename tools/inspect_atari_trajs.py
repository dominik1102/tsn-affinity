#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Low-memory NPZ metadata / sampling (avoid loading huge observations arrays)
# =============================================================================

def read_npy_header_from_npz(npz_path: str, key: str) -> Tuple[Tuple[int, ...], np.dtype, bool, Tuple[int, int]]:
    """
    Read shape/dtype of <key>.npy stored inside a .npz WITHOUT loading the full array.

    Returns: (shape, dtype, fortran_order, version)
    """
    import numpy.lib.format as fmt

    member = f"{key}.npy"
    with zipfile.ZipFile(npz_path, "r") as zf:
        if member not in zf.namelist():
            raise KeyError(f"Key '{key}' not found in npz: {npz_path}. Available: {zf.namelist()}")
        with zf.open(member, "r") as f:
            version = fmt.read_magic(f)
            shape, fortran_order, dtype = fmt._read_array_header(f, version)
    return tuple(shape), dtype, bool(fortran_order), tuple(version)


def sample_prefix_from_observations_npz(
    npz_path: str,
    n_obs: int = 4,
) -> Optional[np.ndarray]:
    """
    Read ONLY the first `n_obs` observations from observations.npy inside the npz,
    without loading the full array.

    Returns array of shape [n_obs, C, H, W] (or None if missing).
    """
    import numpy.lib.format as fmt

    member = "observations.npy"
    with zipfile.ZipFile(npz_path, "r") as zf:
        if member not in zf.namelist():
            return None
        with zf.open(member, "r") as f:
            version = fmt.read_magic(f)
            shape, fortran_order, dtype = fmt._read_array_header(f, version)
            shape = tuple(shape)

            if len(shape) != 4:
                # We only support sampling for [T,C,H,W]
                return None

            T, C, H, W = shape
            n = int(min(int(n_obs), int(T)))
            if n <= 0:
                return None

            obs_elems = int(C) * int(H) * int(W)
            n_elems = n * obs_elems
            n_bytes = n_elems * int(dtype.itemsize)

            raw = f.read(n_bytes)
            if len(raw) < n_bytes:
                # unexpected short read; still try parse what we have
                n_elems_avail = len(raw) // int(dtype.itemsize)
                if n_elems_avail <= 0:
                    return None
                n = max(1, n_elems_avail // obs_elems)
                n_elems = n * obs_elems
                raw = raw[: n_elems * int(dtype.itemsize)]

            arr = np.frombuffer(raw, dtype=dtype, count=n_elems).reshape((n, int(C), int(H), int(W)))
            if bool(fortran_order):
                # extremely unlikely here; but keep safe
                arr = np.ascontiguousarray(arr)
            return arr


# =============================================================================
# Main dataset loading (excluding observations by default)
# =============================================================================

def load_nonobs_arrays(path: str) -> Dict[str, np.ndarray]:
    """
    Load actions/rewards/dones/episode_lengths (small-ish) from concatenated npz.
    Avoid loading observations (can be huge).
    """
    d = np.load(path, allow_pickle=False)

    required = {"actions", "rewards", "dones", "episode_lengths"}
    if not required.issubset(set(d.files)):
        raise ValueError(f"Unknown npz format. keys={list(d.files)}")

    actions = d["actions"]
    rewards = d["rewards"]
    dones = d["dones"]
    lengths = d["episode_lengths"].astype(np.int64)

    # Normalize rewards shape: [sum_T, 1] -> [sum_T]
    if rewards.ndim == 2 and rewards.shape[1] == 1:
        rewards = rewards[:, 0]

    total = int(lengths.sum())
    if actions.shape[0] != total or rewards.shape[0] != total or dones.shape[0] != total:
        raise ValueError(
            "Inconsistent shapes: sum(episode_lengths) must equal first dim of arrays.\n"
            f"sum_len={total}, actions={actions.shape}, rewards={rewards.shape}, dones={dones.shape}"
        )

    starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    ends = starts + lengths

    return {
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "episode_lengths": lengths,
        "starts": starts,
        "ends": ends,
    }


def compute_episode_returns(rewards: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    rets = np.empty(len(starts), dtype=np.float32)
    for i, (s, e) in enumerate(zip(starts, ends)):
        rets[i] = float(rewards[int(s):int(e)].sum())
    return rets


def summarize_all(task_name: str, lengths: np.ndarray, rets: np.ndarray) -> None:
    per_step = rets / np.maximum(lengths.astype(np.float32), 1.0)

    print(f"\n==================== {task_name} ====================")
    print(f"episodes: {len(lengths)}")
    print(
        "return  min/mean/median/max: "
        f"{rets.min():.2f} / {rets.mean():.2f} / {np.median(rets):.2f} / {rets.max():.2f}"
    )
    print(f"return  p90/p95/p99: {np.percentile(rets, [90, 95, 99]).round(2)}")
    print(
        "length  min/mean/median/max: "
        f"{int(lengths.min())} / {lengths.mean():.1f} / {int(np.median(lengths))} / {int(lengths.max())}"
    )
    print(
        "reward/step min/mean/median/max: "
        f"{per_step.min():.4f} / {per_step.mean():.4f} / {np.median(per_step):.4f} / {per_step.max():.4f}"
    )
    print(f"total steps: {int(lengths.sum())}")


def sanity_checks(task_name: str, npz_path: str, data: Dict[str, np.ndarray], obs_sample_n: int) -> None:
    actions = data["actions"]
    rewards = data["rewards"]
    dones = data["dones"]
    lengths = data["episode_lengths"]
    starts = data["starts"]
    ends = data["ends"]

    # --- basic numeric checks ---
    if np.isnan(rewards).any() or np.isinf(rewards).any():
        print("[WARN] rewards contain NaN/Inf")
    if np.isnan(actions.astype(np.float32, copy=False)).any():
        # actions are int, but keep defensive
        print("[WARN] actions contain NaN (unexpected)")

    # reward range (often clipped in your config)
    rmin, rmax = float(rewards.min()), float(rewards.max())
    print(f"[check] rewards min/max: {rmin:.3f} / {rmax:.3f}")
    if rmin < -1.0001 or rmax > 1.0001:
        print("[WARN] reward values outside [-1,1]. If you expected clip_rewards=True, something is off.")

    # action range
    amin, amax = int(actions.min()), int(actions.max())
    print(f"[check] actions min/max: {amin} / {amax}")
    if amin < 0:
        print("[WARN] actions contain negative values (unexpected for Atari)")

    # unique actions (sampled to avoid heavy work)
    sample_n = min(actions.shape[0], 200000)
    uniq = np.unique(actions[:sample_n])
    print(f"[check] unique actions in first {sample_n} steps: {len(uniq)} (min={int(uniq.min())}, max={int(uniq.max())})")

    # dones consistency
    last_done = dones[ends - 1]
    n_last_done_true = int(np.sum(last_done.astype(np.int32)))
    print(f"[check] episodes with last done=True: {n_last_done_true}/{len(lengths)}")
    if n_last_done_true < len(lengths):
        print("[WARN] Some episodes end without done=True on the last step -> likely truncated at max_len.")

    early_done_eps = 0
    for i, (s, e) in enumerate(zip(starts, ends)):
        s_i, e_i = int(s), int(e)
        if e_i - s_i <= 1:
            continue
        if bool(dones[s_i : e_i - 1].any()):
            early_done_eps += 1
    print(f"[check] episodes with early done=True (before last step): {early_done_eps}/{len(lengths)}")
    if early_done_eps > 0:
        print("[WARN] Some episodes contain done=True before the last step. For your collector this usually should not happen.")

    # --- observations metadata check (no full load) ---
    try:
        obs_shape, obs_dtype, _, _ = read_npy_header_from_npz(npz_path, "observations")
        print(f"[check] observations header: shape={obs_shape}, dtype={obs_dtype}")
        if len(obs_shape) != 4:
            print("[WARN] observations do not have shape [T,C,H,W].")
    except Exception as e:
        print(f"[WARN] Could not read observations header: {e}")

    # --- lightweight obs sample (prefix only) ---
    if obs_sample_n > 0:
        try:
            sample = sample_prefix_from_observations_npz(npz_path, n_obs=obs_sample_n)
            if sample is None:
                print("[WARN] Could not sample observations (missing or unexpected format).")
            else:
                smin = float(sample.min())
                smax = float(sample.max())
                print(f"[check] obs sample (first {sample.shape[0]} obs) min/max: {smin:.4f} / {smax:.4f}  (dtype={sample.dtype})")

                if np.issubdtype(sample.dtype, np.floating):
                    if smin < -1e-3 or smax > 1.0 + 1e-3:
                        print("[WARN] Obs sample appears outside [0,1] range for floats.")
                elif sample.dtype == np.uint8:
                    if smin < 0 or smax > 255:
                        print("[WARN] Obs sample appears outside [0,255] range for uint8.")
        except Exception as e:
            print(f"[WARN] Could not sample observations: {e}")


def load_manifest_if_exists(root: Path) -> Optional[List[Dict]]:
    mf = root / "manifest.json"
    if not mf.exists():
        return None
    try:
        return json.loads(mf.read_text(encoding="utf-8"))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="resources/atari_expert", help="root folder, e.g. resources/atari_expert")
    ap.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="task names, e.g. A_Pong B_Breakout C_Seaquest (if empty, tries manifest.json, else defaults to ABC)",
    )
    ap.add_argument(
        "--npz-name",
        default="expert_minari_dqn.npz",
        help="file name inside each task dir (default: expert_minari_dqn.npz)",
    )
    ap.add_argument(
        "--obs-sample",
        type=int,
        default=4,
        help="how many first observations to sample for range check (0 disables, default=4)",
    )
    args = ap.parse_args()

    root = Path(args.root)

    tasks: List[Tuple[str, str]] = []

    if args.tasks and len(args.tasks) > 0:
        for t in args.tasks:
            npz_path = root / t / args.npz_name
            tasks.append((t, str(npz_path)))
    else:
        manifest = load_manifest_if_exists(root)
        if manifest is not None:
            for entry in manifest:
                name = entry.get("name", "UNKNOWN")
                npz_path = entry.get("npz_path", None)
                if npz_path is None:
                    continue
                tasks.append((name, npz_path))
        else:
            # fallback: ABC defaults
            for t in ["A_Pong", "B_Breakout", "C_Seaquest"]:
                npz_path = root / t / args.npz_name
                tasks.append((t, str(npz_path)))

    any_missing = False

    for task_name, npz_path in tasks:
        if not os.path.exists(npz_path):
            print(f"\n[ERROR] Missing file for {task_name}: {npz_path}")
            any_missing = True
            continue

        print(f"\n[load] {task_name}: {npz_path}")
        data = load_nonobs_arrays(npz_path)
        rets = compute_episode_returns(data["rewards"], data["starts"], data["ends"])

        summarize_all(task_name, data["episode_lengths"], rets)
        sanity_checks(task_name, npz_path, data, obs_sample_n=int(args.obs_sample))

    if any_missing:
        print("\n[hint] Check that export went to the same --root and that task folder names match.")


if __name__ == "__main__":
    main()
