#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

from clbench.adapters.atari import AtariAdapter
from clbench.benchmark.metrics_extra import per_step_report
from clbench.benchmark.metrics import StandardCLMetrics
from clbench.benchmark.runner import make_tasks, describe_tasks, BenchmarkResults
from clbench.core.registry import TaskRegistry
from clbench.io.run_logger import build_run_dir, bench_short, save_json, save_matrix_csv
from clbench.io.serialize import load_task_specs

from dt.model import DecisionTransformer

# Ensure Atari adapter is registered
TaskRegistry.register("atari", AtariAdapter())


# =============================================================================
# Gymnasium/Gym API helpers
# =============================================================================

def reset_env(env):
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
    return out, {}


def step_env(env, action: int):
    out = env.step(int(action))
    # gymnasium: (obs, reward, terminated, truncated, info)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, float(reward), done, info
    # old gym: (obs, reward, done, info)
    obs, reward, done, info = out
    return obs, float(reward), bool(done), info


def maybe_fire_after_reset(env, obs):
    """
    Many Atari games (notably Breakout) require FIRE to start.
    If the action meanings include "FIRE", we press it once after reset.

    Returns the (possibly updated) observation after the FIRE step.
    """
    try:
        meanings = env.unwrapped.get_action_meanings()
        if isinstance(meanings, (list, tuple)) and "FIRE" in meanings:
            fire_action = meanings.index("FIRE")
            obs2, _r, done, _info = step_env(env, fire_action)
            # If the env ends immediately (rare), just return original obs.
            if not done:
                return obs2
    except Exception:
        pass
    return obs


# =============================================================================
# Offline dataset loading
# =============================================================================

def _choose_npz_file(task_dir: str, preferred_name: str) -> str:
    preferred_path = os.path.join(task_dir, preferred_name)
    if os.path.isfile(preferred_path):
        return preferred_path

    # Otherwise pick a reasonable candidate:
    # - ignore "*_stats*.npz" if any
    candidates = [
        f for f in os.listdir(task_dir)
        if f.endswith(".npz") and ("stats" not in f.lower())
    ]
    if not candidates:
        raise FileNotFoundError(f"No .npz trajectory files found in {task_dir}")

    candidates.sort()
    return os.path.join(task_dir, candidates[-1])


def load_npz_dataset_for_task(
    dataset_root: str,
    task_name: str,
    dataset_file: str,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Load trajectories saved in the concatenated .npz format:
        observations:    [N, C, H, W]
        actions:         [N]
        rewards:         [N]
        dones:           [N]
        episode_lengths: [n_episodes]

    Returns:
        episodes_obs:     list of [T, C, H, W]
        episodes_actions: list of [T]
        episodes_rewards: list of [T]
        returns:          [n_episodes]
    """
    task_dir = os.path.join(dataset_root, task_name)
    if not os.path.isdir(task_dir):
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    npz_path = _choose_npz_file(task_dir, preferred_name=dataset_file)
    print(f"[data] loading trajectories from {npz_path}")

    d = np.load(npz_path, allow_pickle=False)

    required = {"observations", "actions", "rewards", "dones", "episode_lengths"}
    if not required.issubset(set(d.files)):
        raise ValueError(f"Unknown npz format. keys={list(d.files)}")

    observations = d["observations"]  # [N, C, H, W]
    actions = d["actions"].reshape(-1)  # [N]
    rewards = d["rewards"].reshape(-1)  # [N]
    dones = d["dones"].reshape(-1)  # [N]
    episode_lengths = d["episode_lengths"].astype(np.int64)  # [E]

    total = int(episode_lengths.sum())
    if (
        observations.shape[0] != total
        or actions.shape[0] != total
        or rewards.shape[0] != total
        or dones.shape[0] != total
    ):
        raise ValueError(
            "Inconsistent shapes: sum(episode_lengths) must equal first dim of arrays.\n"
            f"sum_len={total}, obs={observations.shape}, actions={actions.shape}, rewards={rewards.shape}, dones={dones.shape}"
        )

    episodes_obs: List[np.ndarray] = []
    episodes_actions: List[np.ndarray] = []
    episodes_rewards: List[np.ndarray] = []

    idx = 0
    for L in episode_lengths:
        L = int(L)
        episodes_obs.append(observations[idx:idx + L])
        episodes_actions.append(actions[idx:idx + L])
        episodes_rewards.append(rewards[idx:idx + L])
        idx += L

    returns = np.array([float(np.sum(r)) for r in episodes_rewards], dtype=np.float32)

    print(
        f"[data] episodes={len(episodes_obs)}, "
        f"avg_return={returns.mean():.1f}, "
        f"min={returns.min():.1f}, "
        f"max={returns.max():.1f}"
    )

    # Quick obs sanity preview
    try:
        sample = episodes_obs[0][0]
        print(
            f"[data] obs dtype={sample.dtype}, "
            f"min={float(sample.min()):.3f}, max={float(sample.max()):.3f}, "
            f"shape={sample.shape}"
        )
    except Exception:
        pass

    return episodes_obs, episodes_actions, episodes_rewards, returns


# =============================================================================
# Offline minibatch generator
# =============================================================================

def _needs_div255(obs_sample: np.ndarray) -> bool:
    if obs_sample.dtype == np.uint8:
        return True
    # float but stored in 0..255?
    try:
        return float(np.max(obs_sample)) > 1.5
    except Exception:
        return False


def make_offline_minibatches(
    episodes_obs: List[np.ndarray],
    episodes_actions: List[np.ndarray],
    episodes_rewards: List[np.ndarray],
    seq_len: int,
    batch_size: int,
    device: torch.device,
):
    """
    Infinite generator of minibatches from offline Atari episodes.

    Yields:
        obs:   [B, L, C, H, W] float32 in [0,1]
        acts:  [B, L] int64, padded with -1 (if needed)
        rtg:   [B, L, 1] float32 (return-to-go)
        ts:    [B, L] int64 (timesteps)
        mask:  [B, L] bool (True = valid timestep)
    """
    num_episodes = len(episodes_obs)
    assert num_episodes > 0, "No episodes in offline dataset"

    # Determine obs scaling policy from a small sample
    obs_div255 = _needs_div255(episodes_obs[0][0])

    frame_shape = episodes_obs[0].shape[1:]  # (C, H, W)

    # Precompute RTG per episode
    episodes_rtg: List[np.ndarray] = []
    episode_lengths: List[int] = []
    for rew in episodes_rewards:
        r = np.asarray(rew, dtype=np.float32)
        rtg = np.flip(np.cumsum(np.flip(r)))  # rtg[t] = sum_{k=t}^{T-1} r[k]
        episodes_rtg.append(rtg.astype(np.float32))
        episode_lengths.append(int(len(r)))

    def loader():
        while True:
            obs_batch = np.zeros((batch_size, seq_len) + frame_shape, dtype=np.float32)
            actions_batch = np.full((batch_size, seq_len), -1, dtype=np.int64)
            rtg_batch = np.zeros((batch_size, seq_len, 1), dtype=np.float32)
            ts_batch = np.zeros((batch_size, seq_len), dtype=np.int64)
            mask_batch = np.zeros((batch_size, seq_len), dtype=np.bool_)

            for b in range(batch_size):
                ep_idx = np.random.randint(num_episodes)
                ep_len = episode_lengths[ep_idx]
                if ep_len <= 0:
                    continue

                if ep_len >= seq_len:
                    start = np.random.randint(0, ep_len - seq_len + 1)
                    end = start + seq_len
                    length = seq_len
                else:
                    start = 0
                    end = ep_len
                    length = ep_len

                obs_slice = episodes_obs[ep_idx][start:end]
                if obs_div255:
                    obs_batch[b, :length] = obs_slice.astype(np.float32) / 255.0
                else:
                    obs_batch[b, :length] = obs_slice.astype(np.float32)

                actions_batch[b, :length] = episodes_actions[ep_idx][start:end].astype(np.int64)

                rtg_seq = episodes_rtg[ep_idx][start:end]
                rtg_batch[b, :length, 0] = rtg_seq

                ts_batch[b, :length] = np.arange(start, end, dtype=np.int64)
                mask_batch[b, :length] = True

            yield (
                torch.tensor(obs_batch, device=device, dtype=torch.float32),
                torch.tensor(actions_batch, device=device, dtype=torch.long),
                torch.tensor(rtg_batch, device=device, dtype=torch.float32),
                torch.tensor(ts_batch, device=device, dtype=torch.long),
                torch.tensor(mask_batch, device=device, dtype=torch.bool),
            )

    return loader()


# =============================================================================
# Train DT (single task, offline)
# =============================================================================

def train_single_task_dt_offline(
    env,
    obs_shape,
    n_actions: int,
    seq_len: int,
    device: torch.device,
    steps: int,
    batch_size: int,
    episodes_obs: List[np.ndarray],
    episodes_actions: List[np.ndarray],
    episodes_rewards: List[np.ndarray],
    d_model: int,
    n_layers: int,
    n_heads: int,
    p_drop: float,
    max_ep_len_embed: int,
) -> DecisionTransformer:
    model = DecisionTransformer(
        obs_shape=obs_shape,
        n_actions=n_actions,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        seq_len=seq_len,
        p_drop=p_drop,
        max_ep_len=max_ep_len_embed,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    loader = make_offline_minibatches(
        episodes_obs, episodes_actions, episodes_rewards,
        seq_len=seq_len, batch_size=batch_size, device=device
    )

    model.train()
    for step in range(int(steps)):
        obs, actions, rtg, ts, mask = next(loader)

        logits = model(obs, actions, rtg, ts, attention_mask=mask)  # [B, L, n_actions]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            actions.reshape(-1),
            ignore_index=-1,
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % max(1, int(steps) // 10) == 0:
            print(f"[{env.spec.id}] step {step + 1}/{steps}, loss={loss.item():.4f}")

    return model


# =============================================================================
# Evaluate DT in env
# =============================================================================

@torch.no_grad()
def evaluate_dt_in_env(
    model: DecisionTransformer,
    env,
    episodes: int,
    max_steps: int,
    target_return: float,
    fire_reset: bool = True,
) -> float:
    """
    Evaluate DT with a desired target_return.
    We update rtg_remaining -= reward each step (standard DT rollout).
    """
    returns: List[float] = []

    for _ep in range(int(episodes)):
        obs, _ = reset_env(env)
        model.reset_history()

        if fire_reset:
            obs = maybe_fire_after_reset(env, obs)

        rtg_rem = float(target_return)
        ep_ret = 0.0

        for t in range(int(max_steps)):
            a = model.act(obs, rtg_scalar=rtg_rem, t=t, prev_action=0)
            obs, r, done, _ = step_env(env, a)
            ep_ret += float(r)
            rtg_rem -= float(r)

            if done:
                break

        returns.append(ep_ret)

    return float(np.mean(returns)) if returns else 0.0


def pick_target_return(returns: np.ndarray, mode: str) -> float:
    mode = str(mode).lower()
    if returns.size == 0:
        return 0.0
    if mode == "max":
        return float(np.max(returns))
    if mode == "p90":
        return float(np.percentile(returns, 90))
    if mode == "mean":
        return float(np.mean(returns))
    raise ValueError("target mode must be one of: max, p90, mean")


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True, help="JSON spec file, e.g. configs/specs_atari.json")

    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--steps", type=int, default=20000, help="SGD steps per task")
    p.add_argument("--batch-size", type=int, default=64)

    p.add_argument("--episodes-eval", type=int, default=10)
    p.add_argument("--max-ep-len", type=int, default=5000, help="Max env steps per eval episode")

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--p-drop", type=float, default=0.1)

    p.add_argument("--runs-root", type=str, default="runs")
    p.add_argument("--tag", type=str, default="")

    p.add_argument("--dataset-root", type=str, default="resources/atari_expert")
    p.add_argument(
        "--dataset-file",
        type=str,
        default="expert_minari_dqn.npz",
        help="Preferred .npz filename inside each task folder (fallback: last *.npz).",
    )

    p.add_argument(
        "--min-episode-return",
        type=float,
        default=None,
        help="If set, keep only episodes with return >= this value (use 0 for Pong).",
    )

    p.add_argument("--target-mode", type=str, default="max", choices=["max", "p90", "mean"])
    p.add_argument("--target-return", type=float, default=None, help="If set, overrides target-mode.")
    p.add_argument("--fire-reset", action="store_true", help="Press FIRE after reset if available.")
    p.add_argument("--no-fire-reset", dest="fire_reset", action="store_false")
    p.set_defaults(fire_reset=True)

    args = p.parse_args()

    device = torch.device(args.device)
    print(f"[device] using {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Load tasks
    specs = load_task_specs(args.spec)
    bench = "atari"
    envs = make_tasks(bench, specs)
    print(describe_tasks(envs, bench))

    # Run dir
    spec_tag = os.path.splitext(os.path.basename(args.spec))[0]
    run_dir = build_run_dir(args.runs_root, bench, "single", tag=args.tag or spec_tag)
    print(f"[run_dir] {run_dir}")

    results: Dict[str, float] = {}

    for i, (name, env) in enumerate(envs.items(), start=1):
        print(f"\n[Single-task {i}/{len(envs)}] {name}")
        seed = i
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        obs_shape = env.observation_space.shape
        n_actions = env.action_space.n

        # Load dataset
        episodes_obs, episodes_actions, episodes_rewards, ep_returns = load_npz_dataset_for_task(
            dataset_root=args.dataset_root,
            task_name=name,
            dataset_file=args.dataset_file,
        )

        # po load_npz_dataset_for_task(...)
        thr = None
        if name == "A_Pong":
            thr = 15.0  # albo 15.0

        if thr is not None:
            keep = ep_returns >= thr
            episodes_obs = [ep for ep, k in zip(episodes_obs, keep) if k]
            episodes_actions = [ep for ep, k in zip(episodes_actions, keep) if k]
            episodes_rewards = [ep for ep, k in zip(episodes_rewards, keep) if k]
            ep_returns = ep_returns[keep]
            print(f"[data] filtered episodes by return>={thr}: kept {keep.sum()}/{len(keep)}")

            print(
                f"[data] new return stats: mean={ep_returns.mean():.1f} min={ep_returns.min():.1f} max={ep_returns.max():.1f}")

        dataset_max_len = max(len(r) for r in episodes_rewards) if episodes_rewards else 1000
        max_ep_len_embed = int(max(10000, dataset_max_len + 1))

        # Train
        model = train_single_task_dt_offline(
            env=env,
            obs_shape=obs_shape,
            n_actions=n_actions,
            seq_len=args.seq_len,
            device=device,
            steps=args.steps,
            batch_size=args.batch_size,
            episodes_obs=episodes_obs,
            episodes_actions=episodes_actions,
            episodes_rewards=episodes_rewards,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            p_drop=args.p_drop,
            max_ep_len_embed=max_ep_len_embed,
        )

        # Eval
        model.eval()

        if args.target_return is not None:
            target = float(args.target_return)
        else:
            target = pick_target_return(ep_returns, mode=args.target_mode)

        avg_ret = evaluate_dt_in_env(
            model=model,
            env=env,
            episodes=args.episodes_eval,
            max_steps=args.max_ep_len,
            target_return=target,
            fire_reset=bool(args.fire_reset),
        )

        results[name] = float(avg_ret)
        print(
            f"[eval single-task] {name}: avg return over {args.episodes_eval} episodes = "
            f"{avg_ret:.3f} (target_return={target:.2f})"
        )

        try:
            env.close()
        except Exception:
            pass

    # Build diagonal perf matrix
    task_names = list(envs.keys())
    n = len(task_names)
    P = np.zeros((n, n), dtype=np.float32)
    name_to_idx = {n: i for i, n in enumerate(task_names)}
    for name, ret in results.items():
        P[name_to_idx[name], name_to_idx[name]] = float(ret)

    results_obj = BenchmarkResults(
        name=f"DT-single:{args.spec}",
        task_names=task_names,
        perf_matrix=P,
    )
    metrics = StandardCLMetrics.compute(results_obj)

    bench_name_short = bench_short(bench)

    save_json(
        os.path.join(run_dir, "results.json"),
        {
            "name": results_obj.name,
            "task_names": task_names,
            "perf_matrix": P.tolist(),
            "metrics": metrics,
            "mode": "single-task",
            "dataset_root": os.path.abspath(args.dataset_root),
            "dataset_file": args.dataset_file,
            "target_mode": args.target_mode,
            "target_return": args.target_return,
        },
    )
    save_matrix_csv(os.path.join(run_dir, "matrix.csv"), task_names, P)

    steps_report = per_step_report(task_names, P)
    save_json(os.path.join(run_dir, "per_step.json"), {"per_step": steps_report})
    if steps_report:
        import csv
        with open(os.path.join(run_dir, "per_step.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(steps_report[0].keys()))
            w.writeheader()
            w.writerows(steps_report)

    save_json(
        os.path.join(run_dir, f"rez_{bench_name_short}.json"),
        {"metrics": metrics, "strategy": "single"},
    )

    print("\n=== Single-task Atari DT results ===")
    for name, r in results.items():
        print(f"{name}: {r:.3f}")
    print(f"\n[artifacts] saved to: {run_dir}")


if __name__ == "__main__":
    main()
