#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, List, Tuple, Optional, Dict

import gymnasium as gym
import panda_gym  # noqa: F401  (registers Panda envs)
import numpy as np
import torch

from gymnasium.wrappers import FlattenObservation

from paths import RUNS, DATASETS_ROOT
from clbench.benchmark.metrics import StandardCLMetrics
from clbench.benchmark.metrics_extra import per_step_report
from clbench.benchmark.runner import BenchmarkResults
from clbench.io.run_logger import build_run_dir, save_json, save_matrix_csv, bench_short

from dt.dataset import Trajectory
from dt.dataset_panda import load_panda_offline_pkl
from dt.utils import evaluate_dt_panda

from strategies.naive import PandaNaiveStrategy
from strategies.cumulative import PandaCumulativeReplayStrategy


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def traj_return(tr: Trajectory) -> float:
    return float(np.sum(np.asarray(tr.rewards, dtype=np.float32)))


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
    raise ValueError("mode must be max|p90|mean")


def ensure_trajectory(obj: Any) -> Trajectory:
    """
    Accept either:
      - Trajectory object
      - dict with keys like obs/actions/rewards/(timesteps)/(returns_to_go)
    and return a Trajectory.
    """
    if isinstance(obj, Trajectory):
        return obj

    if isinstance(obj, dict):
        obs = np.asarray(obj["obs"], dtype=np.float32)
        actions = np.asarray(obj["actions"], dtype=np.float32)
        rewards = np.asarray(obj["rewards"], dtype=np.float32)

        if "timesteps" in obj:
            ts = np.asarray(obj["timesteps"], dtype=np.int64)
        else:
            ts = np.arange(actions.shape[0], dtype=np.int64)

        if "returns_to_go" in obj:
            rtg = np.asarray(obj["returns_to_go"], dtype=np.float32)
        else:
            # gamma=1 RTG
            r = rewards.astype(np.float32)
            rtg = np.flip(np.cumsum(np.flip(r, axis=0), axis=0), axis=0).astype(np.float32)

        return Trajectory(
            obs=obs,
            actions=actions,
            rewards=rewards,
            timesteps=ts,
            returns_to_go=rtg,
        )

    raise TypeError(f"Unsupported trajectory type: {type(obj)}")


def to_trajectory_list(trajs_raw: Any) -> List[Trajectory]:
    """
    Normalize whatever load_panda_offline_pkl returns into List[Trajectory].
    """
    if isinstance(trajs_raw, dict):
        # If someone saved {id: traj, ...}
        trajs_raw = list(trajs_raw.values())

    if not isinstance(trajs_raw, (list, tuple)):
        raise TypeError(f"Expected list/tuple/dict from loader, got {type(trajs_raw)}")

    trajs: List[Trajectory] = []
    for tr in trajs_raw:
        trajs.append(ensure_trajectory(tr))
    return trajs


def pad_2d_last(x: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Pad or slice a [T, D] array to [T, target_dim] along the last dimension.
    """
    x = np.asarray(x)
    assert x.ndim == 2, f"Expected [T,D], got {x.shape}"
    d = int(target_dim)
    if x.shape[1] == d:
        return x
    if x.shape[1] > d:
        return x[:, :d]
    out = np.zeros((x.shape[0], d), dtype=x.dtype)
    out[:, : x.shape[1]] = x
    return out


def pad_trajectory(tr: Trajectory, obs_dim: int, act_dim: int) -> Trajectory:
    """
    Create a NEW Trajectory with obs/actions padded to global dims.
    Rewards/timesteps/rtg stay unchanged.
    """
    obs = pad_2d_last(np.asarray(tr.obs, dtype=np.float32), obs_dim).astype(np.float32)
    actions = pad_2d_last(np.asarray(tr.actions, dtype=np.float32), act_dim).astype(np.float32)

    rewards = np.asarray(tr.rewards, dtype=np.float32)
    timesteps = np.asarray(tr.timesteps, dtype=np.int64) if hasattr(tr, "timesteps") else np.arange(len(rewards), dtype=np.int64)

    if hasattr(tr, "returns_to_go") and tr.returns_to_go is not None:
        rtg = np.asarray(tr.returns_to_go, dtype=np.float32)
    else:
        rtg = np.flip(np.cumsum(np.flip(rewards, axis=0), axis=0), axis=0).astype(np.float32)

    return Trajectory(
        obs=obs,
        actions=actions,
        rewards=rewards,
        timesteps=timesteps,
        returns_to_go=rtg,
    )


def make_env(env_id: str) -> gym.Env:
    """
    PandaGym often returns Dict observations; we flatten them to match offline vectors.
    """
    env = gym.make(env_id)
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)
    return env


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=str, default=str(RUNS))
    ap.add_argument("--strategy", type=str, default="naive", choices=["naive", "cumulative", "ewc"])
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--steps-per-task", type=int, default=2000)
    ap.add_argument("--episodes-eval", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=None, help="Max env steps per episode (default: env.spec.max_episode_steps).")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)

    # DT conditioning
    ap.add_argument("--target-mode", choices=["max", "p90", "mean"], default="max")
    ap.add_argument("--target-return", type=float, default=None, help="If set, overrides per-task target_return.")

    args = ap.parse_args()

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    # === 1) Define Panda tasks: (name, env_id, dataset_path)
    panda_tasks: List[Tuple[str, str, Path]] = [
        ("PandaReach", "PandaReachDense-v3", DATASETS_ROOT / "panda_reach_dense_1m_expert.pkl"),
        ("PandaPush", "PandaPushDense-v3", DATASETS_ROOT / "panda_push_dense_1m_expert.pkl"),
        ("PandaPickAndPlace", "PandaPickAndPlaceDense-v3", DATASETS_ROOT / "panda_pick_and_place_dense_1m_expert.pkl"),
    ]

    # === 2) Create envs + load offline trajectories
    envs: "OrderedDict[str, gym.Env]" = OrderedDict()
    all_trajs: List[List[Trajectory]] = []
    obs_dims: List[int] = []
    act_dims: List[int] = []

    print("\n=== Loading Panda tasks ===")
    for name, env_id, ds_path in panda_tasks:
        print(f"[TASK] {name} | env={env_id} | data={ds_path}")

        # env
        env = make_env(env_id)
        envs[name] = env

        # dataset
        trajs_raw = load_panda_offline_pkl(ds_path)
        trajs = to_trajectory_list(trajs_raw)
        assert len(trajs) > 0, f"No trajectories loaded from {ds_path}"

        # basic sanity
        t0 = trajs[0]
        o0 = np.asarray(t0.obs)
        a0 = np.asarray(t0.actions)
        print(f"  dataset obs shape: {o0.shape}, actions shape: {a0.shape}")
        print(f"  env obs_space: {env.observation_space}, act_space: {env.action_space}")

        # dims (assume consistent within a task)
        obs_dims.append(int(o0.shape[-1]))
        act_dims.append(int(a0.shape[-1]))

        # return stats
        rets = np.array([traj_return(t) for t in trajs], dtype=np.float32)
        print(f"  dataset episodes={len(trajs)} | return mean={rets.mean():.3f} min={rets.min():.3f} max={rets.max():.3f}")

        all_trajs.append(trajs)

    # === 3) Global model dims (max over tasks) + pad trajectories
    obs_dim_global = int(max(obs_dims)) if obs_dims else 0
    act_dim_global = int(max(act_dims)) if act_dims else 0
    print(f"\n[global dims] obs_dim={obs_dim_global} (per-task={obs_dims}), act_dim={act_dim_global} (per-task={act_dims})")

    # Pad all trajectories to global dims
    for i in range(len(all_trajs)):
        all_trajs[i] = [pad_trajectory(tr, obs_dim_global, act_dim_global) for tr in all_trajs[i]]

    # === 4) Per-task target_return for DT conditioning
    target_return_map: Dict[str, float] = {}
    for i, (name, _env_id, _ds_path) in enumerate(panda_tasks):
        rets = np.array([traj_return(t) for t in all_trajs[i]], dtype=np.float32)
        if args.target_return is not None:
            target = float(args.target_return)
        else:
            target = pick_target_return(rets, args.target_mode)
        target_return_map[name] = float(target)
        print(f"[target] {name}: target_return={target_return_map[name]:.3f} (mode={args.target_mode})")

    # === 5) Setup run dir / metrics bookkeeping
    task_names = list(envs.keys())
    n = len(task_names)
    P = np.zeros((n, n), dtype=np.float32)

    bench = "panda"
    spec_tag = "panda3"
    run_dir = build_run_dir(args.runs_root, bench, args.strategy, tag=args.tag or spec_tag)
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n[run_dir] {run_dir}")
    print(f"[tasks] {task_names}")

    # === 6) Instantiate CL strategy with GLOBAL dims
    obs_shape = (obs_dim_global,)
    if args.strategy == "naive":
        strategy = PandaNaiveStrategy(obs_shape, act_dim_global, args.seq_len, device)
    elif args.strategy == "cumulative":
        strategy = PandaCumulativeReplayStrategy(obs_shape, act_dim_global, args.seq_len, device)
    else:
        raise NotImplementedError("For Panda continuous actions only 'naive' and 'cumulative' are supported here.")

    # === 7) Task loop: train offline, evaluate online
    for i, (task_name, env_train) in enumerate(envs.items()):
        print(f"\n[Task {i + 1}/{n}] Train on {task_name}")

        task_trajs = all_trajs[i]

        strategy.train_task(task_trajs, steps=args.steps_per_task, batch_size=64)
        strategy.after_task(task_trajs)

        # Evaluate on all tasks
        for j, (eval_name, env_eval) in enumerate(envs.items()):
            max_steps = args.max_steps
            if max_steps is None:
                max_steps = getattr(getattr(env_eval, "spec", None), "max_episode_steps", None)
            if max_steps is None:
                max_steps = 200

            score = evaluate_dt_panda(
                model=strategy.model,
                env=env_eval,
                episodes=args.episodes_eval,
                device=torch.device(device),
                max_steps=int(max_steps),
                target_return=float(target_return_map[eval_name]),
                seed=int(args.seed + 1000 * i),  # deterministic per training step
                obs_pad_to=obs_dim_global,
                act_pad_to=act_dim_global,
                clip_action=True,
            )
            P[i, j] = score
            print(f"[eval] after task {i+1} on {eval_name}: {score:.3f} (target={target_return_map[eval_name]:.3f})")

    # === 8) Metrics + saving
    results = BenchmarkResults(
        name=f"DT-{args.strategy}:panda3",
        task_names=task_names,
        perf_matrix=P,
    )
    metrics = StandardCLMetrics.compute(results)

    save_json(os.path.join(run_dir, "results.json"), {
        "name": results.name,
        "task_names": task_names,
        "perf_matrix": P.tolist(),
        "metrics": metrics,
        "obs_dim_global": obs_dim_global,
        "act_dim_global": act_dim_global,
        "target_return_map": target_return_map,
        "seed": args.seed,
        "strategy": args.strategy,
        "seq_len": args.seq_len,
        "steps_per_task": args.steps_per_task,
        "episodes_eval": args.episodes_eval,
        "max_steps": args.max_steps,
    })
    save_matrix_csv(os.path.join(run_dir, "matrix.csv"), task_names, P)

    steps = per_step_report(task_names, P)
    save_json(os.path.join(run_dir, "per_step.json"), {"per_step": steps})
    if steps:
        with open(os.path.join(run_dir, "per_step.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(steps[0].keys()))
            w.writeheader()
            w.writerows(steps)

    save_json(os.path.join(run_dir, f"rez_{bench_short(bench)}.json"), {
        "metrics": metrics,
        "strategy": args.strategy,
    })

    # Close envs
    for env in envs.values():
        try:
            env.close()
        except Exception:
            pass

    print("\nDone. Saved Panda CL results to:", run_dir)


if __name__ == "__main__":
    main()
