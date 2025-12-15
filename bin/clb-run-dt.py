#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import numpy as np
import torch

from types import SimpleNamespace  # NEW: for simple trajectory objects

from clbench.core.registry import TaskRegistry
from clbench.adapters.cartpole import CartPoleAdapter
from clbench.adapters.atari import AtariAdapter
from clbench.io.serialize import load_task_specs
from clbench.benchmark.runner import make_tasks, describe_tasks, BenchmarkResults
from clbench.benchmark.metrics import StandardCLMetrics
from clbench.benchmark.metrics_extra import per_step_report
from clbench.io.run_logger import (
    build_run_dir,
    save_json,
    save_matrix_csv,
    bench_short,
    save_task_gen_json,
)
from dt.dataset import Trajectory

from strategies.naive import NaiveStrategy
from strategies.cumulative import CumulativeReplayStrategy
from strategies.ewc import EWCStrategy

from dt.utils import collect_trajectories, evaluate_dt
from dt.io_traj import save_trajs_pickle

TaskRegistry.register("cartpole", CartPoleAdapter())
TaskRegistry.register("atari", AtariAdapter())


# ====== OFFLINE LOADER: expert_trajs_*.npz -> list of trajectory-like objects ======


def load_offline_trajs_for_task(dataset_root: str, task_name: str):
    """
    Load expert / mixed trajectories for a single task from .npz and
    convert them to a list of Trajectory objects compatible with make_minibatches.

    Expected structure:
        dataset_root/
          task_name/
            expert_trajs_*.npz  (e.g. expert_trajs_p0.70.npz)

    Inside the .npz:
        observations:    [N, ...]
        actions:         [N]
        rewards:         [N]
        dones:           [N]
        episode_lengths: [n_episodes]

    Returns:
        List[Trajectory] with fields:
            - obs:            [T, ...]
            - actions:        [T]
            - returns_to_go:  [T]
            - timesteps:      [T]
    """
    task_dir = os.path.join(dataset_root, task_name)
    if not os.path.isdir(task_dir):
        raise FileNotFoundError(f"[offline] task directory not found: {task_dir}")

    # pick any .npz file with trajectories
    candidates = [f for f in os.listdir(task_dir) if f.endswith(".npz")]
    if not candidates:
        raise FileNotFoundError(f"[offline] no .npz trajectory files found in {task_dir}")
    candidates.sort()
    npz_path = os.path.join(task_dir, candidates[-1])  # last one, e.g. expert_trajs_p0.70.npz

    print(f"[offline] loading trajectories for task '{task_name}' from {npz_path}")
    data = np.load(npz_path)

    observations = data["observations"]        # [N, obs_dim] or [N, C, H, W]
    actions = data["actions"]                  # [N]
    rewards = data["rewards"]                  # [N]
    dones = data["dones"]                      # [N]  # not używane, ale zostawiamy
    episode_lengths = data["episode_lengths"]  # [n_episodes]

    trajs = []
    idx = 0
    returns = []

    for L in episode_lengths:
        L = int(L)
        obs_ep = observations[idx:idx + L]
        act_ep = actions[idx:idx + L]
        rew_ep = rewards[idx:idx + L]
        _done_ep = dones[idx:idx + L]
        idx += L

        # returns-to-go jak w DT: rtg[t] = sum_{k=t}^{T-1} r[k]
        r = rew_ep.astype(np.float32)
        rtg = np.flip(np.cumsum(np.flip(r, axis=0), axis=0), axis=0)  # [T]

        T = len(act_ep)
        ts = np.arange(T, dtype=np.int64)

        traj = Trajectory(
            obs=obs_ep.astype(np.float32),
            actions=act_ep.astype(np.int64),
            rewards=rew_ep.astype(np.float32),
            timesteps=ts,
            returns_to_go=rtg.astype(np.float32),
        )

        trajs.append(traj)
        returns.append(float(r.sum()))

    if trajs:
        print(
            f"[offline] task={task_name}, episodes={len(trajs)}, "
            f"avg_return={np.mean(returns):.1f}, min={np.min(returns):.1f}, "
            f"max={np.max(returns):.1f}"
        )
    else:
        print(f"[offline] task={task_name}, WARNING: no episodes reconstructed")

    return trajs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument("--strategy", choices=["cumulative", "ewc", "naive"], default="cumulative")
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--episodes-eval", type=int, default=5)
    p.add_argument("--steps-per-task", type=int, default=2000)
    p.add_argument("--warm-episodes", type=int, default=10)
    p.add_argument("--collect-episodes", type=int, default=10)
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--runs-root", type=str, default="runs")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--dump-trajs", action="store_true")

    # OFFLINE datasets root: if provided, we use expert trajs instead of online collect
    p.add_argument(
        "--dataset-root",
        type=str,
        default="",
        help=(
            "If non-empty, use offline expert trajectories from this root instead of "
            "collecting on-policy. Expects {dataset_root}/{task_name}/expert_trajs_*.npz"
        ),
    )

    args = p.parse_args()

    specs = load_task_specs(args.spec)
    is_atari = any((s.params or {}).get("game", "").startswith("ALE/") for s in specs)
    bench = "atari" if is_atari else "cartpole"
    envs = make_tasks(bench, specs)
    print(describe_tasks(envs, bench))

    task_names = list(envs.keys())
    n = len(task_names)
    P = np.zeros((n, n), dtype=np.float32)

    spec_tag = os.path.splitext(os.path.basename(args.spec))[0]
    run_dir = build_run_dir(args.runs_root, bench, args.strategy, tag=args.tag or spec_tag)
    print(f"[run_dir] {run_dir}")

    # Work with full env list (for shared model head)
    env_list = list(envs.values())
    first_env = env_list[0]

    obs_shape = (
        first_env.observation_space.shape
        if hasattr(first_env.observation_space, "shape")
        else (first_env.observation_space.n,)
    )

    # MODEL HEAD SIZE = MAX ACTIONS OVER ALL TASKS
    n_actions = max(e.action_space.n for e in env_list)

    if args.strategy == "naive":
        strategy = NaiveStrategy(obs_shape, n_actions, args.seq_len, args.device)
    elif args.strategy == "cumulative":
        strategy = CumulativeReplayStrategy(obs_shape, n_actions, args.seq_len, args.device)
    else:
        strategy = EWCStrategy(obs_shape, n_actions, args.seq_len, args.device)

    use_offline = bool(args.dataset_root)
    if use_offline:
        print(f"[mode] Using OFFLINE expert trajectories from: {args.dataset_root}")
    else:
        print("[mode] Using ON-POLICY collected trajectories (no dataset-root provided)")

    for i, (name, env) in enumerate(envs.items()):
        print(f"[Task {i + 1}/{n}] {name}")

        if use_offline:
            # ---- OFFLINE: load expert / mixed trajectories from disk ----
            onpol = load_offline_trajs_for_task(args.dataset_root, name)
        else:
            # ---- ONLINE: collect on-policy trajectories with current strategy.model ----
            onpol = collect_trajectories(
                env,
                strategy.model,
                n_episodes=args.collect_episodes,
                max_len=1000,
                target_return=1.0,
                device=args.device,
            )

        traj_path = None
        if args.dump_trajs:
            os.makedirs(os.path.join(run_dir, "gen"), exist_ok=True)
            traj_path = os.path.join(run_dir, "gen", f"trajs_task{i}.pkl")
            save_trajs_pickle(traj_path, onpol)

        save_task_gen_json(
            run_dir,
            i + 1,
            {
                "step": i + 1,
                "task_name": name,
                "n_trajectories": len(onpol),
                "traj_file": traj_path,
                "offline": use_offline,
                "dataset_root": args.dataset_root if use_offline else None,
            },
        )

        # Train DT with chosen CL strategy on this task's data
        strategy.train_task(onpol, steps=args.steps_per_task, batch_size=64)
        strategy.after_task(onpol)

        # Evaluate on all tasks (standard CL eval)
        for j, (n2, env2) in enumerate(envs.items()):
            P[i, j] = evaluate_dt(strategy, env2, args.episodes_eval, args.device)

    results = BenchmarkResults(
        name=f"DT-{args.strategy}:{args.spec}",
        task_names=task_names,
        perf_matrix=P,
    )
    metrics = StandardCLMetrics.compute(results)

    save_json(
        os.path.join(run_dir, "results.json"),
        {
            "name": results.name,
            "task_names": task_names,
            "perf_matrix": P.tolist(),
            "metrics": metrics,
        },
    )
    save_matrix_csv(os.path.join(run_dir, "matrix.csv"), task_names, P)

    steps = per_step_report(task_names, P)
    save_json(os.path.join(run_dir, "per_step.json"), {"per_step": steps})
    if steps:
        import csv

        with open(
            os.path.join(run_dir, "per_step.csv"),
            "w",
            encoding="utf-8",
            newline="",
        ) as f:
            w = csv.DictWriter(f, fieldnames=list(steps[0].keys()))
            w.writeheader()
            w.writerows(steps)

    save_json(
        os.path.join(run_dir, f"rez_{bench_short(bench)}.json"),
        {"metrics": metrics, "strategy": args.strategy},
    )

    print("\n=== Continual DT results (offline={} ) ===".format(use_offline))
    print(P)
    print(f"\n[artifacts] saved to: {run_dir}")


if __name__ == "__main__":
    main()
