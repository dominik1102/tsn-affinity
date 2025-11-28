#!/usr/bin/env python3
from __future__ import annotations
import argparse, os
import csv
from collections import OrderedDict

import gymnasium as gym
import panda_gym

from strategies.cumulative import PandaCumulativeReplayStrategy

print([k for k in gym.envs.registry.keys() if "Panda" in k])
from pathlib import Path
import numpy as np
import torch

from clbench.benchmark.metrics import StandardCLMetrics
from clbench.benchmark.metrics_extra import per_step_report
from clbench.benchmark.runner import BenchmarkResults
from clbench.io.run_logger import build_run_dir, save_json, save_matrix_csv, bench_short
from dt.dataset_panda import load_panda_offline_pkl

from strategies.naive import PandaNaiveStrategy

from dt.utils import evaluate_dt_panda


ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = ROOT / "resources" / "datasets"
RUNS = ROOT / "runs"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=str, default=RUNS)
    ap.add_argument("--strategy", type=str, default="naive",
                    choices=["naive", "cumulative", "ewc"])
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--steps-per-task", type=int, default=2000)
    ap.add_argument("--episodes-eval", type=int, default=5)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    # === 1) Define 3 Panda tasks: name, env id, path to .pkl ===
    panda_tasks = [
        # NAME,           GYM ENV ID,           DATASET PATH (Path)
        ("PandaReach", "PandaReachDense-v3", DATASETS_ROOT / "panda_reach_dense_1m_expert.pkl"),
        ("PandaPush", "PandaPushDense-v3", DATASETS_ROOT / "panda_push_dense_1m_expert.pkl"),
        ("PandaPickAndPlace", "PandaPickAndPlaceDense-v3", DATASETS_ROOT / "panda_pick_and_place_dense_1m_expert.pkl"),
    ]

    # === 2) Create envs and load offline trajectories ===
    envs = OrderedDict()
    all_trajs = []  # list[list[Trajectory]]
    for name, env_id, ds_path in panda_tasks:
        print(f"[PANDA TASK] {name} / env={env_id} / data={ds_path}")
        envs[name] = gym.make(env_id)
        trajs = load_panda_offline_pkl(ds_path)
        assert len(trajs) > 0, f"No trajectories loaded from {ds_path}"
        all_trajs.append(trajs)

    task_names = list(envs.keys())
    n = len(task_names)
    P = np.zeros((n, n), dtype=np.float32)

    # === 3) Determine obs_shape and n_actions from envs / trajectories ===
    first_env = next(iter(envs.values()))
    # Observation passed to DT is already flattened inside Trajectory,
    # so we take the dimensionality from trajectories:
    first_traj = all_trajs[0][0]
    obs_shape = (first_traj.obs.shape[-1],)

    # Discrete actions:
    max_act = max(int(tr.actions.max()) for trajs in all_trajs for tr in trajs)
    act_dim = first_traj.actions.shape[-1]   # zamiast n_actions

    bench = "panda"
    spec_tag = "panda3"
    run_dir = build_run_dir(args.runs_root, bench, args.strategy, tag=args.tag or spec_tag)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[run_dir] {run_dir}")
    print(f"[tasks] {task_names}")

    # === 4) CL strategy (same as in your existing code) ===
    if args.strategy == "naive":
        strategy = PandaNaiveStrategy(obs_shape, act_dim, args.seq_len, device)
    elif args.strategy == "cumulative":
        strategy = PandaCumulativeReplayStrategy(obs_shape, act_dim, args.seq_len, device)
    else:
        raise NotImplementedError("For Panda continuous actions only 'naive' is implemented for now.")

    # === 5) Task loop: train offline, evaluate online ===
    for i, (name, env) in enumerate(envs.items()):
        print(f"\n[Task {i + 1}/{n}] {name}")

        # trajectories for this task (offline from .pkl)
        task_trajs = all_trajs[i]

        # training: instead of collect_trajectories, train on offline trajectories
        strategy.train_task(task_trajs, steps=args.steps_per_task, batch_size=64)
        strategy.after_task(task_trajs)

        # evaluation: same as before – rollout in each env
        for j, (name_j, env_j) in enumerate(envs.items()):
            score = evaluate_dt_panda(strategy, env_j, args.episodes_eval, device)
            P[i, j] = score
            print(f"[eval] after task {i+1} on env {name_j}: {score:.3f}")

    # === 6) CL metrics and saving results (same pattern as your main runner) ===
    results = BenchmarkResults(
        name=f"DT-{args.strategy}:panda3",
        task_names=task_names,
        perf_matrix=P,
    )
    metrics = StandardCLMetrics.compute(results)

    # results.json
    save_json(os.path.join(run_dir, "results.json"), {
        "name": results.name,
        "task_names": task_names,
        "perf_matrix": P.tolist(),
        "metrics": metrics,
    })

    # matrix.csv
    save_matrix_csv(os.path.join(run_dir, "matrix.csv"), task_names, P)

    # per_step
    steps = per_step_report(task_names, P)
    save_json(os.path.join(run_dir, "per_step.json"), {"per_step": steps})
    with open(os.path.join(run_dir, "per_step.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(steps[0].keys()))
        w.writeheader()
        w.writerows(steps)

    # summary file with metrics
    save_json(os.path.join(run_dir, f"rez_{bench_short(bench)}.json"), {
        "metrics": metrics,
        "strategy": args.strategy,
    })

    print("\nDone. Saved Panda CL results to:", run_dir)


if __name__ == "__main__":
    main()