#!/usr/bin/env python
from __future__ import annotations
import argparse
import os
import json

import numpy as np
import torch

from clbench.adapters.cartpole import CartPoleAdapter
from clbench.core.registry import TaskRegistry
from clbench.benchmark.runner import make_tasks, describe_tasks, BenchmarkResults
from clbench.io.serialize import load_task_specs
from clbench.io.run_logger import build_run_dir, save_json, save_matrix_csv, bench_short
from clbench.benchmark.metrics_extra import per_step_report
from clbench.benchmark.metrics import StandardCLMetrics
from expertes.cartpole_dqn import train_cartpole_dqn_expert, collect_expert_trajectories, save_trajectories_npz

# Register CartPole in CLBench registry
TaskRegistry.register("cartpole", CartPoleAdapter())


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--spec",
        required=True,
        help="CartPole spec file, e.g. specs_cp.json",
    )
    p.add_argument(
        "--episodes-per-task",
        type=int,
        default=100,
        help="Number of expert episodes to record per task",
    )
    p.add_argument(
        "--max-len",
        type=int,
        default=500,
        help="Maximum episode length when collecting trajectories",
    )
    p.add_argument(
        "--total-steps-expert",
        type=int,
        default=50000,
        help="Number of environment steps for DQN expert training per task",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="data/cartpole_expert",
        help="Root directory for saving expert datasets (.npz + meta.json)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for expert training (cpu / cuda)",
    )
    p.add_argument(
        "--expert-action-prob",
        type=float,
        default=1.0,
        help=(
            "Per-step probability of using expert action when collecting trajectories "
            "(1.0 = pure expert, 0.0 = pure random)."
        ),
    )
    p.add_argument(
        "--runs-root",
        type=str,
        default="runs",
        help="Root directory for CL-style logging of expert performance",
    )
    p.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional run tag used in run_dir construction",
    )

    args = p.parse_args()
    device = torch.device(args.device)
    print(f"[device] using {device}")

    # Load tasks from spec (same as DT scripts)
    specs = load_task_specs(args.spec)
    bench = "cartpole"
    envs = make_tasks(bench, specs)
    print(describe_tasks(envs, bench))

    # CL-style run_dir for expert runs
    spec_tag = os.path.splitext(os.path.basename(args.spec))[0]
    run_dir = build_run_dir(
        args.runs_root,
        bench,
        strategy="expert",
        tag=args.tag or spec_tag,
    )
    print(f"[run_dir] {run_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    # For logging per-task performance (avg return from collected trajectories)
    task_avg_returns = {}

    for name, env in envs.items():
        print("\n======================================")
        print(f"[CartPole Expert] task={name}, env_id={env.spec.id}")
        print("======================================")

        # 1) Train DQN expert for this specific task
        expert = train_cartpole_dqn_expert(
            env,
            device=device,
            total_steps=args.total_steps_expert,
        )

        # 2) Collect expert trajectories (possibly mixed with random)
        trajectories = collect_expert_trajectories(
            env,
            expert,
            device=device,
            n_episodes=args.episodes-per-task if hasattr(args, "episodes-per-task") else args.episodes_per_task,
            max_len=args.max_len,
            expert_action_prob=args.expert_action_prob,
        )

        # Compute per-task average return from collected trajectories
        returns = [float(tr["rewards"].sum()) for tr in trajectories]
        avg_return = float(np.mean(returns)) if returns else 0.0
        max_return = float(np.max(returns)) if returns else 0.0
        task_avg_returns[name] = avg_return
        print(
            f"[expert perf] task={name}, "
            f"episodes={len(trajectories)}, "
            f"avg_return={avg_return:.2f}, max_return={max_return:.2f}"
        )

        # 3) Save to disk: one .npz file per task (dataset for offline DT)
        task_dir = os.path.join(args.out_dir, name)
        os.makedirs(task_dir, exist_ok=True)

        prob_str = f"{args.expert_action_prob:.2f}"
        out_filename = f"expert_trajs_p{prob_str}.npz"
        out_path = os.path.join(task_dir, out_filename)

        save_trajectories_npz(trajectories, out_path)

        # Save simple metadata alongside trajectories
        meta_path = os.path.join(task_dir, "meta.json")
        meta = {
            "task_name": name,
            "env_id": env.spec.id,
            "expert_action_prob": float(args.expert_action_prob),
            "episodes_per_task": int(args.episodes_per_task),
            "max_len": int(args.max_len),
            "total_steps_expert": int(args.total_steps_expert),
            "device": args.device,
            "spec_file": os.path.abspath(args.spec),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[save] trajectories -> {out_path}")
        print(f"[save] meta         -> {meta_path}")

    # ====== CL-style metrics + logging for expert performance ======
    task_names = list(envs.keys())
    n_tasks = len(task_names)
    P = np.zeros((n_tasks, n_tasks), dtype=np.float32)

    for i, name in enumerate(task_names):
        P[i, i] = float(task_avg_returns.get(name, 0.0))

    results_obj = BenchmarkResults(
        name=f"DQN-expert:{args.spec}",
        task_names=task_names,
        perf_matrix=P,
    )
    metrics = StandardCLMetrics.compute(results_obj)

    # results.json
    save_json(
        os.path.join(run_dir, "results.json"),
        {
            "name": results_obj.name,
            "task_names": task_names,
            "perf_matrix": P.tolist(),
            "metrics": metrics,
            "mode": "expert",
        },
    )

    # matrix.csv
    save_matrix_csv(
        os.path.join(run_dir, "matrix.csv"),
        task_names,
        P,
    )

    # per_step.json / per_step.csv
    steps = per_step_report(task_names, P)
    save_json(
        os.path.join(run_dir, "per_step.json"),
        {"per_step": steps},
    )
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

    # rez_cp.json — lightweight summary like in CL runs
    bench_name_short = bench_short(bench)
    save_json(
        os.path.join(run_dir, f"rez_{bench_name_short}.json"),
        {"metrics": metrics, "strategy": "expert"},
    )

    print("\n=== CartPole DQN expert results (avg returns from datasets) ===")
    for name, r in task_avg_returns.items():
        print(f"{name}: {r:.3f}")
    print(f"\n[expert artifacts] saved to: {run_dir}")


if __name__ == "__main__":
    main()
