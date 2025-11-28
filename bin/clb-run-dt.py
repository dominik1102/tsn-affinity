#!/usr/bin/env python
from __future__ import annotations

import argparse, os, numpy as np, torch

from clbench.core.registry import TaskRegistry
from clbench.adapters.cartpole import CartPoleAdapter
from clbench.adapters.atari import AtariAdapter
from clbench.io.serialize import load_task_specs
from clbench.benchmark.runner import make_tasks, describe_tasks, BenchmarkResults
from clbench.benchmark.metrics import StandardCLMetrics
from clbench.benchmark.metrics_extra import per_step_report
from clbench.io.run_logger import build_run_dir, save_json, save_matrix_csv, bench_short, save_task_gen_json

from strategies.naive import NaiveStrategy
from strategies.cumulative import CumulativeReplayStrategy
from strategies.ewc import EWCStrategy

from dt.utils import collect_trajectories, evaluate_dt
from dt.io_traj import save_trajs_pickle

TaskRegistry.register("cartpole", CartPoleAdapter())
TaskRegistry.register("atari", AtariAdapter())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument("--strategy", choices=["cumulative", "ewc", "naive"], default="cumulative")
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--episodes-eval", type=int, default=5)
    p.add_argument("--steps-per-task", type=int, default=2000)
    p.add_argument("--warm-episodes", type=int, default=10)
    p.add_argument("--collect-episodes", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--runs-root", type=str, default="runs")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--dump-trajs", action="store_true")
    args = p.parse_args()

    specs = load_task_specs(args.spec)
    is_atari = any((s.params or {}).get("game", "").startswith("ALE/") for s in specs)
    bench = "atari" if is_atari else "cartpole"
    envs = make_tasks(bench, specs)
    print(describe_tasks(envs, bench))

    envs = make_tasks(bench, specs)
    print(describe_tasks(envs, bench))

    task_names = list(envs.keys())
    n = len(task_names)
    P = np.zeros((n, n), dtype=np.float32)

    spec_tag = os.path.splitext(os.path.basename(args.spec))[0]
    run_dir = build_run_dir(args.runs_root, bench, args.strategy, tag=args.tag or spec_tag)
    print(f"[run_dir] {run_dir}")

    # --- NEW: work with full env list ---
    env_list = list(envs.values())
    first_env = env_list[0]

    obs_shape = first_env.observation_space.shape if hasattr(first_env.observation_space, 'shape') else (
        first_env.observation_space.n,
    )

    # 🔴 MODEL HEAD SIZE = MAX ACTIONS OVER ALL TASKS
    n_actions = max(e.action_space.n for e in env_list)

    strategy = NaiveStrategy(obs_shape, n_actions, args.seq_len,
                             args.device) if args.strategy == 'naive' else CumulativeReplayStrategy(obs_shape,
                                                                                                    n_actions,
                                                                                                    args.seq_len,
                                                                                                    args.device) if args.strategy == 'cumulative' else EWCStrategy(
        obs_shape, n_actions, args.seq_len, args.device)

    for i, (name, env) in enumerate(envs.items()):
        print(f"[Task {i + 1}/{n}] {name}")
        onpol = collect_trajectories(env, strategy.model, n_episodes=args.collect_episodes,
                                     max_len=1000, target_return=1.0, device=args.device)
        traj_path = None
        if args.dump_trajs:
            traj_path = os.path.join(run_dir, "gen", f"trajs_task{i}.pkl")
            save_trajs_pickle(traj_path, onpol)
        save_task_gen_json(run_dir, i + 1, {
            "step": i + 1, "task_name": name,
            "n_trajectories": len(onpol),
            "traj_file": traj_path
        })
        strategy.train_task(onpol, steps=args.steps_per_task, batch_size=64)
        strategy.after_task(onpol)
        for j, (n2, env2) in enumerate(envs.items()):
            P[i, j] = evaluate_dt(strategy, env2, args.episodes_eval, args.device)

    results = BenchmarkResults(name=f"DT-{args.strategy}:{args.spec}", task_names=task_names, perf_matrix=P)
    metrics = StandardCLMetrics.compute(results)
    save_json(os.path.join(run_dir, "results.json"), {
        "name": results.name, "task_names": task_names,
        "perf_matrix": P.tolist(), "metrics": metrics
    })
    save_matrix_csv(os.path.join(run_dir, "matrix.csv"), task_names, P)
    steps = per_step_report(task_names, P)
    save_json(os.path.join(run_dir, "per_step.json"), {"per_step": steps})
    with open(os.path.join(run_dir, "per_step.csv"), "w", encoding="utf-8", newline="") as f:
        import csv;
        w = csv.DictWriter(f, fieldnames=list(steps[0].keys()));
        w.writeheader();
        w.writerows(steps)
    save_json(os.path.join(run_dir, f"rez_{bench_short(bench)}.json"), {"metrics": metrics, "strategy": args.strategy})


if __name__ == "__main__":
    main()
