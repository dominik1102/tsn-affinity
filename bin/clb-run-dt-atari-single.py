#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F

from clbench.adapters.atari import AtariAdapter
from clbench.benchmark.metrics_extra import per_step_report
from clbench.benchmark.runner import make_tasks, describe_tasks, BenchmarkResults
from clbench.core.registry import TaskRegistry
from clbench.io.run_logger import build_run_dir, bench_short, save_json, save_matrix_csv
from clbench.io.serialize import load_task_specs

from clbench.benchmark.metrics import StandardCLMetrics

from dt.model import DecisionTransformer
from dt.dataset import make_minibatches
from dt.utils import collect_trajectories, evaluate_dt

# Ensure Atari adapter is registered in TaskRegistry
TaskRegistry.register("atari", AtariAdapter())


class DummyStrategy:
    """Minimal wrapper so we can reuse evaluate_dt(strategy, env, ...)."""
    def __init__(self, model: torch.nn.Module):
        self.model = model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True, help="JSON spec file, e.g. specs_atari.json")
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--episodes-eval", type=int, default=5)
    p.add_argument("--steps", type=int, default=2000, help="SGD steps per task")
    p.add_argument("--collect-episodes", type=int, default=10)
    p.add_argument(
        "--max-ep-len",
        type=int,
        default=1000,
        help="max length of trajectories collected for training",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--p-drop", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--runs-root", type=str, default="runs")
    p.add_argument("--tag", type=str, default="")
    args = p.parse_args()

    device = args.device

    # ----- Load tasks as in CL, but we'll treat them independently -----
    specs = load_task_specs(args.spec)
    bench = "atari"
    envs = make_tasks(bench, specs)
    print(describe_tasks(envs, bench))

    # Build run directory like in continual scripts (strategy="single")
    spec_tag = os.path.splitext(os.path.basename(args.spec))[0]
    run_dir = build_run_dir(args.runs_root, bench, "single", tag=args.tag or spec_tag)
    print(f"[run_dir] {run_dir}")

    results = {}

    for i, (name, env) in enumerate(envs.items(), start=1):
        print(f"\n[Single-task {i}/{len(envs)}] {name}")

        # --- Build a fresh DT model for THIS env only ---
        obs_shape = env.observation_space.shape
        n_actions = env.action_space.n

        model = DecisionTransformer(
            obs_shape=obs_shape,
            n_actions=n_actions,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            seq_len=args.seq_len,
            p_drop=args.p_drop,
        ).to(device)

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=3e-4,
            weight_decay=1e-4,
        )

        # --- Collect on-policy trajectories from this env only ---
        print(
            f"[collect] env={name} episodes={args.collect_episodes}, "
            f"max_len={args.max_ep_len}"
        )
        trajs = collect_trajectories(
            env,
            model,
            n_episodes=args.collect_episodes,
            max_len=args.max_ep_len,
            target_return=1.0,
            device=device,
        )

        # --- Training loop on this single-task dataset ---
        loader = make_minibatches(
            trajs,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            device=device,
        )

        model.train()
        for step in range(args.steps):
            obs, actions, rtg, ts = next(loader)

            # Shift actions by 1 along time dimension; first prev = -1 (padding)
            prev_actions = torch.roll(actions, shifts=1, dims=1)
            prev_actions[:, 0] = -1

            logits = model(
                obs,
                prev_actions,
                rtg,
                ts,
            )  # [B, L, n_actions]

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                actions.reshape(-1),
                ignore_index=-1,
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if (step + 1) % 500 == 0 or step == 0:
                print(
                    f"[{name}] step {step+1}/{args.steps}, "
                    f"loss={loss.item():.4f}"
                )

        # --- Evaluation only on THIS env ---
        # --- Evaluation only on THIS env ---
        model.eval()
        avg_ret = evaluate_dt(
            DummyStrategy(model),
            env,
            episodes=args.episodes_eval,
            device=device,
        )
        results[name] = float(avg_ret)
        print(
            f"[eval single-task] {name}: avg return over "
            f"{args.episodes_eval} episodes = {avg_ret:.3f}"
        )

    # ----- Save results in the same CL-style structure -----
    task_names = list(envs.keys())
    n = len(task_names)
    P = np.zeros((n, n), dtype=np.float32)

    # Fill only diagonal with single-task scores
    name_to_idx = {name: i for i, name in enumerate(task_names)}
    for name, ret in results.items():
        i = name_to_idx[name]
        P[i, i] = float(ret)

    # CL-style metrics & artifacts (ACC etc. will be computed on this diag matrix)
    bench_name_short = bench_short(bench)
    results_obj = BenchmarkResults(
        name=f"DT-single:{args.spec}",
        task_names=task_names,
        perf_matrix=P,
    )
    metrics = StandardCLMetrics.compute(results_obj)

    # Save main JSON + matrix CSV
    save_json(
        os.path.join(run_dir, "results.json"),
        {
            "name": results_obj.name,
            "task_names": task_names,
            "perf_matrix": P.tolist(),
            "metrics": metrics,
            "mode": "single-task",
        },
    )
    save_matrix_csv(os.path.join(run_dir, "matrix.csv"), task_names, P)

    # Per-step report (here "steps" = tasks, but kept for compatibility)
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

    # Lightweight summary JSON (like in CL runs)
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
