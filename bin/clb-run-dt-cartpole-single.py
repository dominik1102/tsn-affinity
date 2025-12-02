#!/usr/bin/env python
from __future__ import annotations
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from clbench.adapters.cartpole import CartPoleAdapter
from clbench.benchmark.metrics_extra import per_step_report
from clbench.benchmark.runner import make_tasks, describe_tasks, BenchmarkResults
from clbench.core.registry import TaskRegistry
from clbench.io.run_logger import build_run_dir, save_json, save_matrix_csv, bench_short
from clbench.io.serialize import load_task_specs

from clbench.benchmark.metrics import StandardCLMetrics

from dt.model import DecisionTransformer
from dt.dataset import make_minibatches
from dt.utils import collect_trajectories, evaluate_dt


TaskRegistry.register("cartpole", CartPoleAdapter())


class DummyStrategy:
    """
    Minimal wrapper so we can reuse evaluate_dt(strategy, env, ...),
    which expects strategy.model.
    """
    def __init__(self, model: torch.nn.Module):
        self.model = model


def train_single_task_dt(
    env,
    obs_shape,
    n_actions: int,
    seq_len: int,
    device: str,
    steps: int,
    batch_size: int,
    collect_episodes: int,
    max_len: int = 1000,
):
    """
    Simple single-task training loop for CartPole (discrete actions),
    using the same DT and data pipeline as in the continual setup,
    but *without* any replay or multi-task logic.
    """
    device_t = torch.device(device)

    model = DecisionTransformer(
        obs_shape=obs_shape,
        n_actions=n_actions,
        d_model=128,
        n_layers=3,
        n_heads=4,
        seq_len=seq_len,
        p_drop=0.1,
    ).to(device_t)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
    )

    # On-policy trajectories from this env only
    print(f"[collect] episodes={collect_episodes}, max_len={max_len}")
    trajs = collect_trajectories(
        env,
        model,
        n_episodes=collect_episodes,
        max_len=max_len,
        target_return=1.0,
        device=device_t.type,
    )

    loader = make_minibatches(
        trajs,
        seq_len,
        batch_size,
        device_t.type,
    )

    model.train()
    for step in range(steps):
        obs, actions, rtg, ts = next(loader)

        # Previous actions = shifted by 1; first previous = padding (-1)
        prev_actions = torch.roll(actions, shifts=1, dims=1)
        prev_actions[:, 0] = -1

        logits = model(obs, prev_actions, rtg, ts)  # [B, L, n_actions]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            actions.reshape(-1),
            ignore_index=-1,
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % max(1, steps // 10) == 0:
            print(f"[train {env.spec.id}] step {step+1}/{steps}, loss={loss.item():.4f}")

    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True, help="CartPole spec file, e.g. specs_cartpole")
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--episodes-eval", type=int, default=5)
    p.add_argument("--steps-per-task", type=int, default=2000)
    p.add_argument("--collect-episodes", type=int, default=5)
    p.add_argument("--max-len", type=int, default=500, help="Max ep length when collecting trajs")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # NOWE: run-dir jak w continual
    p.add_argument("--runs-root", type=str, default="runs")
    p.add_argument("--tag", type=str, default="")

    args = p.parse_args()

    device = args.device
    print(f"[device] using {device}")

    # Load CartPole tasks from spec (without continual logic)
    specs = load_task_specs(args.spec)
    bench = "cartpole"
    envs = make_tasks(bench, specs)
    print(describe_tasks(envs, bench))

    task_names = list(envs.keys())
    n_tasks = len(task_names)

    # Run dir jak w clb-run-dt.py, ale strategy='single'
    spec_tag = os.path.splitext(os.path.basename(args.spec))[0]
    run_dir = build_run_dir(
        args.runs_root,
        bench,
        strategy="single",
        tag=args.tag or spec_tag,
    )
    print(f"[run_dir] {run_dir}")

    results_scalar = {}

    for i, (name, env) in enumerate(envs.items(), start=1):
        print("\n==============================")
        print(f"[CartPole Single-task {i}/{len(envs)}] {name}")
        print("==============================")

        obs_shape = env.observation_space.shape
        n_actions = env.action_space.n
        print(f"[env] obs_shape={obs_shape}, n_actions={n_actions}")

        model = train_single_task_dt(
            env=env,
            obs_shape=obs_shape,
            n_actions=n_actions,
            seq_len=args.seq_len,
            device=device,
            steps=args.steps_per_task,
            batch_size=64,
            collect_episodes=args.collect_episodes,
            max_len=args.max_len,
        )

        score = evaluate_dt(
            DummyStrategy(model),
            env,
            episodes=args.episodes_eval,
            device=device,
            max_steps=args.max_len,
        )
        results_scalar[name] = float(score)
        print(f"[eval] Single-task DT on {name}: {score:.3f}")

    print("\n=== Single-task CartPole DT results ===")
    for name, r in results_scalar.items():
        print(f"{name}: {r:.3f}")

    # ----- Zapis do runs/ w formacie kompatybilnym z CL -----

    # Macierz P: tylko przekątna = single-task performance
    P = np.zeros((n_tasks, n_tasks), dtype=np.float32)
    for i, name in enumerate(task_names):
        P[i, i] = float(results_scalar.get(name, 0.0))

    results_obj = BenchmarkResults(
        name=f"DT-single:{args.spec}",
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

    # rez_cp.json — tak jak w CL
    save_json(
        os.path.join(run_dir, f"rez_{bench_short(bench)}.json"),
        {"metrics": metrics, "strategy": "single"},
    )


if __name__ == "__main__":
    main()
