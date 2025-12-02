#!/usr/bin/env python
from __future__ import annotations
import argparse
import os
import numpy as np

import torch

from clbench.benchmark.metrics_extra import per_step_report
from clbench.benchmark.runner import BenchmarkResults
from clbench.io.run_logger import bench_short, build_run_dir, save_json, save_matrix_csv
from dt.dataset import Trajectory
from dt.dataset_panda import load_panda_offline_pkl, make_minibatches_panda
from dt.panda_dt import PandaDecisionTransformer
from dt.utils import evaluate_dt_panda

# CL-style logging utilities
from clbench.benchmark.metrics import StandardCLMetrics


# Task definitions (env + dataset) – same as in clb-run-dt-panda.py
PANDA_TASKS = {
    "PandaReach": {
        "env_id": "PandaReachDense-v3",
        "dataset": "resources/datasets/panda_reach_dense_1m_expert.pkl",
    },
    "PandaPush": {
        "env_id": "PandaPushDense-v3",
        "dataset": "resources/datasets/panda_push_dense_1m_expert.pkl",
    },
    "PandaPickAndPlace": {
        "env_id": "PandaPickAndPlaceDense-v3",
        "dataset": "resources/datasets/panda_pick_and_place_dense_1m_expert.pkl",
    },
}


class DummyStrategy:
    """
    Wrapper so we can reuse evaluate_dt_panda(strategy, env, ...),
    which expects strategy.model.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model


def train_single_panda_task(
    trajs: list[Trajectory],
    obs_dim: int,
    act_dim: int,
    seq_len: int,
    device: str,
    steps: int,
    batch_size: int,
):
    """
    Offline training for a single Panda task on its own dataset
    (no continual, no replay).
    """
    device_t = torch.device(device)

    model = PandaDecisionTransformer(
        obs_dim=obs_dim,
        act_dim=act_dim,
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

    loader = make_minibatches_panda(
        trajs,
        seq_len=seq_len,
        batch_size=batch_size,
        device=device_t.type,
        act_dim=act_dim,
        obs_dim=obs_dim,
    )

    model.train()
    for step in range(steps):
        obs, actions, rtg, ts, mask = next(loader)

        # Previous actions: shifted by one; first = zeros
        prev_actions = torch.roll(actions, shifts=1, dims=1)
        prev_actions[:, 0, :] = 0.0

        pred = model(obs, prev_actions, rtg, ts)  # [B, L, act_dim]

        mse_per_step = ((pred - actions) ** 2).mean(dim=-1)  # [B, L]
        valid = mask.sum()
        if valid.item() > 0:
            loss = (mse_per_step * mask).sum() / valid
        else:
            loss = mse_per_step.mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % max(1, steps // 10) == 0:
            print(f"[panda train] step {step+1}/{steps}, loss={loss.item():.4f}")

    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps-per-task", type=int, default=2000)
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--episodes-eval", type=int, default=5)
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--gamma", type=float, default=1.0)

    # CL-style logging
    p.add_argument("--runs-root", type=str, default="runs")
    p.add_argument("--tag", type=str, default="")

    args = p.parse_args()

    device = args.device
    print(f"[device] using {device}")

    import gymnasium as gym  # panda_gym uses gymnasium API
    import panda_gym  # ensure envs are registered

    # ---- Run dir like in continual, but strategy="single" ----
    bench = "panda"
    bench_s = bench_short(bench)
    # nie mamy spec pliku, więc użyjemy prostego tagu "panda_single" (lub nadpisanego przez --tag)
    spec_tag = "panda_single"
    run_dir = build_run_dir(
        args.runs_root,
        bench,
        strategy="single",
        tag=args.tag or spec_tag,
    )
    print(f"[run_dir] {run_dir}")

    task_names = list(PANDA_TASKS.keys())
    n_tasks = len(task_names)
    scores = {}

    for i, (name, cfg) in enumerate(PANDA_TASKS.items(), start=1):
        print("\n==============================")
        print(f"[Panda Single-task {i}/{len(PANDA_TASKS)}] {name}")
        print("==============================")

        env_id = cfg["env_id"]
        dataset_path = cfg["dataset"]
        print(f"[PANDA TASK] {name} / env={env_id} / data={dataset_path}")

        # Load offline dataset for this task
        trajs = load_panda_offline_pkl(dataset_path, gamma=args.gamma)
        if not trajs:
            raise RuntimeError(f"No trajectories loaded from {dataset_path}")

        # Infer obs_dim and act_dim from dataset
        obs_dim = trajs[0].obs.shape[1]
        act_dim = trajs[0].actions.shape[1]
        print(f"[dataset] obs_dim={obs_dim}, act_dim={act_dim}, n_trajs={len(trajs)}")

        # Create online env for evaluation
        env = gym.make(env_id, render_mode=None)

        # Train DT offline on this dataset only
        model = train_single_panda_task(
            trajs=trajs,
            obs_dim=obs_dim,
            act_dim=act_dim,
            seq_len=args.seq_len,
            device=device,
            steps=args.steps_per_task,
            batch_size=args.batch_size,
        )

        # Evaluate policy in the real env (only on its own task)
        score = evaluate_dt_panda(
            DummyStrategy(model),
            env,
            episodes=args.episodes_eval,
            device=device,
        )
        scores[name] = float(score)
        print(f"[eval] Single-task Panda DT on {name}: {score:.3f}")

    print("\n=== Single-task Panda DT results ===")
    for name, sc in scores.items():
        print(f"{name}: {sc:.3f}")

    # ---- Build CL-style perf matrix P (diagonal = single-task performance) ----
    P = np.zeros((n_tasks, n_tasks), dtype=np.float32)
    for i, name in enumerate(task_names):
        P[i, i] = scores.get(name, 0.0)

    # CL-style metrics object
    results_obj = BenchmarkResults(
        name=f"DT-single-panda",
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
            "mode": "single-task",
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

    # Short summary JSON like in CL runs
    save_json(
        os.path.join(run_dir, f"rez_{bench_s}.json"),
        {"metrics": metrics, "strategy": "single"},
    )

    print(f"\n[artifacts] saved to: {run_dir}")


if __name__ == "__main__":
    main()
