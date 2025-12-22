#!/usr/bin/env python
from __future__ import annotations

import os
import csv
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch

# Zostawiamy tylko helpery do katalogu + json (nie są metrykami CL)
from clbench.io.run_logger import build_run_dir, save_json

from dt.dataset import Trajectory
from dt.dataset_panda import load_panda_offline_pkl, make_minibatches_panda
from dt.panda_dt import PandaDecisionTransformer
from dt.utils import evaluate_dt_panda


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
    """Wrapper so we can reuse evaluate_dt_panda(strategy, env, ...),
    which expects strategy.model.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model


def _maybe_set_cuda_device(device_t: torch.device) -> None:
    if device_t.type == "cuda" and device_t.index is not None and torch.cuda.is_available():
        torch.cuda.set_device(device_t.index)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_single_panda_task(
    trajs: List[Trajectory],
    obs_dim: int,
    act_dim: int,
    seq_len: int,
    device: str,
    steps: int,
    batch_size: int,
    log_every: int = 200,
) -> Tuple[torch.nn.Module, List[Dict[str, float]]]:
    """Offline training for a single Panda task on its own dataset."""
    device_t = torch.device(device)
    _maybe_set_cuda_device(device_t)

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
        device=device_t.type,  # "cpu" / "cuda"
        act_dim=act_dim,
        obs_dim=obs_dim,
    )

    train_log: List[Dict[str, float]] = []

    model.train()
    for step in range(steps):
        try:
            obs, actions, rtg, ts, mask = next(loader)
        except StopIteration:
            # jeśli loader jest skończony, restart
            loader = make_minibatches_panda(
                trajs,
                seq_len=seq_len,
                batch_size=batch_size,
                device=device_t.type,
                act_dim=act_dim,
                obs_dim=obs_dim,
            )
            obs, actions, rtg, ts, mask = next(loader)

        # Previous actions: shifted by one; first = zeros
        prev_actions = torch.roll(actions, shifts=1, dims=1)
        prev_actions[:, 0, :] = 0.0

        pred = model(obs, prev_actions, rtg, ts)  # [B, L, act_dim]
        mse_per_step = ((pred - actions) ** 2).mean(dim=-1)  # [B, L]

        mask_f = mask.float()
        valid = mask_f.sum()
        numer = (mse_per_step * mask_f).sum()
        loss_masked = numer / (valid + 1e-8)
        loss = torch.where(valid > 0, loss_masked, mse_per_step.mean())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if log_every > 0 and ((step + 1) % log_every == 0 or step == 0):
            # logujemy jawnie (to jest sync CPU, ale rzadko)
            lval = float(loss.detach().cpu().item())
            train_log.append({"step": float(step + 1), "loss": lval})
            print(f"[panda train] step {step+1}/{steps}, loss={lval:.6f}")

    return model, train_log


def _write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="all", choices=["all", *PANDA_TASKS.keys()])

    p.add_argument("--steps-per-task", type=int, default=2000)
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--episodes-eval", type=int, default=5)
    p.add_argument("--log-every", type=int, default=200)

    p.add_argument("--seed", type=int, default=0)

    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--gamma", type=float, default=1.0)

    # tylko katalog wyników (bez CL metryk)
    p.add_argument("--runs-root", type=str, default="runs")
    p.add_argument("--tag", type=str, default="")

    args = p.parse_args()
    _set_seed(args.seed)

    device_t = torch.device(args.device)
    _maybe_set_cuda_device(device_t)
    print(f"[device] using {device_t}")

    import gymnasium as gym
    import panda_gym  # noqa: F401

    # run_dir (dalej możesz używać tej samej struktury katalogów)
    bench = "panda"
    spec_tag = f"panda_single_{args.task}"
    run_dir = build_run_dir(
        args.runs_root,
        bench,
        strategy="single_no_clmetrics",
        tag=args.tag or spec_tag,
    )
    print(f"[run_dir] {run_dir}")

    # wybór zadań
    if args.task == "all":
        tasks = list(PANDA_TASKS.items())
    else:
        tasks = [(args.task, PANDA_TASKS[args.task])]

    scores: Dict[str, float] = {}

    for i, (name, cfg) in enumerate(tasks, start=1):
        print("\n==============================")
        print(f"[Panda Single-task {i}/{len(tasks)}] {name}")
        print("==============================")

        env_id = cfg["env_id"]
        dataset_path = cfg["dataset"]
        print(f"[PANDA TASK] {name} / env={env_id} / data={dataset_path}")

        # Load offline dataset
        trajs = load_panda_offline_pkl(dataset_path, gamma=args.gamma)
        if not trajs:
            raise RuntimeError(f"No trajectories loaded from {dataset_path}")

        obs_dim = trajs[0].obs.shape[-1]
        act_dim = trajs[0].actions.shape[-1]
        print(f"[dataset] obs_dim={obs_dim}, act_dim={act_dim}, n_trajs={len(trajs)}")

        # subdir na artefakty per task
        task_dir = os.path.join(run_dir, name)
        os.makedirs(task_dir, exist_ok=True)

        # env do ewaluacji
        env = gym.make(env_id, render_mode=None)
        try:
            # (opcjonalne) ustaw seed env
            try:
                env.reset(seed=args.seed)
            except TypeError:
                pass

            model, train_log = train_single_panda_task(
                trajs=trajs,
                obs_dim=obs_dim,
                act_dim=act_dim,
                seq_len=args.seq_len,
                device=str(device_t),
                steps=args.steps_per_task,
                batch_size=args.batch_size,
                log_every=args.log_every,
            )

            # zapisz log treningu (loss vs step)
            if train_log:
                _write_csv(
                    os.path.join(task_dir, "train_log.csv"),
                    train_log,
                    fieldnames=["step", "loss"],
                )

            # evaluate w środowisku
            model.eval()
            with torch.no_grad():
                score = evaluate_dt_panda(
                    DummyStrategy(model),
                    env,
                    episodes=args.episodes_eval,
                    device=str(device_t),
                )

            scores[name] = float(score)
            print(f"[eval] Single-task Panda DT on {name}: {score:.3f}")

            # results per-task
            save_json(
                os.path.join(task_dir, "task_results.json"),
                {
                    "task": name,
                    "env_id": env_id,
                    "dataset": dataset_path,
                    "score": float(score),
                    "obs_dim": int(obs_dim),
                    "act_dim": int(act_dim),
                    "n_trajs": int(len(trajs)),
                    "steps_per_task": int(args.steps_per_task),
                    "seq_len": int(args.seq_len),
                    "batch_size": int(args.batch_size),
                    "episodes_eval": int(args.episodes_eval),
                    "gamma": float(args.gamma),
                    "seed": int(args.seed),
                    "device": str(device_t),
                },
            )
        finally:
            try:
                env.close()
            except Exception:
                pass

            # free memory
            try:
                del model
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # podsumowanie
    print("\n=== Single-task Panda DT results ===")
    for name, sc in scores.items():
        print(f"{name}: {sc:.3f}")

    avg_score = float(np.mean(list(scores.values()))) if scores else 0.0

    # zapis summary
    save_json(
        os.path.join(run_dir, "results.json"),
        {
            "mode": "single-task",
            "tasks_ran": list(scores.keys()),
            "scores": scores,
            "avg_score": avg_score,
            "device": str(device_t),
            "steps_per_task": args.steps_per_task,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "episodes_eval": args.episodes_eval,
            "gamma": args.gamma,
            "seed": args.seed,
        },
    )

    # prosty CSV z wynikami
    rows = [{"task": k, "score": v} for k, v in scores.items()]
    if rows:
        _write_csv(os.path.join(run_dir, "scores.csv"), rows, fieldnames=["task", "score"])

    print(f"\n[artifacts] saved to: {run_dir}")


if __name__ == "__main__":
    main()
