#!/usr/bin/env python3
from __future__ import annotations

import os
import csv
import argparse
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch

import gymnasium as gym
import panda_gym  # noqa: F401 (register Panda envs)
from gymnasium.wrappers import FlattenObservation

from bin.config import PANDA_TASKS
from clbench.io.run_logger import build_run_dir, save_json

from dt.dataset import Trajectory
from dt.dataset_panda import load_panda_offline_pkl, make_minibatches_panda
from dt.panda_dt import PandaDecisionTransformer
from dt.utils import evaluate_dt_panda


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _maybe_set_cuda_device(device_t: torch.device) -> None:
    if device_t.type == "cuda" and device_t.index is not None and torch.cuda.is_available():
        torch.cuda.set_device(device_t.index)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


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


def ensure_traj_list(trajs_raw: Any) -> List[Trajectory]:
    if isinstance(trajs_raw, dict):
        trajs_raw = list(trajs_raw.values())
    if not isinstance(trajs_raw, (list, tuple)):
        raise TypeError(f"Expected list/tuple/dict from loader, got {type(trajs_raw)}")
    trajs: List[Trajectory] = []
    for tr in trajs_raw:
        if isinstance(tr, Trajectory):
            trajs.append(tr)
        elif isinstance(tr, dict):
            obs = np.asarray(tr["obs"], dtype=np.float32)
            actions = np.asarray(tr["actions"], dtype=np.float32)
            rewards = np.asarray(tr["rewards"], dtype=np.float32)
            if "timesteps" in tr:
                ts = np.asarray(tr["timesteps"], dtype=np.int64)
            else:
                ts = np.arange(actions.shape[0], dtype=np.int64)
            if "returns_to_go" in tr:
                rtg = np.asarray(tr["returns_to_go"], dtype=np.float32)
            else:
                r = rewards.astype(np.float32)
                rtg = np.flip(np.cumsum(np.flip(r, axis=0), axis=0), axis=0).astype(np.float32)

            trajs.append(Trajectory(obs=obs, actions=actions, rewards=rewards, timesteps=ts, returns_to_go=rtg))
        else:
            raise TypeError(f"Unsupported trajectory type: {type(tr)}")
    return trajs


def make_panda_env(env_id: str) -> gym.Env:
    try:
        env = gym.make(env_id, render_mode="rgb_array")
    except TypeError:
        env = gym.make(env_id, render="rgb_array")

    if isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)
    return env


def compute_mean_std(trajs: List[Trajectory], obs_dim: int):
    """
    Mean/std po wszystkich obserwacjach z datasetu (bez paddingów),
    w stylu Twojego starego projektu.
    """
    count = 0
    mean = np.zeros(obs_dim, dtype=np.float64)
    M2 = np.zeros(obs_dim, dtype=np.float64)

    for tr in trajs:
        x = np.asarray(tr.obs, dtype=np.float64).reshape(-1, obs_dim)
        if x.size == 0:
            continue
        bcount = x.shape[0]
        bmean = x.mean(axis=0)
        bvar = x.var(axis=0)

        if count == 0:
            mean = bmean
            M2 = bvar * bcount
            count = bcount
        else:
            delta = bmean - mean
            tot = count + bcount
            mean = mean + delta * (bcount / tot)
            M2 = M2 + bvar * bcount + (delta ** 2) * (count * bcount / tot)
            count = tot

    var = M2 / max(count, 1)
    std = np.sqrt(var) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


# ------------------------------------------------------------
# Training (single task offline)
# ------------------------------------------------------------
def train_single_panda_task(
    trajs: List[Trajectory],
    obs_dim: int,
    act_dim: int,
    seq_len: int,
    device_t: torch.device,
    steps: int,
    batch_size: int,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    rtg_scale: float,
    log_every: int = 200,
) -> Tuple[torch.nn.Module, List[Dict[str, float]]]:
    _maybe_set_cuda_device(device_t)

    model = PandaDecisionTransformer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        d_model=128,
        n_layers=3,
        n_heads=4,
        seq_len=seq_len,
        p_drop=0.1,
        # NEW:
        obs_mean=obs_mean,
        obs_std=obs_std,
        rtg_scale=float(rtg_scale),
        act_tanh=False,  # możesz potem przetestować True
        max_ep_len=2048,
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
        device=device_t.type,  # "cpu"/"cuda"
        act_dim=act_dim,
        obs_dim=obs_dim,
    )

    train_log: List[Dict[str, float]] = []

    model.train()
    for step in range(int(steps)):
        obs, actions, rtg, ts, mask = next(loader)

        mask = mask.to(dtype=torch.bool)

        pred = model(obs, actions, rtg, ts, attention_mask=mask)  # [B, L, act_dim]
        mse_per_step = ((pred - actions) ** 2).mean(dim=-1)       # [B, L]

        mask_f = mask.float()
        valid = mask_f.sum()
        numer = (mse_per_step * mask_f).sum()
        loss_masked = numer / (valid + 1e-8)
        loss = torch.where(valid > 0, loss_masked, mse_per_step.mean())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if log_every > 0 and (((step + 1) % log_every) == 0 or step == 0):
            lval = float(loss.detach().cpu().item())
            train_log.append({"step": float(step + 1), "loss": lval})
            print(f"[panda train] step {step+1}/{steps}, loss={lval:.6f}")

    return model, train_log


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="all", choices=["all", *PANDA_TASKS.keys()])

    p.add_argument("--steps-per-task", type=int, default=2000)
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--episodes-eval", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--log-every", type=int, default=200)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--gamma", type=float, default=1.0)

    # DT conditioning for evaluation
    p.add_argument("--target-mode", choices=["max", "p90", "mean"], default="max")
    p.add_argument("--target-return", type=float, default=None)

    # NEW: rtg_scale
    p.add_argument(
        "--rtg-scale",
        type=float,
        default=None,
        help="If set, overrides auto rtg_scale. If None: rtg_scale = max(1.0, max_abs_return).",
    )

    p.add_argument("--runs-root", type=str, default="runs")
    p.add_argument("--tag", type=str, default="")

    args = p.parse_args()

    _set_seed(args.seed)

    device_t = torch.device(args.device)
    _maybe_set_cuda_device(device_t)
    print(f"[device] using {device_t}")

    bench = "panda"
    spec_tag = f"panda_single_{args.task}"
    run_dir = build_run_dir(
        args.runs_root,
        bench,
        strategy="single_no_clmetrics",
        tag=args.tag or spec_tag,
    )
    print(f"[run_dir] {run_dir}")

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
        print(f"[task] env={env_id} dataset={dataset_path}")

        trajs_raw = load_panda_offline_pkl(dataset_path, gamma=float(args.gamma))
        trajs = ensure_traj_list(trajs_raw)
        if not trajs:
            raise RuntimeError(f"No trajectories loaded from {dataset_path}")

        obs_dim = int(np.asarray(trajs[0].obs).shape[-1])
        act_dim = int(np.asarray(trajs[0].actions).shape[-1])
        rets = np.array([traj_return(t) for t in trajs], dtype=np.float32)

        print(f"[dataset] obs_dim={obs_dim}, act_dim={act_dim}, n_trajs={len(trajs)}")
        print(f"[dataset] return mean={rets.mean():.3f} min={rets.min():.3f} max={rets.max():.3f}")

        # --- NEW: obs normalization (old-project style) ---
        obs_mean, obs_std = compute_mean_std(trajs, obs_dim)

        # --- NEW: rtg_scale (spójnie z Atari) ---
        if args.rtg_scale is not None:
            rtg_scale = float(args.rtg_scale)
        else:
            rtg_scale = float(max(1.0, float(np.max(np.abs(rets))) if rets.size else 1.0))
        print(f"[rtg_scale] {rtg_scale:.6f}")

        # Choose target_return for evaluation (DT conditioning)
        if args.target_return is not None:
            target = float(args.target_return)
        else:
            target = pick_target_return(rets, args.target_mode)
        print(f"[eval] target_return={target:.6f} (mode={args.target_mode})")

        task_dir = os.path.join(run_dir, name)
        os.makedirs(task_dir, exist_ok=True)

        env = make_panda_env(env_id)

        try:
            try:
                env.reset(seed=int(args.seed))
            except TypeError:
                pass

            model, train_log = train_single_panda_task(
                trajs=trajs,
                obs_dim=obs_dim,
                act_dim=act_dim,
                seq_len=args.seq_len,
                device_t=device_t,
                steps=args.steps_per_task,
                batch_size=args.batch_size,
                obs_mean=obs_mean,
                obs_std=obs_std,
                rtg_scale=rtg_scale,
                log_every=args.log_every,
            )

            if train_log:
                _write_csv(
                    os.path.join(task_dir, "train_log.csv"),
                    train_log,
                    fieldnames=["step", "loss"],
                )

            model.eval()

            max_steps = args.max_steps
            if max_steps is None:
                max_steps = getattr(getattr(env, "spec", None), "max_episode_steps", None)
            if max_steps is None:
                max_steps = 200

            score = evaluate_dt_panda(
                model=model,
                env=env,
                episodes=int(args.episodes_eval),
                device=device_t,
                max_steps=int(max_steps),
                target_return=float(target),
                seed=int(args.seed),
                obs_pad_to=obs_dim,
                act_pad_to=act_dim,
                clip_action=True,
            )

            scores[name] = float(score)
            print(f"[eval] Single-task Panda DT on {name}: {score:.3f}")

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
                    "max_steps": int(max_steps),
                    "gamma": float(args.gamma),
                    "target_mode": str(args.target_mode),
                    "target_return": float(target),
                    "seed": int(args.seed),
                    "device": str(device_t),
                    # NEW:
                    "rtg_scale": float(rtg_scale),
                    "obs_mean": obs_mean.tolist(),
                    "obs_std": obs_std.tolist(),
                },
            )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        finally:
            try:
                env.close()
            except Exception:
                pass

    print("\n=== Single-task Panda DT results ===")
    for name, sc in scores.items():
        print(f"{name}: {sc:.3f}")

    avg_score = float(np.mean(list(scores.values()))) if scores else 0.0

    save_json(
        os.path.join(run_dir, "results.json"),
        {
            "mode": "single-task",
            "tasks_ran": list(scores.keys()),
            "scores": scores,
            "avg_score": avg_score,
            "device": str(device_t),
            "steps_per_task": int(args.steps_per_task),
            "seq_len": int(args.seq_len),
            "batch_size": int(args.batch_size),
            "episodes_eval": int(args.episodes_eval),
            "max_steps": args.max_steps,
            "gamma": float(args.gamma),
            "target_mode": str(args.target_mode),
            "target_return_override": args.target_return,
            "seed": int(args.seed),
            # NEW:
            "rtg_scale_override": args.rtg_scale,
        },
    )

    rows = [{"task": k, "score": v} for k, v in scores.items()]
    if rows:
        _write_csv(os.path.join(run_dir, "scores.csv"), rows, fieldnames=["task", "score"])

    print(f"\n[artifacts] saved to: {run_dir}")


if __name__ == "__main__":
    main()
