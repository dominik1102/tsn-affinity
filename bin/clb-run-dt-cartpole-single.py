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
from dt.utils import evaluate_dt  # we no longer need collect_trajectories


TaskRegistry.register("cartpole", CartPoleAdapter())


class DummyStrategy:
    """
    Minimal wrapper so we can reuse evaluate_dt(strategy, env, ...),
    which expects strategy.model.
    """
    def __init__(self, model: torch.nn.Module):
        self.model = model


# ====== Offline dataset loading helpers ======


def load_npz_dataset_for_task(dataset_root: str, task_name: str):
    """
    Load trajectories for a given task from a .npz file produced by train_cartpole_expert.py.

    Expected structure:
        dataset_root/
          task_name/
            *.npz   (e.g. expert_trajs_p1.00.npz)

    Inside the .npz:
        observations:    [N, obs_dim]
        actions:         [N]
        rewards:         [N]
        dones:           [N]
        episode_lengths: [n_episodes]
    """
    task_dir = os.path.join(dataset_root, task_name)
    if not os.path.isdir(task_dir):
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    # pick any .npz (e.g. expert_trajs_p1.00.npz, expert_trajs_p0.70.npz, ...)
    candidates = [f for f in os.listdir(task_dir) if f.endswith(".npz")]
    if not candidates:
        raise FileNotFoundError(f"No .npz trajectory files found in {task_dir}")
    candidates.sort()
    npz_path = os.path.join(task_dir, candidates[-1])  # last one, e.g. highest prob, newest, etc.

    print(f"[data] loading trajectories from {npz_path}")
    data = np.load(npz_path)

    observations = data["observations"]        # [N, obs_dim]
    actions = data["actions"]                  # [N]
    rewards = data["rewards"]                  # [N]
    dones = data["dones"]                      # [N]  (not strictly needed)
    episode_lengths = data["episode_lengths"]  # [n_episodes]

    episodes_obs = []
    episodes_actions = []
    episodes_rewards = []

    idx = 0
    for L in episode_lengths:
        L = int(L)
        episodes_obs.append(observations[idx:idx+L])
        episodes_actions.append(actions[idx:idx+L])
        episodes_rewards.append(rewards[idx:idx+L])
        idx += L

    # some simple stats
    returns = [float(np.sum(r)) for r in episodes_rewards]
    print(
        f"[data] episodes={len(episodes_obs)}, "
        f"avg_return={np.mean(returns):.1f}, "
        f"min={np.min(returns):.1f}, "
        f"max={np.max(returns):.1f}"
    )

    return episodes_obs, episodes_actions, episodes_rewards


def make_offline_minibatches(
    episodes_obs,
    episodes_actions,
    episodes_rewards,
    seq_len: int,
    batch_size: int,
    device: torch.device,
):
    """
    Build an infinite generator of minibatches from offline episodes.

    Output per iteration:
        obs:  [B, L, obs_dim]
        actions: [B, L]          (padded with -1)
        rtg:  [B, L, 1]          (returns-to-go)
        ts:   [B, L]             (timesteps within episode)
    """
    num_episodes = len(episodes_obs)
    assert num_episodes > 0, "No episodes in offline dataset"

    obs_dim = episodes_obs[0].shape[-1]

    # Precompute returns-to-go for each episode
    episodes_rtg = []
    episode_lengths = []
    for rew in episodes_rewards:
        r = np.asarray(rew, dtype=np.float32)
        rtg = np.flip(np.cumsum(np.flip(r)))  # rtg[t] = sum_{k=t}^{T-1} r[k]
        episodes_rtg.append(rtg)
        episode_lengths.append(len(r))

    def loader():
        while True:
            obs_batch = np.zeros((batch_size, seq_len, obs_dim), dtype=np.float32)
            actions_batch = np.full((batch_size, seq_len), -1, dtype=np.int64)
            rtg_batch = np.zeros((batch_size, seq_len, 1), dtype=np.float32)
            ts_batch = np.zeros((batch_size, seq_len), dtype=np.int64)

            for b in range(batch_size):
                ep_idx = np.random.randint(num_episodes)
                ep_len = episode_lengths[ep_idx]
                if ep_len == 0:
                    continue

                # random starting index within episode
                start = np.random.randint(0, ep_len)  # inclusive
                end = min(ep_len, start + seq_len)
                length = end - start

                obs_batch[b, :length] = episodes_obs[ep_idx][start:end]
                actions_batch[b, :length] = episodes_actions[ep_idx][start:end]

                rtg_seq = episodes_rtg[ep_idx][start:end]
                rtg_batch[b, :length, 0] = rtg_seq

                ts_batch[b, :length] = np.arange(start, end, dtype=np.int64)

            obs_t = torch.tensor(obs_batch, device=device, dtype=torch.float32)
            actions_t = torch.tensor(actions_batch, device=device, dtype=torch.long)
            rtg_t = torch.tensor(rtg_batch, device=device, dtype=torch.float32)
            ts_t = torch.tensor(ts_batch, device=device, dtype=torch.long)
            yield obs_t, actions_t, rtg_t, ts_t

    return loader()


# ====== Single-task DT training (offline) ======


def train_single_task_dt(
    env,
    obs_shape,
    n_actions: int,
    seq_len: int,
    device: str,
    steps: int,
    batch_size: int,
    dataset_root: str,
    task_name: str,
):
    """
    Simple single-task training loop for CartPole (discrete actions),
    but using OFFLINE trajectories loaded from disk (expert / mixed),
    instead of on-policy data collection.
    """
    device_t = torch.device(device)

    # 1) Load offline trajectories for this task
    episodes_obs, episodes_actions, episodes_rewards = load_npz_dataset_for_task(
        dataset_root, task_name
    )

    loader = make_offline_minibatches(
        episodes_obs,
        episodes_actions,
        episodes_rewards,
        seq_len=seq_len,
        batch_size=batch_size,
        device=device_t,
    )

    # 2) Build DT model
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

    # 3) Offline training from the replay buffer
    model.train()
    for step in range(steps):
        obs, actions, rtg, ts = next(loader)

        # # Previous actions = shifted by 1; first previous = padding (-1)
        # prev_actions = torch.roll(actions, shifts=1, dims=1)
        # prev_actions[:, 0] = -1

        logits = model(obs, actions, rtg, ts)  # [B, L, n_actions]

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
    # collect-episodes / max-len are no longer used (we train from offline data),
    # but we keep the flags for backward compatibility.
    p.add_argument("--collect-episodes", type=int, default=5)
    p.add_argument("--max-len", type=int, default=500, help="Max ep length when evaluating")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # NEW: where expert/mixed trajectories are stored
    p.add_argument(
        "--dataset-root",
        type=str,
        default="data/cartpole_expert",
        help="Root directory with offline trajectories per task "
             "(e.g. output of train_cartpole_expert.py)",
    )

    # Run-dir like in continual version
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

    # Run dir like in clb-run-dt.py, but strategy='single'
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
            dataset_root=args.dataset_root,
            task_name=name,
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

    # ----- Save to runs/ in CL-compatible format -----

    # Performance matrix P: only diagonal = single-task performance
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

    # rez_cp.json — same as in CL
    save_json(
        os.path.join(run_dir, f"rez_{bench_short(bench)}.json"),
        {"metrics": metrics, "strategy": "single"},
    )


if __name__ == "__main__":
    main()
