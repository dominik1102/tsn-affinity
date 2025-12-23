#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

import ale_py
import gymnasium as gym
# opcjonalnie (w Gymnasium v1.0 to jest "no-op", ale pomaga IDE i jest w docs):
gym.register_envs(ale_py)

from dt.model import DecisionTransformer
from dt.utils import (
    make_minari_atari_env,
    load_npz_dataset_for_task,
    make_offline_minibatches,
    debug_replay_episode,
    evaluate_dt,
)


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


def train_offline_dt(
    model: DecisionTransformer,
    episodes_obs,
    episodes_actions,
    episodes_rewards,
    steps: int,
    batch_size: int,
    device: torch.device,
    seq_len: int,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = make_offline_minibatches(
        episodes_obs, episodes_actions, episodes_rewards,
        seq_len=seq_len, batch_size=batch_size, device=device
    )

    model.train()
    for step in range(int(steps)):
        obs, actions, rtg, ts, mask = next(loader)

        logits = model(obs, actions, rtg, ts, attention_mask=mask)  # [B,L,A]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            actions.reshape(-1),
            ignore_index=-1,
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % max(1, int(steps) // 10) == 0:
            print(f"[train] step {step+1}/{steps}, loss={loss.item():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="JSON list of tasks (name, seed, params.game)")
    ap.add_argument("--dataset-root", default="resources/atari_expert")
    ap.add_argument("--dataset-file", default="expert_minari_dqn.npz")

    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=64)

    ap.add_argument("--episodes-eval", type=int, default=10)
    ap.add_argument("--max-ep-len", type=int, default=27000)

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--p-drop", type=float, default=0.1)

    ap.add_argument("--min-episode-return", type=float, default=None)

    ap.add_argument("--target-mode", choices=["max", "p90", "mean"], default="max")
    ap.add_argument("--target-return", type=float, default=None)

    ap.add_argument("--debug-replay", action="store_true")
    ap.add_argument("--auto-fire", action="store_true")
    ap.add_argument("--auto-fire-on-life-loss", action="store_true")

    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[device] {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    tasks = json.loads(open(args.spec, "r", encoding="utf-8").read())
    assert isinstance(tasks, list) and len(tasks) > 0

    results: Dict[str, float] = {}

    for i, task in enumerate(tasks, start=1):
        name = task["name"]
        seed = int(task.get("seed", i))
        env_id = task["params"]["game"]
        frame_stack = int(task["params"].get("frame_stack", 4))
        clip_rewards = bool(task["params"].get("clip_rewards", True))

        print(f"\n[Single-task {i}/{len(tasks)}] {name} ({env_id})")

        # seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # env zgodny z Minari expert-v0
        env = make_minari_atari_env(
            env_id=env_id,
            seed=seed,
            frame_stack=frame_stack,
            dqn_size=84,
            clip_rewards=clip_rewards,
        )

        # dataset path
        npz_path = os.path.join(args.dataset_root, name, args.dataset_file)
        print(f"[data] loading {npz_path}")
        episodes_obs, episodes_actions, episodes_rewards, ep_returns = load_npz_dataset_for_task(npz_path)

        print(f"[data] episodes={len(episodes_obs)}, mean={ep_returns.mean():.1f}, min={ep_returns.min():.1f}, max={ep_returns.max():.1f}")
        sample = episodes_obs[0][0]
        print(f"[data] obs dtype={sample.dtype}, min={float(sample.min()):.3f}, max={float(sample.max()):.3f}, shape={sample.shape}")

        # optional filter
        if args.min_episode_return is not None:
            thr = float(args.min_episode_return)
            keep = ep_returns >= thr
            episodes_obs = [ep for ep, k in zip(episodes_obs, keep) if k]
            episodes_actions = [ep for ep, k in zip(episodes_actions, keep) if k]
            episodes_rewards = [ep for ep, k in zip(episodes_rewards, keep) if k]
            ep_returns = ep_returns[keep]
            print(f"[data] filtered return>={thr}: kept {keep.sum()}/{len(keep)}")
            print(f"[data] new mean={ep_returns.mean():.1f} min={ep_returns.min():.1f} max={ep_returns.max():.1f}")

        # debug replay: MUST be without auto-fire
        if args.debug_replay:
            ds_ret = float(np.sum(episodes_rewards[0]))
            env_ret = debug_replay_episode(env, episodes_actions[0], dataset_return=ds_ret, seed=seed)
            print(f"[debug] replay expert episode[0]: dataset_return={ds_ret:.2f} env_return={env_ret:.2f}")

        obs_shape = env.observation_space.shape
        n_actions = env.action_space.n

        max_ep_len_embed = max(10000, max(len(r) for r in episodes_rewards) + 1)

        model = DecisionTransformer(
            obs_shape=obs_shape,
            n_actions=n_actions,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            seq_len=args.seq_len,
            p_drop=args.p_drop,
            max_ep_len=max_ep_len_embed,
        ).to(device)

        train_offline_dt(
            model=model,
            episodes_obs=episodes_obs,
            episodes_actions=episodes_actions,
            episodes_rewards=episodes_rewards,
            steps=args.steps,
            batch_size=args.batch_size,
            device=device,
            seq_len=args.seq_len,
        )

        model.eval()

        if args.target_return is not None:
            target = float(args.target_return)
        else:
            target = pick_target_return(ep_returns, args.target_mode)

        avg_ret = evaluate_dt(
            model=model,
            env=env,
            episodes=args.episodes_eval,
            device=device,
            max_steps=args.max_ep_len,
            target_return=target,
            auto_fire=bool(args.auto_fire),
            auto_fire_on_life_loss=bool(args.auto_fire_on_life_loss),
        )

        results[name] = float(avg_ret)
        print(f"[eval] {name}: avg_return={avg_ret:.3f} (target_return={target:.2f})")

        try:
            env.close()
        except Exception:
            pass

    print("\n=== Single-task Minari Atari DT results ===")
    for k, v in results.items():
        print(f"{k}: {v:.3f}")


if __name__ == "__main__":
    main()
