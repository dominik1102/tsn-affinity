#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import numpy as np

# Reuse exactly the same DQN preprocessing functions as the current runner/env wrapper.
# This file must be run from the project root so dt.utils is importable.
from dt.utils import _preprocess_frame_dqn_hw01  # type: ignore


def _stack_observations(raw_obs: np.ndarray, frame_stack: int = 4, dqn_size: int = 84) -> np.ndarray:
    """Convert raw RGB Atari observations to stacked CHW float32 observations.

    Minari Atari observations are raw RGB frames, usually length T+1, where T is the
    number of actions. We produce one stacked observation per action, i.e. T stacks.
    The first stack repeats the reset frame exactly as MinariDQNStackWrapper does.
    """
    raw_obs = np.asarray(raw_obs)
    frames = [_preprocess_frame_dqn_hw01(frame, dqn_size=dqn_size) for frame in raw_obs]
    if not frames:
        return np.zeros((0, frame_stack, dqn_size, dqn_size), dtype=np.float32)

    from collections import deque
    dq = deque(maxlen=frame_stack)
    # initial reset stack = first frame repeated
    for _ in range(frame_stack):
        dq.append(frames[0].copy())

    stacks: List[np.ndarray] = []
    # stack before each action: at i=0, repeated reset frame; then include obs[i]
    for i in range(len(frames) - 1):
        if i > 0:
            dq.append(frames[i])
        stacks.append(np.stack(list(dq), axis=0).astype(np.float32))
    return np.stack(stacks, axis=0).astype(np.float32) if stacks else np.zeros((0, frame_stack, dqn_size, dqn_size), dtype=np.float32)


def _episodes(dataset: Any) -> Iterable[Any]:
    if hasattr(dataset, "iterate_episodes"):
        return dataset.iterate_episodes()
    if hasattr(dataset, "sample_episodes"):
        n = int(getattr(dataset, "total_episodes", 0) or 0)
        if n <= 0:
            n = 10
        return dataset.sample_episodes(n)
    raise AttributeError("Unsupported Minari dataset object: no iterate_episodes or sample_episodes")


def _get_field(ep: Any, name: str) -> Any:
    if hasattr(ep, name):
        return getattr(ep, name)
    if isinstance(ep, dict) and name in ep:
        return ep[name]
    raise AttributeError(f"Episode has no field {name!r}: {type(ep)}")


def export_one(dataset_id: str, out_dir: Path, *, frame_stack: int, dqn_size: int, clip_rewards: bool, max_episodes: int | None) -> None:
    import minari

    print(f"[load] {dataset_id}")
    dataset = minari.load_dataset(dataset_id, download=False)

    obs_all: List[np.ndarray] = []
    act_all: List[np.ndarray] = []
    rew_all: List[np.ndarray] = []
    done_all: List[np.ndarray] = []
    lengths: List[int] = []

    for ep_idx, ep in enumerate(_episodes(dataset)):
        if max_episodes is not None and ep_idx >= max_episodes:
            break
        raw_obs = np.asarray(_get_field(ep, "observations"))
        actions = np.asarray(_get_field(ep, "actions"), dtype=np.int64).reshape(-1)
        rewards = np.asarray(_get_field(ep, "rewards"), dtype=np.float32).reshape(-1)
        terms = np.asarray(_get_field(ep, "terminations"), dtype=bool).reshape(-1)
        truncs = np.asarray(_get_field(ep, "truncations"), dtype=bool).reshape(-1)

        T = int(actions.shape[0])
        if raw_obs.shape[0] == T:
            # Some formats may omit the final next observation. Add a dummy copy only
            # for stack construction; output still has T observations.
            raw_obs_for_stack = np.concatenate([raw_obs, raw_obs[-1:]], axis=0)
        elif raw_obs.shape[0] == T + 1:
            raw_obs_for_stack = raw_obs
        else:
            raise ValueError(f"Unexpected obs/actions length for {dataset_id}: obs={raw_obs.shape[0]} actions={T}")

        obs_stacked = _stack_observations(raw_obs_for_stack, frame_stack=frame_stack, dqn_size=dqn_size)
        obs_stacked = obs_stacked[:T]
        if clip_rewards:
            rewards = np.clip(rewards, -1.0, 1.0).astype(np.float32)
        dones = np.logical_or(terms, truncs).astype(np.bool_)
        if dones.shape[0] != T:
            dones = np.zeros((T,), dtype=np.bool_)
            dones[-1] = True
        else:
            dones[-1] = True

        obs_all.append(obs_stacked.astype(np.float32))
        act_all.append(actions.astype(np.int64))
        rew_all.append(rewards.astype(np.float32))
        done_all.append(dones.astype(np.bool_))
        lengths.append(T)
        print(f"  ep={ep_idx:02d} len={T} return={float(rewards.sum()):.3f}")

    if not lengths:
        raise RuntimeError(f"No episodes exported for {dataset_id}")

    observations = np.concatenate(obs_all, axis=0).astype(np.float32)
    actions = np.concatenate(act_all, axis=0).astype(np.int64)
    rewards = np.concatenate(rew_all, axis=0).astype(np.float32)
    dones = np.concatenate(done_all, axis=0).astype(np.bool_)
    episode_lengths = np.asarray(lengths, dtype=np.int64)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "expert_minari_dqn.npz"
    np.savez_compressed(
        out_path,
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
        episode_lengths=episode_lengths,
    )
    returns = [float(x.sum()) for x in rew_all]
    print(f"[saved] {out_path}")
    print(f"        observations={observations.shape} actions={actions.shape} rewards={rewards.shape} episodes={len(lengths)}")
    print(f"        returns mean={np.mean(returns):.3f} min={np.min(returns):.3f} max={np.max(returns):.3f}")


TASKS: List[Tuple[str, str, str]] = [
    ("A_Alien", "atari/alien/expert-v0", "ALE/Alien-v5"),
    ("B_Atlantis", "atari/atlantis/expert-v0", "ALE/Atlantis-v5"),
    ("C_Boxing", "atari/boxing/expert-v0", "ALE/Boxing-v5"),
    ("D_Breakout", "atari/breakout/expert-v0", "ALE/Breakout-v5"),
    ("E_Centipede", "atari/centipede/expert-v0", "ALE/Centipede-v5"),
    ("F_DoubleDunk", "atari/doubledunk/expert-v0", "ALE/DoubleDunk-v5"),
    ("G_Freeway", "atari/freeway/expert-v0", "ALE/Freeway-v5"),
    ("H_Pong", "atari/pong/expert-v0", "ALE/Pong-v5"),
    ("I_SpaceInvaders", "atari/spaceinvaders/expert-v0", "ALE/SpaceInvaders-v5"),
    ("J_Tennis", "atari/tennis/expert-v0", "ALE/Tennis-v5"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, help="Output root, e.g. /net/tscratch/.../datasets/atari_expert")
    ap.add_argument("--only-missing", action="store_true")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--frame-stack", type=int, default=4)
    ap.add_argument("--dqn-size", type=int, default=84)
    ap.add_argument("--no-clip-rewards", action="store_true")
    ap.add_argument("--tasks", nargs="*", default=["F_DoubleDunk", "G_Freeway", "H_Pong", "I_SpaceInvaders", "J_Tennis"],
                    help="Task directories to export; default exports missing CL-10 tasks F-J")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    task_map = {name: (dsid, env) for name, dsid, env in TASKS}
    for task_name in args.tasks:
        if task_name not in task_map:
            raise KeyError(f"Unknown task {task_name}. Known: {list(task_map)}")
        dsid, _ = task_map[task_name]
        out_dir = root / task_name
        out_path = out_dir / "expert_minari_dqn.npz"
        if args.only_missing and out_path.exists():
            print(f"[skip] {task_name}: exists {out_path}")
            continue
        export_one(
            dsid,
            out_dir,
            frame_stack=args.frame_stack,
            dqn_size=args.dqn_size,
            clip_rewards=not args.no_clip_rewards,
            max_episodes=args.max_episodes,
        )


if __name__ == "__main__":
    main()
