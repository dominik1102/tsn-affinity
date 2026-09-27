
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import gymnasium as gym

from dt.mgdt_pytorch import (
    MGDTConfig,
    MultiGameDecisionTransformer,
    sample_action_from_logits,
    sample_expert_return,
)
from dt.utils import load_npz_dataset_for_task, make_minari_atari_env


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_ale_registered() -> None:
    try:
        import gymnasium as gym
        import ale_py

        gym.register_envs(ale_py)
        print("[ale] ALE successfully registered with Gymnasium")
    except Exception as e:
        raise RuntimeError(
            "ALE environments are not available. Install 'ale-py' or 'gymnasium[atari]' "
            "and ensure ALE is registered before env evaluation."
        ) from e


def _cuda_mem_mb(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "cuda_mem_alloc_MB": float(torch.cuda.memory_allocated(idx) / 1024**2),
        "cuda_mem_reserved_MB": float(torch.cuda.memory_reserved(idx) / 1024**2),
        "cuda_max_mem_alloc_MB": float(torch.cuda.max_memory_allocated(idx) / 1024**2),
        "cuda_max_mem_reserved_MB": float(torch.cuda.max_memory_reserved(idx) / 1024**2),
    }


def _print_cuda_setup(device: torch.device, model: torch.nn.Module) -> None:
    print(f"[cuda-check] torch.cuda.is_available()={torch.cuda.is_available()}")
    print(f"[cuda-check] requested/active device={device}")
    print(f"[cuda-check] model_device={next(model.parameters()).device}")

    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        print(f"[cuda-check] current_device_index={idx}")
        print(f"[cuda-check] gpu_name={torch.cuda.get_device_name(idx)}")
        print(f"[cuda-check] capability={torch.cuda.get_device_capability(idx)}")
        print(f"[cuda-check] device_count={torch.cuda.device_count()}")
        print(f"[cuda-check] tf32_matmul={torch.backends.cuda.matmul.allow_tf32}")
        print(f"[cuda-check] tf32_cudnn={torch.backends.cudnn.allow_tf32}")
        print(f"[cuda-check] cudnn_benchmark={torch.backends.cudnn.benchmark}")
        mem = _cuda_mem_mb(device)
        if mem:
            print(f"[cuda-check] initial_mem={json.dumps(mem)}")


def obs_diff_stats(obs_env, obs_ds) -> Tuple[float, float]:
    a = np.asarray(obs_env, dtype=np.float32)
    b = np.asarray(obs_ds, dtype=np.float32)

    if a.size and a.max() > 1.5:
        a = a / 255.0
    if b.size and b.max() > 1.5:
        b = b / 255.0

    diff = np.abs(a - b)
    return float(diff.max()), float(diff.mean())


def replay_actions(
    env: gym.Env,
    actions: np.ndarray,
    *,
    seed: int,
    max_steps: Optional[int] = None,
) -> Tuple[float, int, bool, bool, Optional[int]]:
    obs, info = env.reset(seed=int(seed))
    total = 0.0
    terminated = False
    truncated = False
    steps = 0
    last_info = info

    for t, a in enumerate(np.asarray(actions, dtype=np.int64)):
        if max_steps is not None and t >= int(max_steps):
            break
        obs, r, terminated, truncated, info = env.step(int(a))
        total += float(r)
        steps = t + 1
        last_info = info
        if terminated or truncated:
            break

    lives = None
    if isinstance(last_info, dict):
        lives = last_info.get("lives", None)
    return float(total), int(steps), bool(terminated), bool(truncated), lives


def infer_action_map_sparse_discrete(
    env: gym.Env,
    ep_actions: np.ndarray,
    ds_return: float,
    *,
    seed: int,
    tol: float = 1e-3,
    max_unique: int = 4,
) -> Optional[np.ndarray]:
    """Try to infer a sparse ds->env action map from a single episode."""
    if not isinstance(env.action_space, gym.spaces.Discrete):
        return None
    n = int(env.action_space.n)

    ep_actions = np.asarray(ep_actions, dtype=np.int64).reshape(-1)
    if ep_actions.size == 0 or ep_actions.min() < 0:
        return None

    uniq = sorted(int(x) for x in np.unique(ep_actions))
    if len(uniq) == 0 or len(uniq) > int(max_unique):
        return None

    ret0, *_ = replay_actions(env, ep_actions, seed=seed)
    diff0 = abs(ret0 - float(ds_return))
    if diff0 <= tol:
        return None

    map_size = max(int(ep_actions.max()) + 1, n)
    best_map = None
    best_diff = diff0

    for cand in itertools.permutations(range(n), len(uniq)):
        m = np.arange(map_size, dtype=np.int64)
        for u, v in zip(uniq, cand):
            m[int(u)] = int(v)

        mapped = m[ep_actions]
        ret, *_ = replay_actions(env, mapped, seed=seed)
        d = abs(ret - float(ds_return))
        if d < best_diff:
            best_diff = d
            best_map = m
            if best_diff <= tol:
                break

    if best_map is not None and best_diff <= tol:
        return best_map
    return None


def replay_check_and_maybe_remap_task(
    task: 'TaskData',
    *,
    dqn_size: int,
    tol: float = 1e-3,
    max_unique: int = 4,
) -> Optional[np.ndarray]:
    env = make_minari_atari_env(
        env_id=task.env_id,
        seed=task.seed,
        frame_stack=int(task.frame_stack),
        dqn_size=int(dqn_size),
        clip_rewards=bool(task.clip_rewards),
    )

    try:
        ep0 = task.episodes[0]
        ds_return = float(np.sum(np.asarray(ep0.rewards, dtype=np.float32)))
        env_return, steps, terminated, truncated, _ = replay_actions(
            env, np.asarray(ep0.actions, dtype=np.int64), seed=int(task.seed)
        )

        obs_reset, _ = env.reset(seed=int(task.seed))
        mx, mean = obs_diff_stats(obs_reset, ep0.observations[0])

        print(
            f"[replay-check] {task.name}: dataset_return={ds_return:.2f} env_return={env_return:.2f} "
            f"steps={steps} term={terminated} trunc={truncated} obs_diff_max={mx:.4f} obs_diff_mean={mean:.4f}"
        )

        if abs(env_return - ds_return) <= tol:
            print(f"[replay-check] {task.name}: action mapping identity OK")
            return None

        action_map = infer_action_map_sparse_discrete(
            env,
            np.asarray(ep0.actions, dtype=np.int64),
            ds_return,
            seed=int(task.seed),
            tol=float(tol),
            max_unique=int(max_unique),
        )

        if action_map is not None:
            for ep in task.episodes:
                ep.actions = action_map[np.asarray(ep.actions, dtype=np.int64)]

            env_return2, *_ = replay_actions(
                env, np.asarray(task.episodes[0].actions, dtype=np.int64), seed=int(task.seed)
            )
            try:
                meanings = env.unwrapped.get_action_meanings()
            except Exception:
                meanings = []
            print(f"[replay-check] {task.name}: remap applied, env_return_after={env_return2:.2f}, map_size={len(action_map)}")
            if meanings:
                print(f"[replay-check] {task.name}: env meanings = {meanings}")
            return action_map

        print(
            f"[replay-check][warn] {task.name}: no sparse action remap found. "
            f"This may indicate env/dataset mismatch (wrappers, frameskip, sticky/noop, or action semantics)."
        )
        return None
    finally:
        try:
            env.close()
        except Exception:
            pass


def apply_paper_model_hparams(args: argparse.Namespace) -> None:
    """Override model size to one of the paper's DT variants."""
    if args.paper_model is None:
        return
    if args.paper_model == "10m":
        args.d_model = 512
        args.n_layers = 4
        args.n_heads = 8
    elif args.paper_model == "40m":
        args.d_model = 768
        args.n_layers = 6
        args.n_heads = 12
    elif args.paper_model == "200m":
        args.d_model = 1280
        args.n_layers = 10
        args.n_heads = 20
    else:
        raise ValueError(f"Unknown paper_model: {args.paper_model}")


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass
class AtariEpisode:
    task_name: str
    env_id: str
    observations: np.ndarray  # [T,C,H,W]
    actions: np.ndarray       # [T]
    rewards: np.ndarray       # [T]
    returns_to_go: np.ndarray # [T]
    seed: int


@dataclass
class TaskData:
    name: str
    env_id: str
    seed: int
    frame_stack: int
    clip_rewards: bool
    episodes: List[AtariEpisode]
    episode_returns: np.ndarray


@dataclass
class JointBatch:
    observations: torch.Tensor  # [B,T,C,H,W]
    returns_to_go: torch.Tensor # [B,T]
    actions: torch.Tensor       # [B,T]
    rewards: torch.Tensor       # [B,T]
    valid_steps: torch.Tensor   # [B,T] bool
    task_names: List[str]


# -----------------------------------------------------------------------------
# Batcher
# -----------------------------------------------------------------------------


class MultiGameAtariBatcher:
    """Uniform-over-task batcher for MGDT Atari experiments."""

    def __init__(
        self,
        tasks: Sequence[TaskData],
        seq_len: int,
        device: torch.device,
        sample_task_uniform: bool = True,
    ):
        self.tasks = list(tasks)
        self.task_map = {t.name: t for t in self.tasks}
        self.seq_len = int(seq_len)
        self.device = device
        self.sample_task_uniform = bool(sample_task_uniform)

        if not self.tasks:
            raise ValueError("MultiGameAtariBatcher needs at least one task")

        self.obs_shape = tuple(self.tasks[0].episodes[0].observations[0].shape)

        if self.sample_task_uniform:
            self.task_probs = np.ones((len(self.tasks),), dtype=np.float64)
            self.task_probs /= self.task_probs.sum()
        else:
            counts = np.array(
                [sum(len(ep.observations) for ep in t.episodes) for t in self.tasks],
                dtype=np.float64,
            )
            self.task_probs = counts / counts.sum()

    def _sample_episode(self, task: TaskData) -> AtariEpisode:
        idx = np.random.randint(0, len(task.episodes))
        return task.episodes[idx]

    def _sample_window(
        self, ep: AtariEpisode
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        T = len(ep.actions)
        start = np.random.randint(0, T)
        end = min(T, start + self.seq_len)
        L = end - start

        obs = np.zeros((self.seq_len, *self.obs_shape), dtype=np.float32)
        rtg = np.zeros((self.seq_len,), dtype=np.float32)
        acts = np.zeros((self.seq_len,), dtype=np.int64)
        rews = np.zeros((self.seq_len,), dtype=np.float32)
        valid = np.zeros((self.seq_len,), dtype=np.bool_)

        obs[:L] = ep.observations[start:end].astype(np.float32)
        rtg[:L] = ep.returns_to_go[start:end].astype(np.float32)
        acts[:L] = ep.actions[start:end].astype(np.int64)
        rews[:L] = ep.rewards[start:end].astype(np.float32)
        valid[:L] = True
        return obs, rtg, acts, rews, valid

    def next_batch(self, batch_size: int) -> JointBatch:
        obs_list, rtg_list, act_list, rew_list, valid_list = [], [], [], [], []
        task_names: List[str] = []

        task_indices = np.random.choice(len(self.tasks), size=int(batch_size), p=self.task_probs)
        for ti in task_indices:
            task = self.tasks[int(ti)]
            ep = self._sample_episode(task)
            obs, rtg, acts, rews, valid = self._sample_window(ep)

            obs_list.append(obs)
            rtg_list.append(rtg)
            act_list.append(acts)
            rew_list.append(rews)
            valid_list.append(valid)
            task_names.append(task.name)

        return JointBatch(
            observations=torch.as_tensor(np.stack(obs_list), device=self.device, dtype=torch.float32),
            returns_to_go=torch.as_tensor(np.stack(rtg_list), device=self.device, dtype=torch.float32),
            actions=torch.as_tensor(np.stack(act_list), device=self.device, dtype=torch.long),
            rewards=torch.as_tensor(np.stack(rew_list), device=self.device, dtype=torch.float32),
            valid_steps=torch.as_tensor(np.stack(valid_list), device=self.device, dtype=torch.bool),
            task_names=task_names,
        )


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def compute_rtg(rewards: np.ndarray) -> np.ndarray:
    r = np.asarray(rewards, dtype=np.float32)
    return np.flip(np.cumsum(np.flip(r, axis=0), axis=0), axis=0).astype(np.float32)


def load_joint_tasks(spec_path: str, dataset_root: str, dataset_file: str) -> List[TaskData]:
    with open(spec_path, "r", encoding="utf-8") as f:
        tasks_json = json.load(f)
    if not isinstance(tasks_json, list) or not tasks_json:
        raise ValueError("spec must be a non-empty JSON list")

    out: List[TaskData] = []
    for i, task in enumerate(tasks_json, start=1):
        name = str(task["name"])
        params = dict(task.get("params", {}))
        env_id = str(params["game"])
        seed = int(task.get("seed", i))
        frame_stack = int(params.get("frame_stack", 4))
        clip_rewards = bool(params.get("clip_rewards", True))

        npz_path = os.path.join(dataset_root, name, dataset_file)
        episodes_obs, episodes_actions, episodes_rewards, ep_returns = load_npz_dataset_for_task(npz_path)

        episodes: List[AtariEpisode] = []
        for obs, acts, rews in zip(episodes_obs, episodes_actions, episodes_rewards):
            obs_np = np.asarray(obs, dtype=np.float32)
            acts_np = np.asarray(acts, dtype=np.int64).reshape(-1)
            rews_np = np.asarray(rews, dtype=np.float32).reshape(-1)
            rtg_np = compute_rtg(rews_np)
            episodes.append(
                AtariEpisode(
                    task_name=name,
                    env_id=env_id,
                    observations=obs_np,
                    actions=acts_np,
                    rewards=rews_np,
                    returns_to_go=rtg_np,
                    seed=seed,
                )
            )

        out.append(
            TaskData(
                name=name,
                env_id=env_id,
                seed=seed,
                frame_stack=frame_stack,
                clip_rewards=clip_rewards,
                episodes=episodes,
                episode_returns=np.asarray(ep_returns, dtype=np.float32),
            )
        )
    return out


# -----------------------------------------------------------------------------
# Offline evaluation
# -----------------------------------------------------------------------------


def evaluate_offline_per_task(
    model: MultiGameDecisionTransformer,
    tasks: Sequence[TaskData],
    seq_len: int,
    batch_size: int,
    n_batches: int,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    batchers = {
        t.name: MultiGameAtariBatcher([t], seq_len=seq_len, device=device, sample_task_uniform=True)
        for t in tasks
    }

    model.eval()
    out: Dict[str, Dict[str, float]] = {}
    with torch.no_grad():
        for t in tasks:
            losses = {"loss": [], "return_loss": [], "action_loss": [], "reward_loss": []}
            for _ in range(max(1, int(n_batches))):
                b = batchers[t.name].next_batch(batch_size)
                d = model.loss(
                    b.observations,
                    b.returns_to_go,
                    b.actions,
                    b.rewards,
                    valid_steps=b.valid_steps,
                )
                for k in losses:
                    if k in d:
                        losses[k].append(float(d[k].detach().cpu().item()))
            out[t.name] = {
                k: (float(np.mean(v)) if v else float("nan"))
                for k, v in losses.items()
                if v
            }
    return out


# -----------------------------------------------------------------------------
# Environment evaluation
# -----------------------------------------------------------------------------


def _task_uses_fire(task: TaskData, fire_id: int) -> bool:
    for ep in task.episodes:
        if (np.asarray(ep.actions).reshape(-1) == int(fire_id)).any():
            return True
    return False


def _task_rtg_stats(task: TaskData, model_return_range: Tuple[int, int]) -> Dict[str, int]:
    low, high = model_return_range
    high_inclusive = int(high - 1)

    rets = np.asarray(task.episode_returns, dtype=np.float32)

    def _clip(v: float) -> int:
        return int(max(low, min(high_inclusive, int(round(float(v))))))

    return {
        "min": _clip(float(np.min(rets))),
        "p10": _clip(float(np.percentile(rets, 10))),
        "p90": _clip(float(np.percentile(rets, 90))),
        "max": _clip(float(np.max(rets))),
    }


@torch.no_grad()
def sample_expert_return_clamped(
    return_logits: torch.Tensor,
    return_range: Tuple[int, int],
    *,
    allowed_low: int,
    allowed_high: int,
    kappa: float = 10.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    low, high = return_range
    if not (low <= allowed_low <= allowed_high < high):
        raise ValueError(
            f"Invalid clamp bounds: allowed_low={allowed_low}, allowed_high={allowed_high}, "
            f"global_range={return_range}"
        )

    logits = return_logits.clone()
    values = torch.arange(low, high, device=logits.device, dtype=torch.long)
    invalid = (values < int(allowed_low)) | (values > int(allowed_high))
    logits[..., invalid] = -1e9

    return sample_expert_return(
        logits,
        return_range,
        kappa=kappa,
        temperature=temperature,
    )


@torch.no_grad()
def evaluate_mgdt_env(
    model: MultiGameDecisionTransformer,
    task: TaskData,
    device: torch.device,
    *,
    episodes: int = 10,
    max_steps: int = 27_000,
    seq_len: int = 20,
    dqn_size: int = 84,
    kappa: float = 10.0,
    temperature: float = 1.0,
    greedy_actions: bool = False,
    inference_mode: str = "sampled_return",
    clip_rewards_for_eval: bool = False,
    topk_rollouts: int = 3,
) -> Dict[str, object]:
    env = make_minari_atari_env(
        env_id=task.env_id,
        seed=None,
        frame_stack=int(task.frame_stack),
        dqn_size=int(dqn_size),
        clip_rewards=bool(clip_rewards_for_eval),
    )

    try:
        meanings = env.unwrapped.get_action_meanings()
    except Exception:
        meanings = []

    fire_id = int(meanings.index("FIRE")) if (isinstance(meanings, (list, tuple)) and "FIRE" in meanings) else None
    use_auto_fire = fire_id is not None and (not _task_uses_fire(task, fire_id))

    low_return = int(model.cfg.return_range[0])
    high_return = int(model.cfg.return_range[1]) - 1

    task_rtg = _task_rtg_stats(task, model.cfg.return_range)

    fixed_rtg_value: Optional[float] = None
    decay_mode = False

    if inference_mode == "fixed_max_rtg":
        fixed_rtg_value = float(task_rtg["max"])
    elif inference_mode == "fixed_p90_rtg":
        fixed_rtg_value = float(task_rtg["p90"])
    elif inference_mode == "fixed_max_rtg_decay":
        fixed_rtg_value = float(task_rtg["max"])
        decay_mode = True
    elif inference_mode == "fixed_p90_rtg_decay":
        fixed_rtg_value = float(task_rtg["p90"])
        decay_mode = True

    def _clamp_rtg(x: float) -> float:
        return float(max(low_return, min(high_return, int(round(float(x))))))

    model.eval()
    episode_returns: List[float] = []
    episode_lengths: List[int] = []
    used_rtg_values: List[float] = []
    chosen_actions: List[int] = []
    first_debug: Dict[str, object] = {}

    for ep_idx in range(int(episodes)):
        obs, info = env.reset(seed=int(task.seed + ep_idx))
        total = 0.0
        terminated = truncated = False
        last_lives = info.get("lives", None) if isinstance(info, dict) else None

        obs_hist: List[np.ndarray] = []
        rtg_hist: List[float] = []
        act_hist: List[int] = []
        rew_hist: List[float] = []
        current_obs = np.asarray(obs, dtype=np.float32)
        fire_pending = bool(use_auto_fire)
        env_steps = 0

        current_decay_rtg = _clamp_rtg(fixed_rtg_value) if decay_mode and fixed_rtg_value is not None else None

        first_episode_rtg: List[float] = []
        first_episode_actions: List[int] = []

        for _ in range(int(max_steps)):
            if fire_pending and fire_id is not None:
                next_obs, r, terminated, truncated, info = env.step(int(fire_id))
                env_steps += 1
                total += float(r)
                current_obs = np.asarray(next_obs, dtype=np.float32)
                fire_pending = False
                last_lives = info.get("lives", last_lives) if isinstance(info, dict) else last_lives
                if terminated or truncated:
                    break

            obs_hist.append(current_obs.copy())

            if decay_mode and current_decay_rtg is not None:
                rtg_hist.append(float(current_decay_rtg))
            elif fixed_rtg_value is not None and not decay_mode:
                rtg_hist.append(float(fixed_rtg_value))
            else:
                rtg_hist.append(float(low_return))

            act_hist.append(0)
            rew_hist.append(0.0)

            obs_win = obs_hist[-seq_len:]
            rtg_win = rtg_hist[-seq_len:]
            act_win = act_hist[-seq_len:]
            rew_win = rew_hist[-seq_len:]
            L = len(obs_win)

            obs_arr = np.zeros((1, seq_len, *current_obs.shape), dtype=np.float32)
            rtg_arr = np.zeros((1, seq_len), dtype=np.float32)
            act_arr = np.zeros((1, seq_len), dtype=np.int64)
            rew_arr = np.zeros((1, seq_len), dtype=np.float32)
            valid_arr = np.zeros((1, seq_len), dtype=np.bool_)

            obs_arr[0, :L] = np.asarray(obs_win, dtype=np.float32)
            rtg_arr[0, :L] = np.asarray(rtg_win, dtype=np.float32)
            act_arr[0, :L] = np.asarray(act_win, dtype=np.int64)
            rew_arr[0, :L] = np.asarray(rew_win, dtype=np.float32)
            valid_arr[0, :L] = True

            obs_t = torch.as_tensor(obs_arr, device=device, dtype=torch.float32)
            rtg_t = torch.as_tensor(rtg_arr, device=device, dtype=torch.float32)
            act_t = torch.as_tensor(act_arr, device=device, dtype=torch.long)
            rew_t = torch.as_tensor(rew_arr, device=device, dtype=torch.float32)
            valid_t = torch.as_tensor(valid_arr, device=device, dtype=torch.bool)

            if inference_mode in ("fixed_max_rtg", "fixed_p90_rtg"):
                sampled_ret_value = _clamp_rtg(float(fixed_rtg_value))

            elif inference_mode in ("fixed_max_rtg_decay", "fixed_p90_rtg_decay"):
                sampled_ret_value = _clamp_rtg(float(current_decay_rtg))

            elif inference_mode == "sampled_return":
                out = model(obs_t, rtg_t, act_t, rew_t, valid_steps=valid_t)
                ret_logits = out["return_logits"][0, L - 1]
                sampled_ret_tok = sample_expert_return(
                    ret_logits.unsqueeze(0),
                    model.cfg.return_range,
                    kappa=kappa,
                    temperature=temperature,
                )
                sampled_ret_value = float(low_return + int(sampled_ret_tok.item()))

            elif inference_mode == "sampled_return_clamped":
                out = model(obs_t, rtg_t, act_t, rew_t, valid_steps=valid_t)
                ret_logits = out["return_logits"][0, L - 1]
                sampled_ret_tok = sample_expert_return_clamped(
                    ret_logits.unsqueeze(0),
                    model.cfg.return_range,
                    allowed_low=int(task_rtg["p10"]),
                    allowed_high=int(task_rtg["max"]),
                    kappa=kappa,
                    temperature=temperature,
                )
                sampled_ret_value = float(low_return + int(sampled_ret_tok.item()))

            else:
                raise ValueError(f"Unknown inference_mode: {inference_mode}")

            rtg_hist[-1] = sampled_ret_value
            rtg_arr[0, L - 1] = sampled_ret_value
            rtg_t = torch.as_tensor(rtg_arr, device=device, dtype=torch.float32)

            out = model(obs_t, rtg_t, act_t, rew_t, valid_steps=valid_t)
            action_logits = out["action_logits"][0, L - 1].clone()

            n_valid_actions = int(env.action_space.n)
            if n_valid_actions < action_logits.shape[-1]:
                action_logits[n_valid_actions:] = -1e9

            action = int(
                sample_action_from_logits(
                    action_logits.unsqueeze(0),
                    greedy=bool(greedy_actions),
                ).item()
            )
            act_hist[-1] = action

            next_obs, r, terminated, truncated, info = env.step(action)
            env_steps += 1
            total += float(r)
            rew_hist[-1] = float(r)
            current_obs = np.asarray(next_obs, dtype=np.float32)

            used_rtg_values.append(float(sampled_ret_value))
            chosen_actions.append(int(action))

            if ep_idx == 0 and len(first_episode_rtg) < 50:
                first_episode_rtg.append(float(sampled_ret_value))
                first_episode_actions.append(int(action))

            if decay_mode and current_decay_rtg is not None:
                current_decay_rtg = _clamp_rtg(float(current_decay_rtg) - float(r))

            if use_auto_fire and fire_id is not None and isinstance(info, dict):
                lives = info.get("lives", None)
                if last_lives is not None and lives is not None and lives < last_lives and not (terminated or truncated):
                    fire_pending = True
                last_lives = lives

            if terminated or truncated:
                break

        episode_returns.append(float(total))
        episode_lengths.append(int(env_steps))

        if ep_idx == 0:
            first_debug = {
                "first_50_rtg": first_episode_rtg,
                "first_50_actions": first_episode_actions,
            }

    try:
        env.close()
    except Exception:
        pass

    action_hist = dict(Counter(chosen_actions))
    total_actions = max(1, len(chosen_actions))
    if action_hist:
        top_action = int(max(action_hist, key=action_hist.get))
        top_action_frac = float(action_hist[top_action] / total_actions)
    else:
        top_action = -1
        top_action_frac = 0.0

    sorted_returns = sorted(episode_returns, reverse=True)
    k = min(max(1, int(topk_rollouts)), len(sorted_returns))
    topk_mean_return = float(np.mean(sorted_returns[:k])) if sorted_returns else 0.0

    stats: Dict[str, object] = {
        "mean_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "std_return": float(np.std(episode_returns)) if episode_returns else 0.0,
        "topk_mean_return": float(topk_mean_return),
        "mean_ep_len": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "rtg_mean": float(np.mean(used_rtg_values)) if used_rtg_values else 0.0,
        "rtg_std": float(np.std(used_rtg_values)) if used_rtg_values else 0.0,
        "top_action": int(top_action),
        "top_action_frac": float(top_action_frac),
        "action_hist": {str(k): int(v) for k, v in sorted(action_hist.items())},
        "episode_returns": [float(x) for x in episode_returns],
        "episode_lengths": [int(x) for x in episode_lengths],
        "inference_mode": inference_mode,
        "task_rtg_min": int(task_rtg["min"]),
        "task_rtg_p10": int(task_rtg["p10"]),
        "task_rtg_p90": int(task_rtg["p90"]),
        "task_rtg_max": int(task_rtg["max"]),
        "clip_rewards_for_eval": bool(clip_rewards_for_eval),
    }
    stats.update(first_debug)
    return stats


# -----------------------------------------------------------------------------
# Main training script
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Joint MGDT training on an Atari spec (paper-closer variant)")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--dataset-file", default="train_run1_sample100.npz")
    ap.add_argument("--seq-len", type=int, default=4)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)

    # Paper-like model presets from supplementary Table 1:
    # DT-10M: 4 layers, 512 d_model, 8 heads
    # DT-40M: 6 layers, 768 d_model, 12 heads
    # DT-200M: 10 layers, 1280 d_model, 20 heads
    ap.add_argument("--paper-model", choices=["10m", "40m", "200m"], default="40m")

    # Manual overrides still possible
    ap.add_argument("--d-model", type=int, default=768)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=12)
    ap.add_argument("--p-drop", type=float, default=0.1)
    ap.add_argument("--patch-size", type=int, default=14)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    # Paper-like defaults
    ap.add_argument("--reward-values", type=int, nargs="*", default=[-1, 0, 1])
    ap.add_argument("--return-range-low", type=int, default=-20)
    ap.add_argument("--return-range-high", type=int, default=101)
    ap.add_argument("--num-actions", type=int, default=18)

    ap.add_argument("--eval-every", type=int, default=10000)
    ap.add_argument("--offline-eval-batches", type=int, default=2)
    ap.add_argument("--episodes-eval", type=int, default=3)
    ap.add_argument("--env-eval-every", type=int, default=0)
    ap.add_argument("--max-ep-len", type=int, default=5000)
    ap.add_argument("--dqn-size", type=int, default=84)
    ap.add_argument("--kappa", type=float, default=10.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--greedy-actions", action="store_true",
                    help="If set, use argmax actions. Paper-like setting leaves this OFF (stochastic actions).")
    ap.add_argument("--env-eval-clip-rewards", action="store_true",
                    help="If set, env eval uses clipped rewards. Leave OFF for paper-like raw-score evaluation.")
    ap.add_argument("--topk-rollouts", type=int, default=3,
                    help="Also report the mean of top-k rollout returns, useful for comparison to paper Sec. 4.6.")
    ap.add_argument("--skip-replay-check", action="store_true",
                    help="Skip pre-training replay alignment check and sparse action remapping.")
    ap.add_argument("--replay-check-tol", type=float, default=1e-3)
    ap.add_argument("--replay-check-max-unique", type=int, default=4,
                    help="Try sparse action remap only if the first episode uses at most this many unique actions.")
    ap.add_argument(
        "--inference-mode",
        choices=[
            "sampled_return",
            "fixed_max_rtg",
            "fixed_p90_rtg",
            "fixed_max_rtg_decay",
            "fixed_p90_rtg_decay",
            "sampled_return_clamped",
        ],
        default="sampled_return",
    )

    ap.add_argument("--runs-root", default="runs_mgdt")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    apply_paper_model_hparams(args)

    set_all_seeds(int(args.seed))

    requested_device = str(args.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but torch.cuda.is_available() is False. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats(device)
        torch.set_float32_matmul_precision("high")

    if int(args.env_eval_every) > 0:
        _ensure_ale_registered()

    tasks = load_joint_tasks(args.spec, args.dataset_root, args.dataset_file)

    if not args.skip_replay_check:
        print("[replay-check] running per-task dataset/env alignment check on first episode...")
        for t in tasks:
            replay_check_and_maybe_remap_task(
                t,
                dqn_size=int(args.dqn_size),
                tol=float(args.replay_check_tol),
                max_unique=int(args.replay_check_max_unique),
            )

    obs_shape = tuple(tasks[0].episodes[0].observations[0].shape)
    if len(obs_shape) != 3:
        raise ValueError(f"Expected Atari obs shape [C,H,W], got {obs_shape}")
    in_channels, H, W = obs_shape

    reward_values = tuple(int(v) for v in args.reward_values)
    return_low = int(args.return_range_low)
    return_high = int(args.return_range_high)
    global_n_actions = int(args.num_actions)

    cfg = MGDTConfig(
        image_size=(int(H), int(W)),
        in_channels=int(in_channels),
        patch_size=int(args.patch_size),
        d_model=int(args.d_model),
        n_head=int(args.n_heads),
        n_layer=int(args.n_layers),
        dropout=float(args.p_drop),
        max_steps=int(args.seq_len),
        num_actions=int(global_n_actions),
        reward_values=tuple(reward_values),
        return_range=(int(return_low), int(return_high)),
        predict_reward=True,
        use_spatial_tokens=True,
    )

    model = MultiGameDecisionTransformer(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    batcher = MultiGameAtariBatcher(
        tasks,
        seq_len=int(args.seq_len),
        device=device,
        sample_task_uniform=True,
    )

    run_name = args.tag or os.path.splitext(os.path.basename(args.spec))[0]
    run_dir = os.path.join(args.runs_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"[mgdt] device={device}")
    print(f"[mgdt] tasks={[t.name for t in tasks]}")
    print(f"[mgdt] obs_shape={obs_shape} num_actions={global_n_actions} reward_values={reward_values} return_range={(return_low, return_high)}")
    print(f"[mgdt] model_preset={args.paper_model} d_model={args.d_model} layers={args.n_layers} heads={args.n_heads}")
    print(f"[mgdt] paper_like_eval_raw_rewards={not args.env_eval_clip_rewards}")
    print(f"[mgdt] run_dir={run_dir}")
    _print_cuda_setup(device, model)

    history: List[Dict[str, float]] = []
    env_debug_history: List[Dict[str, object]] = []
    last_log_step = 0
    last_log_time = time.time()

    for step in range(1, int(args.steps) + 1):
        b = batcher.next_batch(int(args.batch_size))

        if step == 1:
            print(f"[cuda-check] first_batch_obs_device={b.observations.device}")
            print(f"[cuda-check] first_batch_rtg_device={b.returns_to_go.device}")
            print(f"[cuda-check] first_batch_actions_device={b.actions.device}")
            print(f"[cuda-check] first_batch_rewards_device={b.rewards.device}")
            print(f"[cuda-check] first_batch_valid_device={b.valid_steps.device}")
            if device.type == "cuda":
                print(f"[cuda-check] first_batch_mem={json.dumps(_cuda_mem_mb(device))}")

        losses = model.loss(
            b.observations,
            b.returns_to_go,
            b.actions,
            b.rewards,
            valid_steps=b.valid_steps,
        )

        opt.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        opt.step()

        if step == 1 or step % max(1, int(args.eval_every)) == 0 or step == int(args.steps):
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            now = time.time()
            steps_since_last = step - last_log_step
            elapsed = now - last_log_time
            sec_per_step = elapsed / max(1, steps_since_last)

            train_log: Dict[str, float] = {
                "step": float(step),
                "train_loss": float(losses["loss"].detach().cpu().item()),
                "train_return_loss": float(losses["return_loss"].detach().cpu().item()),
                "train_action_loss": float(losses["action_loss"].detach().cpu().item()),
                "sec_per_step": float(sec_per_step),
            }
            if "reward_loss" in losses:
                train_log["train_reward_loss"] = float(losses["reward_loss"].detach().cpu().item())

            offline_eval = evaluate_offline_per_task(
                model,
                tasks,
                seq_len=int(args.seq_len),
                batch_size=int(args.batch_size),
                n_batches=int(args.offline_eval_batches),
                device=device,
            )
            for task_name, d in offline_eval.items():
                train_log[f"offline_action_loss/{task_name}"] = float(d.get("action_loss", float("nan")))
                train_log[f"offline_total_loss/{task_name}"] = float(d.get("loss", float("nan")))

            if device.type == "cuda":
                train_log.update(_cuda_mem_mb(device))

            if int(args.env_eval_every) > 0 and (step % int(args.env_eval_every) == 0 or step == int(args.steps)):
                t0_env = time.time()
                env_debug_step: Dict[str, object] = {"step": int(step), "inference_mode": args.inference_mode}
                for t in tasks:
                    stats = evaluate_mgdt_env(
                        model,
                        t,
                        device,
                        episodes=int(args.episodes_eval),
                        max_steps=int(args.max_ep_len),
                        seq_len=int(args.seq_len),
                        dqn_size=int(args.dqn_size),
                        kappa=float(args.kappa),
                        temperature=float(args.temperature),
                        greedy_actions=bool(args.greedy_actions),
                        inference_mode=str(args.inference_mode),
                        clip_rewards_for_eval=bool(args.env_eval_clip_rewards),
                        topk_rollouts=int(args.topk_rollouts),
                    )
                    train_log[f"env_return/{t.name}"] = float(stats["mean_return"])
                    train_log[f"env_return_std/{t.name}"] = float(stats["std_return"])
                    train_log[f"env_topk_return/{t.name}"] = float(stats["topk_mean_return"])
                    train_log[f"env_ep_len/{t.name}"] = float(stats["mean_ep_len"])
                    train_log[f"env_rtg_mean/{t.name}"] = float(stats["rtg_mean"])
                    train_log[f"env_rtg_std/{t.name}"] = float(stats["rtg_std"])
                    train_log[f"env_top_action/{t.name}"] = float(stats["top_action"])
                    train_log[f"env_top_action_frac/{t.name}"] = float(stats["top_action_frac"])
                    train_log[f"env_task_rtg_min/{t.name}"] = float(stats["task_rtg_min"])
                    train_log[f"env_task_rtg_p10/{t.name}"] = float(stats["task_rtg_p10"])
                    train_log[f"env_task_rtg_p90/{t.name}"] = float(stats["task_rtg_p90"])
                    train_log[f"env_task_rtg_max/{t.name}"] = float(stats["task_rtg_max"])
                    env_debug_step[t.name] = stats

                train_log["env_eval_sec"] = float(time.time() - t0_env)
                env_debug_history.append(env_debug_step)

                with open(os.path.join(run_dir, "env_debug.json"), "w", encoding="utf-8") as f:
                    json.dump(env_debug_history, f, indent=2)

            history.append(train_log)
            print(json.dumps(train_log, ensure_ascii=False))

            last_log_step = step
            last_log_time = time.time()

    torch.save(
        {"model_state_dict": model.state_dict(), "config": cfg.__dict__},
        os.path.join(run_dir, "mgdt_joint.pt"),
    )
    with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[mgdt] saved checkpoint and logs to {run_dir}")


if __name__ == "__main__":
    main()
