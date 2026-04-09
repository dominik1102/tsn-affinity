#!/usr/bin/env python3
from __future__ import annotations

import os
import csv
import math
import argparse
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

import gymnasium as gym
import panda_gym  # noqa: F401

from bin.config import PANDA_TASKS
from clbench.io.run_logger import build_run_dir, save_json

from dt.dataset import Trajectory
from dt.dataset_panda import load_panda_offline_pkl
from dt.panda_dt import PandaDecisionTransformer


PANDA_OBS_KEYS: Tuple[str, ...] = ("observation", "achieved_goal", "desired_goal")


class PandaTimeFeatureWrapper(gym.Wrapper):
    """
    Add a single normalized remaining-time feature to obs["observation"].

    This is the closest simple equivalent of the old `time-feature` wrapper.
    At reset: 1.0
    After one step with max_steps=50: 0.98
    """

    def __init__(self, env: gym.Env, max_steps: Optional[int] = None):
        super().__init__(env)

        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise ValueError("PandaTimeFeatureWrapper requires Dict observation space")
        if "observation" not in env.observation_space.spaces:
            raise ValueError("Expected key 'observation' in Dict observation space")

        base = env.observation_space.spaces["observation"]
        if not isinstance(base, gym.spaces.Box) or len(base.shape) != 1:
            raise ValueError("Expected obs['observation'] to be 1D Box")

        inferred = getattr(getattr(env, "spec", None), "max_episode_steps", None)
        if max_steps is None:
            max_steps = inferred
        if max_steps is None:
            max_steps = 50
        self.max_steps = max(1, int(max_steps))
        self.elapsed_steps = 0

        low = np.asarray(base.low, dtype=np.float32).reshape(-1)
        high = np.asarray(base.high, dtype=np.float32).reshape(-1)

        spaces = dict(env.observation_space.spaces)
        spaces["observation"] = gym.spaces.Box(
            low=np.concatenate([low, np.array([0.0], dtype=np.float32)], axis=0),
            high=np.concatenate([high, np.array([1.0], dtype=np.float32)], axis=0),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(spaces)

    def _time_value(self) -> np.ndarray:
        v = 1.0 - (float(self.elapsed_steps) / float(self.max_steps))
        v = float(np.clip(v, 0.0, 1.0))
        return np.array([v], dtype=np.float32)

    def _augment(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        out = dict(obs)
        base = np.asarray(out["observation"], dtype=np.float32).reshape(-1)
        out["observation"] = np.concatenate([base, self._time_value()], axis=0).astype(np.float32, copy=False)
        return out

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.elapsed_steps = 0
        obs, info = self.env.reset(seed=seed, options=options)
        return self._augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.elapsed_steps += 1
        return self._augment(obs), reward, terminated, truncated, info


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


def flatten_panda_obs(obs: Any, obs_keys: Optional[Sequence[str]] = None) -> np.ndarray:
    if isinstance(obs, dict):
        if obs_keys is None:
            if all(k in obs for k in PANDA_OBS_KEYS):
                obs_keys = PANDA_OBS_KEYS
            else:
                obs_keys = tuple(obs.keys())
        parts = [np.asarray(obs[k], dtype=np.float32).ravel() for k in obs_keys]
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def inspect_env_obs_dim(env_id: str, add_time_feature: bool) -> Tuple[int, List[str]]:
    env = gym.make(env_id)
    try:
        if add_time_feature:
            env = PandaTimeFeatureWrapper(env)

        obs_keys: List[str]
        if isinstance(env.observation_space, gym.spaces.Dict):
            if all(k in env.observation_space.spaces for k in PANDA_OBS_KEYS):
                obs_keys = list(PANDA_OBS_KEYS)
            else:
                obs_keys = list(env.observation_space.spaces.keys())
        else:
            obs_keys = []

        obs, _ = env.reset(seed=0)
        flat = flatten_panda_obs(obs, obs_keys if obs_keys else None)
        return int(flat.shape[0]), obs_keys
    finally:
        try:
            env.close()
        except Exception:
            pass


def decide_time_feature_mode(env_id: str, dataset_obs_dim: int, mode: str) -> Tuple[bool, List[str]]:
    mode = str(mode).lower()
    raw_dim, obs_keys = inspect_env_obs_dim(env_id, add_time_feature=False)
    tf_dim, _ = inspect_env_obs_dim(env_id, add_time_feature=True)

    if mode == "on":
        return True, obs_keys
    if mode == "off":
        return False, obs_keys
    if mode != "auto":
        raise ValueError("time_feature must be auto|on|off")

    if dataset_obs_dim == tf_dim:
        return True, obs_keys
    if dataset_obs_dim == raw_dim:
        return False, obs_keys

    raise ValueError(
        f"Cannot match dataset obs_dim={dataset_obs_dim} to env dims: raw={raw_dim}, with_time_feature={tf_dim}."
    )


def make_panda_env(env_id: str, add_time_feature: bool) -> gym.Env:
    try:
        env = gym.make(env_id, render_mode="rgb_array")
    except TypeError:
        env = gym.make(env_id, render="rgb_array")

    if add_time_feature:
        env = PandaTimeFeatureWrapper(env)
    return env


def traj_return(tr: Trajectory) -> float:
    rtg = np.asarray(tr.returns_to_go, dtype=np.float32).reshape(-1)
    if rtg.size > 0:
        return float(rtg[0])
    return float(np.sum(np.asarray(tr.rewards, dtype=np.float32)))


def pick_target_return(returns: np.ndarray, mode: str) -> float:
    mode = str(mode).lower()
    if returns.size == 0:
        return 0.0
    if mode == "max":
        return float(np.max(returns))
    if mode == "p90":
        return float(np.percentile(returns, 90))
    if mode == "p75":
        return float(np.percentile(returns, 75))
    if mode == "mean":
        return float(np.mean(returns))
    raise ValueError("mode must be max|p90|p75|mean")


def ensure_traj_list(trajs_raw: Any) -> List[Trajectory]:
    if isinstance(trajs_raw, dict):
        trajs_raw = list(trajs_raw.values())
    if not isinstance(trajs_raw, (list, tuple)):
        raise TypeError(f"Expected list/tuple/dict from loader, got {type(trajs_raw)}")

    trajs: List[Trajectory] = []
    for tr in trajs_raw:
        if isinstance(tr, Trajectory):
            if len(tr.actions) > 0:
                trajs.append(tr)
            continue

        if isinstance(tr, dict):
            obs = np.asarray(tr["obs"], dtype=np.float32)
            actions = np.asarray(tr["actions"], dtype=np.float32)
            rewards = np.asarray(tr["rewards"], dtype=np.float32)
            if actions.shape[0] == 0:
                continue

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
            continue

        raise TypeError(f"Unsupported trajectory type: {type(tr)}")

    return trajs


def compute_mean_std(trajs: List[Trajectory], obs_dim: int) -> Tuple[np.ndarray, np.ndarray]:
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


def estimate_updates_per_epoch(total_transitions: int, batch_size: int) -> int:
    return max(1, int(math.ceil(float(total_transitions) / float(batch_size))))


def make_minibatches_panda(
    trajs: List[Trajectory],
    *,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    obs_dim: int,
    act_dim: int,
):
    seq_len = int(seq_len)
    batch_size = int(batch_size)
    dev = device

    lens = np.array([len(t.actions) for t in trajs], dtype=np.int64)
    lens = np.clip(lens, 1, None)
    cdf = np.cumsum(lens / lens.sum())

    while True:
        u = np.random.rand(batch_size)
        idxs = np.searchsorted(cdf, u, side="right")

        obs_b = np.zeros((batch_size, seq_len, obs_dim), dtype=np.float32)
        act_b = np.zeros((batch_size, seq_len, act_dim), dtype=np.float32)
        rtg_b = np.zeros((batch_size, seq_len, 1), dtype=np.float32)
        ts_b = np.zeros((batch_size, seq_len), dtype=np.int64)
        mask_b = np.zeros((batch_size, seq_len), dtype=np.bool_)

        for b, idx in enumerate(idxs):
            tr = trajs[int(idx)]
            L = int(len(tr.actions))
            si = np.random.randint(0, L)
            end = min(si + seq_len, L)
            tlen = end - si
            pad = seq_len - tlen

            obs = np.asarray(tr.obs, dtype=np.float32).reshape(-1, obs_dim)[si:end]
            act = np.asarray(tr.actions, dtype=np.float32).reshape(-1, act_dim)[si:end]
            rtg = np.asarray(tr.returns_to_go, dtype=np.float32).reshape(-1)[si:end]

            obs_b[b, pad:, :] = obs
            act_b[b, pad:, :] = act
            rtg_b[b, pad:, 0] = rtg
            ts_b[b, pad:] = np.arange(si, si + tlen, dtype=np.int64)
            mask_b[b, pad:] = True

        yield (
            torch.from_numpy(obs_b).to(dev),
            torch.from_numpy(act_b).to(dev),
            torch.from_numpy(rtg_b).to(dev),
            torch.from_numpy(ts_b).to(dev),
            torch.from_numpy(mask_b).to(dev),
        )


@torch.no_grad()
def evaluate_dt_panda(
    model,
    env: gym.Env,
    episodes: int,
    device: Union[str, torch.device],
    max_steps: int,
    target_return: float,
    *,
    seed: Optional[int],
    obs_keys: Optional[Sequence[str]],
    obs_pad_to: Optional[int],
    act_pad_to: Optional[int],
    gamma: float,
    clip_action: bool,
    use_prev_action: bool,
    timestep_clip_max: Optional[int],
) -> float:
    if (not hasattr(model, "act")) and hasattr(model, "model"):
        model = model.model
    if not hasattr(model, "act"):
        raise AttributeError("Model must implement .act()")
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError("evaluate_dt_panda expects Box action space")

    env_act_dim = int(np.prod(env.action_space.shape))
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)

    model_act_dim = getattr(model, "act_dim", None)
    if model_act_dim is not None:
        model_act_dim = int(model_act_dim)

    def _pad_or_trunc(x: np.ndarray, size: Optional[int]) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if size is None:
            return x
        size = int(size)
        if x.size < size:
            return np.pad(x, (0, size - x.size), mode="constant")
        if x.size > size:
            return x[:size]
        return x

    returns: List[float] = []
    model.eval()

    for ep in range(int(episodes)):
        if hasattr(model, "reset_history"):
            model.reset_history()

        if seed is None:
            obs, info = env.reset()
        else:
            obs, info = env.reset(seed=int(seed + ep))

        total = 0.0
        rtg = float(target_return)
        prev_action = None

        for t in range(int(max_steps)):
            obs_vec = flatten_panda_obs(obs, obs_keys)
            obs_vec = _pad_or_trunc(obs_vec, obs_pad_to)

            t_model = int(t)
            if timestep_clip_max is not None:
                t_model = min(t_model, int(timestep_clip_max))

            a_pred = model.act(
                obs_vec,
                rtg_scalar=rtg,
                t=t_model,
                prev_action=(prev_action if use_prev_action else None),
                device=str(device),
            )
            a_pred = _pad_or_trunc(a_pred, act_pad_to)

            if a_pred.size < env_act_dim:
                a_env_flat = np.pad(a_pred, (0, env_act_dim - a_pred.size), mode="constant")
            else:
                a_env_flat = a_pred[:env_act_dim]

            if clip_action:
                a_env_flat = np.clip(a_env_flat, low[:env_act_dim], high[:env_act_dim])

            a_env = a_env_flat.astype(np.float32, copy=False).reshape(env.action_space.shape)
            obs, r, terminated, truncated, info = env.step(a_env)
            r = float(r)
            total += r

            if use_prev_action:
                pa = a_env_flat.astype(np.float32, copy=False).reshape(-1)
                if model_act_dim is not None:
                    prev_action = _pad_or_trunc(pa, model_act_dim)
                else:
                    prev_action = pa
            else:
                prev_action = None

            if gamma == 1.0:
                rtg = rtg - r
            else:
                rtg = (rtg - r) / float(gamma)

            if terminated or truncated:
                break

        returns.append(total)

    return float(np.mean(returns)) if returns else 0.0


def train_single_panda_task(
    trajs: List[Trajectory],
    obs_dim: int,
    act_dim: int,
    seq_len: int,
    max_ep_len: int,
    device_t: torch.device,
    updates: int,
    batch_size: int,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    rtg_scale: float,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    n_heads: int,
    log_every: int,
) -> Tuple[torch.nn.Module, List[Dict[str, float]]]:
    _maybe_set_cuda_device(device_t)

    model = PandaDecisionTransformer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        d_model=128,
        n_layers=3,
        n_heads=int(n_heads),
        seq_len=int(seq_len),
        p_drop=0.1,
        max_ep_len=int(max_ep_len),
        act_tanh=False,
        obs_mean=obs_mean,
        obs_std=obs_std,
        rtg_scale=float(rtg_scale),
    ).to(device_t)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )

    loader = make_minibatches_panda(
        trajs,
        seq_len=seq_len,
        batch_size=batch_size,
        device=device_t,
        obs_dim=obs_dim,
        act_dim=act_dim,
    )

    train_log: List[Dict[str, float]] = []
    model.train()

    for update in range(int(updates)):
        obs, actions, rtg, ts, mask = next(loader)
        pred = model(obs, actions, rtg, ts, attention_mask=mask)

        mse_per_step = ((pred - actions) ** 2).mean(dim=-1)
        mask_f = mask.float()
        denom = mask_f.sum().clamp(min=1.0)
        loss = (mse_per_step * mask_f).sum() / denom

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        opt.step()

        if log_every > 0 and (((update + 1) % log_every) == 0 or update == 0):
            lval = float(loss.detach().cpu().item())
            train_log.append({"update": int(update + 1), "loss": lval})
            print(f"[panda train] update {update + 1}/{updates}, loss={lval:.6f}")

    return model, train_log


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="all", choices=["all", *PANDA_TASKS.keys()])

    p.add_argument("--steps-per-task", type=int, default=20000)
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--max-ep-len", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--episodes-eval", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--log-every", type=int, default=500)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--gamma", type=float, default=1.0)

    p.add_argument("--target-mode", choices=["max", "p90", "p75", "mean"], default="max")
    p.add_argument("--target-return", type=float, default=None)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=0.25)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--rtg-scale", type=float, default=1000.0)

    p.add_argument("--time-feature", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--use-prev-action", action="store_true", default=True)
    p.add_argument("--no-prev-action", dest="use_prev_action", action="store_false")
    p.add_argument("--sweep-targets", action="store_true")

    p.add_argument("--runs-root", type=str, default="runs")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--use-best-sweep-target", action="store_true")

    args = p.parse_args()

    _set_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

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

        # load once with standard Panda key order
        trajs_raw = load_panda_offline_pkl(dataset_path, gamma=float(args.gamma), obs_keys=PANDA_OBS_KEYS)
        trajs = ensure_traj_list(trajs_raw)
        if not trajs:
            raise RuntimeError(f"No trajectories loaded from {dataset_path}")

        lens = np.array([len(t.actions) for t in trajs], dtype=np.int64)
        rets = np.array([traj_return(t) for t in trajs], dtype=np.float32)

        total_transitions = int(lens.sum())
        train_max_len = int(lens.max())
        seq_len_use = min(int(args.seq_len), train_max_len)

        obs_dim = int(np.asarray(trajs[0].obs).shape[-1])
        act_dim = int(np.asarray(trajs[0].actions).shape[-1])

        use_time_feature, obs_keys = decide_time_feature_mode(env_id, obs_dim, args.time_feature)

        print(f"[dataset] obs_keys={obs_keys}")
        print(f"[env] use_time_feature={use_time_feature}")
        print(f"[dataset] using seq_len={seq_len_use}")
        print(
            f"[dataset] n_trajs={len(trajs)} total_transitions={total_transitions} "
            f"traj_len mean={lens.mean():.2f} min={lens.min()} max={lens.max()} "
            f"p50={np.percentile(lens, 50):.0f} p90={np.percentile(lens, 90):.0f}"
        )
        print(f"[dataset] obs_dim={obs_dim}, act_dim={act_dim}")
        print(f"[dataset] return mean={rets.mean():.3f} min={rets.min():.3f} max={rets.max():.3f}")

        obs_mean, obs_std = compute_mean_std(trajs, obs_dim)
        print(f"[dataset] computed obs_mean/std (std min={obs_std.min():.6f}, max={obs_std.max():.6f})")
        print(f"[dataset] rtg_scale={float(args.rtg_scale):.6f}")

        if args.target_return is not None:
            target = float(args.target_return)
        else:
            target = pick_target_return(rets, args.target_mode)
        print(f"[eval] selected target_return={target:.6f} ({args.target_mode if args.target_return is None else 'manual'})")

        task_dir = os.path.join(run_dir, name)
        os.makedirs(task_dir, exist_ok=True)

        env = make_panda_env(env_id, add_time_feature=use_time_feature)

        try:
            try:
                env.reset(seed=int(args.seed))
            except TypeError:
                pass

            max_steps = int(args.max_steps)
            max_ep_len_use = max(int(args.max_ep_len), train_max_len)
            timestep_clip_max = max(0, train_max_len - 1)
            updates_per_epoch = estimate_updates_per_epoch(total_transitions, int(args.batch_size))
            approx_epochs = float(args.steps_per_task) / float(updates_per_epoch)

            print(f"[dataset] using max_ep_len={max_ep_len_use}")
            print(
                f"[train] optimizer_updates={int(args.steps_per_task)} "
                f"approx_updates_per_epoch={updates_per_epoch} approx_epochs={approx_epochs:.2f}"
            )
            print(
                f"[model] d_model=128 n_layers=3 n_heads={int(args.n_heads)} "
                f"seq_len={seq_len_use} max_ep_len={max_ep_len_use} rtg_scale={float(args.rtg_scale)}"
            )

            model, train_log = train_single_panda_task(
                trajs=trajs,
                obs_dim=obs_dim,
                act_dim=act_dim,
                seq_len=seq_len_use,
                max_ep_len=max_ep_len_use,
                device_t=device_t,
                updates=int(args.steps_per_task),
                batch_size=int(args.batch_size),
                obs_mean=obs_mean,
                obs_std=obs_std,
                rtg_scale=float(args.rtg_scale),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                grad_clip=float(args.grad_clip),
                n_heads=int(args.n_heads),
                log_every=int(args.log_every),
            )

            if train_log:
                _write_csv(os.path.join(task_dir, "train_log.csv"), train_log, fieldnames=["update", "loss"])

            model.eval()

            if args.sweep_targets or args.use_best_sweep_target:
                eval_targets = [
                    float(np.max(rets)),
                    float(np.percentile(rets, 90)),
                    float(np.percentile(rets, 75)),
                    float(np.mean(rets)),
                    0.0,
                    float(target),
                ]
                uniq_targets: List[float] = []
                for x in eval_targets:
                    if not any(abs(float(x) - y) <= 1e-12 for y in uniq_targets):
                        uniq_targets.append(float(x))
                eval_targets = uniq_targets
            else:
                eval_targets = [float(target)]

            sweep_rows: List[Dict[str, float]] = []
            selected_score: Optional[float] = None
            selected_target = float(target)

            for trg in eval_targets:
                score = evaluate_dt_panda(
                    model=model,
                    env=env,
                    episodes=int(args.episodes_eval),
                    device=device_t,
                    max_steps=max_steps,
                    target_return=float(trg),
                    seed=int(args.seed),
                    obs_keys=obs_keys,
                    obs_pad_to=obs_dim,
                    act_pad_to=act_dim,
                    gamma=float(args.gamma),
                    clip_action=True,
                    use_prev_action=bool(args.use_prev_action),
                    timestep_clip_max=timestep_clip_max,
                )
                sweep_rows.append({"target": float(trg), "score": float(score)})
                print(f"[eval] target={trg:.3f} -> score={score:.3f}")

                if abs(float(trg) - float(target)) <= 1e-12:
                    selected_score = float(score)

            best_row = max(sweep_rows, key=lambda r: float(r["score"]))
            best_target = float(best_row["target"])
            best_score = float(best_row["score"])
            print(f"[eval] best sweep target={best_target:.3f} -> score={best_score:.3f}")

            if args.use_best_sweep_target:
                selected_target = best_target
                selected_score = best_score
            elif selected_score is None:
                selected_score = float(
                    evaluate_dt_panda(
                        model=model,
                        env=env,
                        episodes=int(args.episodes_eval),
                        device=device_t,
                        max_steps=max_steps,
                        target_return=float(target),
                        seed=int(args.seed),
                        obs_keys=obs_keys,
                        obs_pad_to=obs_dim,
                        act_pad_to=act_dim,
                        gamma=float(args.gamma),
                        clip_action=True,
                        use_prev_action=bool(args.use_prev_action),
                        timestep_clip_max=timestep_clip_max,
                    )
                )

            scores[name] = float(selected_score)
            print(f"[eval] Single-task Panda DT on {name}: {selected_score:.3f}")

            _write_csv(os.path.join(task_dir, "target_sweep.csv"), sweep_rows, fieldnames=["target", "score"])

            save_json(
                os.path.join(task_dir, "task_results.json"),
                {
                    "task": name,
                    "env_id": env_id,
                    "dataset": dataset_path,
                    "score": float(selected_score),
                    "target_sweep": sweep_rows,
                    "obs_keys": list(obs_keys),
                    "use_time_feature": bool(use_time_feature),
                    "obs_dim": int(obs_dim),
                    "act_dim": int(act_dim),
                    "n_trajs": int(len(trajs)),
                    "total_transitions": int(total_transitions),
                    "steps_per_task": int(args.steps_per_task),
                    "updates_per_task": int(args.steps_per_task),
                    "approx_updates_per_epoch": int(updates_per_epoch),
                    "approx_epochs": float(approx_epochs),
                    "seq_len_requested": int(args.seq_len),
                    "seq_len_used": int(seq_len_use),
                    "max_ep_len_used": int(max_ep_len_use),
                    "timestep_clip_max": int(timestep_clip_max),
                    "batch_size": int(args.batch_size),
                    "episodes_eval": int(args.episodes_eval),
                    "max_steps": int(max_steps),
                    "gamma": float(args.gamma),
                    "seed": int(args.seed),
                    "device": str(device_t),
                    "rtg_scale": float(args.rtg_scale),
                    "lr": float(args.lr),
                    "weight_decay": float(args.weight_decay),
                    "grad_clip": float(args.grad_clip),
                    "n_heads": int(args.n_heads),
                    "use_prev_action": bool(args.use_prev_action),
                    "selected_target_return": float(selected_target),
                    "best_sweep_target": float(best_target),
                    "best_sweep_score": float(best_score),
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
            "max_ep_len": int(args.max_ep_len),
            "batch_size": int(args.batch_size),
            "episodes_eval": int(args.episodes_eval),
            "max_steps": int(args.max_steps),
            "gamma": float(args.gamma),
            "target_mode": str(args.target_mode),
            "target_return_override": args.target_return,
            "seed": int(args.seed),
            "rtg_scale": float(args.rtg_scale),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "grad_clip": float(args.grad_clip),
            "n_heads": int(args.n_heads),
            "time_feature": str(args.time_feature),
            "use_prev_action": bool(args.use_prev_action),
        },
    )

    rows = [{"task": k, "score": v} for k, v in scores.items()]
    if rows:
        _write_csv(os.path.join(run_dir, "scores.csv"), rows, fieldnames=["task", "score"])

    print(f"\n[artifacts] saved to: {run_dir}")


if __name__ == "__main__":
    main()
