#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from collections import OrderedDict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import panda_gym  # noqa: F401
import numpy as np
import torch

from bin.helper import _set_seed
from dt.utils import evaluate_dt_panda_cl
from paths import RUNS
from bin.config import PANDA_TASKS
from clbench.benchmark.metrics import StandardCLMetrics
from clbench.benchmark.metrics_extra import per_step_report
from clbench.benchmark.runner import BenchmarkResults
from clbench.io.run_logger import build_run_dir, save_json, save_matrix_csv, bench_short

from dt.dataset import Trajectory
from dt.dataset_panda import load_panda_offline_pkl
from strategies.ewc import PandaEWCStrategy
from strategies.naive import PandaNaiveStrategy
from strategies.cumulative import PandaCumulativeReplayStrategy
from strategies.si import PandaSIStrategy
from strategies.tsn_improved_reuse_panda import TSNImprovedReusePandaStrategy
from strategies.tsn_strategy_panda_dt_v1 import TSNPandaStrategy
from strategies.tsn_original_reuse_panda import TSNOriginalReusePandaStrategy

PANDA_OBS_KEYS = ("observation", "achieved_goal", "desired_goal")

HARD_TARGET_RETURNS: Dict[str, float] = {
    "PandaReach": -0.000320,
    "PandaPush": -0.436000,
    "PandaPickAndPlace": -0.001000,
}

class PandaTimeFeatureWrapper(gym.Wrapper):
    """
    Append one normalized time feature to obs["observation"] in Dict observations.
    time = 1 - elapsed/max_steps
    """

    def __init__(self, env: gym.Env, max_steps: Optional[int] = None):
        super().__init__(env)

        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise ValueError("PandaTimeFeatureWrapper requires Dict obs")
        if "observation" not in env.observation_space.spaces:
            raise ValueError("Expected key 'observation'")

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


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _reuse_signature(args) -> str:
    """
    Extra suffix for run_dir so reuse runs are easy to distinguish.
    """
    if getattr(args, "strategy", "") == "tsn_origin_reuse":
        mmc = "inf" if int(getattr(args, "tsn_max_model_copies", 0)) <= 0 else int(args.tsn_max_model_copies)
        return (
            f"__rsm-old"
            f"_mem{int(args.tsn_reuse_memory_size)}"
            f"_kl{float(args.tsn_reuse_kl_threshold):g}"
            f"_mc{mmc}"
        )

    if getattr(args, "strategy", "") != "tsn_improved_reuse":
        return ""

    mode = str(getattr(args, "tsn_reuse_score_mode", "action"))
    if mode == "action":
        return f"__rsm-{mode}_athr{float(args.tsn_action_reuse_threshold):g}"
    if mode == "latent":
        return f"__rsm-{mode}_lthr{float(args.tsn_latent_reuse_threshold):g}"
    return (
        f"__rsm-{mode}"
        f"_hthr{float(args.tsn_hybrid_reuse_threshold):g}"
        f"_ha{float(args.tsn_hybrid_alpha):.2f}"
    )


def _safe_copy_state_stats(strategy: Any) -> Dict[str, Any]:
    """
    Extract copy-level metadata for reuse strategies if present.
    """
    if not hasattr(strategy, "copy_states"):
        return {
            "num_model_copies": None,
            "num_parameters_all_copies_total": None,
            "task_to_copy": None,
            "task_similarity": None,
        }

    copy_states = getattr(strategy, "copy_states", [])
    total_params_all_copies = 0
    try:
        for st in copy_states:
            total_params_all_copies += int(sum(p.numel() for p in st.model.parameters()))
    except Exception:
        total_params_all_copies = None

    return {
        "num_model_copies": int(len(copy_states)),
        "num_parameters_all_copies_total": total_params_all_copies,
        "task_to_copy": dict(getattr(strategy, "task_to_copy", {})),
        "task_similarity": dict(getattr(strategy, "task_similarity", {})),
    }


def _args_model_signature(args) -> str:
    return (
        f"dm{int(args.d_model)}"
        f"_L{int(args.n_layers)}"
        f"_H{int(args.n_heads)}"
        f"_K{int(args.seq_len)}"
        f"_drop{float(args.p_drop):.2f}"
    )


def _dt_cfg_dict_from_model(model: Any) -> Optional[Dict[str, Any]]:
    cfg = getattr(getattr(model, "dt", None), "cfg", None)
    if cfg is None:
        return None
    if is_dataclass(cfg):
        return asdict(cfg)
    try:
        return dict(vars(cfg))
    except Exception:
        return None


def _model_signature_from_model(model: Any, fallback: str) -> str:
    cfg = getattr(getattr(model, "dt", None), "cfg", None)
    if cfg is None:
        return fallback

    n_embd = getattr(cfg, "n_embd", None)
    n_layer = getattr(cfg, "n_layer", None)
    n_head = getattr(cfg, "n_head", None)
    K = getattr(cfg, "K", getattr(model, "seq_len", None))
    dropout = getattr(cfg, "dropout", None)

    if None in (n_embd, n_layer, n_head, K, dropout):
        return fallback

    return (
        f"dm{int(n_embd)}"
        f"_L{int(n_layer)}"
        f"_H{int(n_head)}"
        f"_K{int(K)}"
        f"_drop{float(dropout):.2f}"
    )


def _raw_panda_flat_dim(env_id: str) -> int:
    env = gym.make(env_id)
    try:
        if not isinstance(env.observation_space, gym.spaces.Dict):
            shp = getattr(env.observation_space, "shape", None)
            if shp is None:
                raise ValueError(f"Unsupported obs space: {env.observation_space}")
            return int(np.prod(shp))

        spaces = env.observation_space.spaces
        for k in PANDA_OBS_KEYS:
            if k not in spaces:
                raise ValueError(f"Missing key {k} in env obs space keys={list(spaces.keys())}")

        dim = 0
        for k in PANDA_OBS_KEYS:
            dim += int(np.prod(spaces[k].shape))
        return int(dim)
    finally:
        try:
            env.close()
        except Exception:
            pass


def _needs_time_feature(env_id: str, dataset_obs_dim: int) -> bool:
    raw_dim = _raw_panda_flat_dim(env_id)
    if int(dataset_obs_dim) == int(raw_dim):
        return False
    if int(dataset_obs_dim) == int(raw_dim) + 1:
        return True
    raise ValueError(
        f"Obs dim mismatch: dataset_obs_dim={dataset_obs_dim}, env_raw_flat_dim={raw_dim} "
        f"(expected either raw or raw+1 for time-feature)."
    )


def make_env(env_id: str, *, add_time_feature: bool) -> gym.Env:
    env = gym.make(env_id)
    if add_time_feature:
        env = PandaTimeFeatureWrapper(env)
    return env


def traj_return(tr: Trajectory) -> float:
    return float(np.sum(np.asarray(tr.rewards, dtype=np.float32)))


def ensure_trajectory(obj: Any) -> Trajectory:
    if isinstance(obj, Trajectory):
        return obj
    if isinstance(obj, dict):
        obs = np.asarray(obj["obs"], dtype=np.float32)
        actions = np.asarray(obj["actions"], dtype=np.float32)
        rewards = np.asarray(obj["rewards"], dtype=np.float32)

        if "timesteps" in obj:
            ts = np.asarray(obj["timesteps"], dtype=np.int64)
        else:
            ts = np.arange(actions.shape[0], dtype=np.int64)

        if "returns_to_go" in obj:
            rtg = np.asarray(obj["returns_to_go"], dtype=np.float32)
        else:
            r = rewards.astype(np.float32)
            rtg = np.flip(np.cumsum(np.flip(r, axis=0), axis=0), axis=0).astype(np.float32)

        return Trajectory(obs=obs, actions=actions, rewards=rewards, timesteps=ts, returns_to_go=rtg)

    raise TypeError(f"Unsupported trajectory type: {type(obj)}")


def to_trajectory_list(trajs_raw: Any) -> List[Trajectory]:
    if isinstance(trajs_raw, dict):
        trajs_raw = list(trajs_raw.values())
    if not isinstance(trajs_raw, (list, tuple)):
        raise TypeError(f"Expected list/tuple/dict from loader, got {type(trajs_raw)}")
    return [ensure_trajectory(tr) for tr in trajs_raw]


def pad_2d_last(x: np.ndarray, target_dim: int) -> np.ndarray:
    x = np.asarray(x)
    assert x.ndim == 2, f"Expected [T,D], got {x.shape}"
    d = int(target_dim)
    if x.shape[1] == d:
        return x
    if x.shape[1] > d:
        return x[:, :d]
    out = np.zeros((x.shape[0], d), dtype=x.dtype)
    out[:, : x.shape[1]] = x
    return out


def pad_trajectory(tr: Trajectory, obs_dim: int, act_dim: int) -> Trajectory:
    obs = pad_2d_last(np.asarray(tr.obs, dtype=np.float32), obs_dim).astype(np.float32)
    actions = pad_2d_last(np.asarray(tr.actions, dtype=np.float32), act_dim).astype(np.float32)

    rewards = np.asarray(tr.rewards, dtype=np.float32)
    timesteps = (
        np.asarray(tr.timesteps, dtype=np.int64)
        if hasattr(tr, "timesteps")
        else np.arange(len(rewards), dtype=np.int64)
    )

    if hasattr(tr, "returns_to_go") and tr.returns_to_go is not None:
        rtg = np.asarray(tr.returns_to_go, dtype=np.float32)
    else:
        rtg = np.flip(np.cumsum(np.flip(rewards, axis=0), axis=0), axis=0).astype(np.float32)

    return Trajectory(obs=obs, actions=actions, rewards=rewards, timesteps=timesteps, returns_to_go=rtg)


def _collect_origin_reuse_meta(strategy: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if hasattr(strategy, "task_to_copy"):
        out["tsn_task_to_copy"] = {str(k): int(v) for k, v in getattr(strategy, "task_to_copy", {}).items()}
    if hasattr(strategy, "task_similarity"):
        sim = {}
        for k, v in getattr(strategy, "task_similarity", {}).items():
            sim[str(k)] = v
        out["tsn_task_similarity"] = sim
    if hasattr(strategy, "copy_states"):
        out["num_model_copies"] = int(len(getattr(strategy, "copy_states", [])))
        try:
            out["num_parameters_all_copies_total"] = int(
                sum(sum(p.numel() for p in st.model.parameters()) for st in strategy.copy_states)
            )
        except Exception:
            out["num_parameters_all_copies_total"] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=str, default=str(RUNS))
    ap.add_argument("--strategy", choices=["cumulative", "ewc", "naive", "si", "tsn", "tsn_origin_reuse", "tsn_improved_reuse"], default="cumulative")
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--steps-per-task", type=int, default=1_000_000)
    ap.add_argument("--episodes-eval", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--batch-size", type=int, default=128)

    # replay knobs
    ap.add_argument("--mix", type=float, default=0.5, help="Replay fraction in each minibatch.")
    ap.add_argument("--rehearsal-capacity", type=int, default=5000)

    # model/optim
    ap.add_argument("--target-mode", choices=["max", "p90", "mean"], default="max")
    ap.add_argument("--target-return", type=float, default=None, help="If set, overrides per-task target_return.")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--n-heads", type=int, default=1)
    ap.add_argument("--p-drop", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=0.25)
    ap.add_argument("--max-ep-len", type=int, default=50)
    ap.add_argument("--rtg-scale", type=float, default=1000.0)

    # EWC hyperparams
    ap.add_argument("--ewc-lambda", type=float, default=50.0)
    ap.add_argument("--fisher-n-batches", type=int, default=50)
    ap.add_argument("--fisher-batch-size", type=int, default=32)

    # SI hyperparams
    ap.add_argument("--si-lambda", type=float, default=0.5)
    ap.add_argument("--si-epsilon", type=float, default=0.1)
    ap.add_argument("--si-omega-max", type=float, default=10.0)
    ap.add_argument("--no-si-clamp-min0", dest="si_clamp_min0", action="store_false", default=True)
    ap.add_argument("--si-debug-every", type=int, default=0)

    # TSN hyperparams
    ap.add_argument("--tsn-keep-ratio", type=float, default=0.5)
    ap.add_argument("--tsn-quant-clusters", type=int, default=16)
    ap.add_argument("--tsn-include-embeddings", action="store_true", default=False)
    ap.add_argument("--tsn-allow-weight-reuse", action="store_true", default=False)
    ap.add_argument("--no-tsn-quantize-after-task", dest="tsn_quantize_after_task", action="store_false", default=True)
    ap.add_argument(
        "--no-tsn-freeze-non-mask-params-after-first",
        dest="tsn_freeze_non_mask_params_after_first",
        action="store_false",
        default=True,
    )
    ap.add_argument("--no-tsn-store-task-obs-stats", dest="tsn_store_task_obs_stats", action="store_false", default=True)
    ap.add_argument("--no-tsn-patch-model-act", dest="tsn_patch_model_act", action="store_false", default=True)

    ap.add_argument("--tsn-reuse-kl-threshold", type=float, default=0.25)

    # Reuse extras (origin + improved)
    ap.add_argument("--tsn-reuse-memory-size", type=int, default=256)
    ap.add_argument(
        "--tsn-max-model-copies",
        type=int,
        default=0,
        help="0 means no explicit limit for reuse copies.",
    )

    # Improved reuse extras
    ap.add_argument(
        "--tsn-reuse-score-mode",
        choices=["action", "latent", "hybrid"],
        default="action",
        help="Routing score used by tsn_improved_reuse.",
    )
    ap.add_argument("--tsn-routing-n-batches", type=int, default=4)
    ap.add_argument("--tsn-routing-batch-size", type=int, default=64)

    ap.add_argument("--tsn-action-reuse-threshold", type=float, default=0.05)
    ap.add_argument("--tsn-latent-reuse-threshold", type=float, default=25.0)

    # TEGO CI BRAKUJE:
    ap.add_argument("--tsn-hybrid-reuse-threshold", type=float, default=0.50)
    ap.add_argument("--tsn-hybrid-alpha", type=float, default=0.70)

    ap.add_argument(
        "--no-tsn-normalize-similarity-scores",
        dest="tsn_normalize_similarity_scores",
        action="store_false",
        default=True,
    )
    ap.add_argument(
        "--no-tsn-warmstart-source-scores",
        dest="tsn_warmstart_source_scores",
        action="store_false",
        default=True,
    )
    ap.add_argument("--tsn-warmstart-strength", type=float, default=2.0)
    ap.add_argument("--tsn-warmstart-noise-std", type=float, default=0.02)
    ap.add_argument("--tsn-warmstart-on-new-copy", action="store_true")

    args = ap.parse_args()

    _set_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"[device] {device}")

    # task order
    TASK_ORDER = ["PandaReach", "PandaPush", "PandaPickAndPlace"]
    # TASK_ORDER = ["PandaPickAndPlace", "PandaPush", "PandaReach"]

    panda_tasks: List[Tuple[str, str, Path]] = []
    for name in TASK_ORDER:
        cfg = PANDA_TASKS[name]
        panda_tasks.append((name, cfg["env_id"], Path(cfg["dataset"])))

    # Load envs + datasets
    envs: "OrderedDict[str, gym.Env]" = OrderedDict()
    all_trajs: List[List[Trajectory]] = []
    obs_dims: List[int] = []
    act_dims: List[int] = []
    task_records: List[Dict[str, Any]] = []

    print("\n=== Loading Panda tasks ===")
    for name, env_id, ds_path in panda_tasks:
        print(f"[TASK] {name} | env={env_id} | data={ds_path}")

        trajs_raw = load_panda_offline_pkl(str(ds_path), obs_keys=PANDA_OBS_KEYS)
        trajs = to_trajectory_list(trajs_raw)
        assert len(trajs) > 0, f"No trajectories loaded from {ds_path}"

        t0 = trajs[0]
        o0 = np.asarray(t0.obs)
        a0 = np.asarray(t0.actions)
        dataset_obs_dim = int(o0.shape[-1])

        use_time = _needs_time_feature(env_id, dataset_obs_dim)
        env = make_env(env_id, add_time_feature=use_time)
        envs[name] = env

        obs_dims.append(int(o0.shape[-1]))
        act_dims.append(int(a0.shape[-1]))

        rets = np.array([traj_return(t) for t in trajs], dtype=np.float32)
        print(f"  use_time_feature={use_time}")
        print(f"  dataset obs shape: {o0.shape}, actions shape: {a0.shape}")
        print(f"  env obs_space: {env.observation_space}, act_space: {env.action_space}")
        print(f"  dataset episodes={len(trajs)} | return mean={rets.mean():.3f} min={rets.min():.3f} max={rets.max():.3f}")

        task_records.append({
            "name": name,
            "env_id": env_id,
            "dataset": str(ds_path),
            "use_time_feature": bool(use_time),
            "dataset_episodes": int(len(trajs)),
            "dataset_obs_dim": int(o0.shape[-1]),
            "dataset_act_dim": int(a0.shape[-1]),
            "dataset_return_mean": float(rets.mean()),
            "dataset_return_min": float(rets.min()),
            "dataset_return_max": float(rets.max()),
        })

        all_trajs.append(trajs)

    # max_len per task (for timestep_clip)
    max_len_map: Dict[str, int] = {}
    for idx, (name, _env_id, _ds_path) in enumerate(panda_tasks):
        max_len = max(int(len(tr.actions)) for tr in all_trajs[idx])
        max_len_map[name] = max_len
        print(f"[lens] {name}: max_len={max_len} -> timestep_clip_max={max_len - 1}")
        task_records[idx]["max_len"] = int(max_len)
        task_records[idx]["timestep_clip_max"] = int(max_len - 1)

    # Global dims
    obs_dim_global = int(max(obs_dims))
    act_dim_global = int(max(act_dims))
    print(f"\n[global dims] obs_dim={obs_dim_global} (per-task={obs_dims}), act_dim={act_dim_global} (per-task={act_dims})")

    for i in range(len(all_trajs)):
        all_trajs[i] = [pad_trajectory(tr, obs_dim_global, act_dim_global) for tr in all_trajs[i]]

    # target_return per task
    target_return_map: Dict[str, float] = {}
    for idx, (name, _env_id, _ds_path) in enumerate(panda_tasks):
        if args.target_return is not None:
            target = float(args.target_return)
            src = "global --target-return"
        else:
            target = float(HARD_TARGET_RETURNS[name])
            src = "hardcoded"
        target_return_map[name] = float(target)
        task_records[idx]["target_return"] = float(target)
        task_records[idx]["target_return_source"] = src
        print(f"[target] {name}: target_return={target:.6f} ({src})")

    # strategy
    task_names = list(envs.keys())
    n = len(task_names)
    P = np.zeros((n, n), dtype=np.float32)

    obs_shape = (obs_dim_global,)
    if args.strategy == "naive":
        strategy = PandaNaiveStrategy(obs_shape, act_dim_global, args.seq_len, device)
    elif args.strategy == "cumulative":
        strategy = PandaCumulativeReplayStrategy(
            obs_shape,
            act_dim_global,
            args.seq_len,
            device,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            p_drop=args.p_drop,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            rehearsal_capacity=args.rehearsal_capacity,
            max_ep_len=args.max_ep_len,
            rtg_scale=args.rtg_scale,
        )
    elif args.strategy == "ewc":
        strategy = PandaEWCStrategy(
            obs_shape,
            act_dim_global,
            args.seq_len,
            device,
            n_heads=args.n_heads,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            max_ep_len=args.max_ep_len,
            rtg_scale=args.rtg_scale,
            ewc_lambda=args.ewc_lambda,
            fisher_n_batches=args.fisher_n_batches,
            fisher_batch_size=args.fisher_batch_size,
        )
    elif args.strategy == "si":
        strategy = PandaSIStrategy(
            obs_shape,
            act_dim_global,
            args.seq_len,
            device,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            p_drop=args.p_drop,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            max_ep_len=args.max_ep_len,
            rtg_scale=args.rtg_scale,
            si_lambda=args.si_lambda,
            si_epsilon=args.si_epsilon,
            omega_max=args.si_omega_max,
            clamp_min0=args.si_clamp_min0,
            debug_every=args.si_debug_every,
        )
    elif args.strategy == "tsn":
        strategy = TSNPandaStrategy(
            obs_dim=obs_dim_global,
            act_dim=act_dim_global,
            seq_len=args.seq_len,
            device=device,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            p_drop=args.p_drop,
            max_ep_len=args.max_ep_len,
            rtg_scale=args.rtg_scale,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            keep_ratio=args.tsn_keep_ratio,
            include_embeddings=args.tsn_include_embeddings,
            quantize_after_task=args.tsn_quantize_after_task,
            quant_clusters=args.tsn_quant_clusters,
            allow_weight_reuse=args.tsn_allow_weight_reuse,
            freeze_non_mask_params_after_first=args.tsn_freeze_non_mask_params_after_first,
            store_task_obs_stats=args.tsn_store_task_obs_stats,
            patch_model_act=args.tsn_patch_model_act,
        )
    elif args.strategy == "tsn_origin_reuse":
        max_model_copies = None if int(args.tsn_max_model_copies) <= 0 else int(args.tsn_max_model_copies)
        strategy = TSNOriginalReusePandaStrategy(
            obs_dim=obs_dim_global,
            act_dim=act_dim_global,
            seq_len=args.seq_len,
            device=device,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            p_drop=args.p_drop,
            max_ep_len=args.max_ep_len,
            rtg_scale=args.rtg_scale,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            keep_ratio=args.tsn_keep_ratio,
            include_embeddings=args.tsn_include_embeddings,
            quantize_after_task=args.tsn_quantize_after_task,
            quant_clusters=args.tsn_quant_clusters,
            freeze_non_mask_params_after_first=args.tsn_freeze_non_mask_params_after_first,
            expected_num_tasks=int(n),
            keep_ratio_schedule="constant",
            min_keep_ratio=1e-3,
            reuse_memory_size=args.tsn_reuse_memory_size,
            reuse_kl_threshold=args.tsn_reuse_kl_threshold,
            max_model_copies=max_model_copies,
            store_task_obs_stats=args.tsn_store_task_obs_stats,
            patch_model_act=args.tsn_patch_model_act,
        )
    elif args.strategy == "tsn_improved_reuse":
        max_model_copies = None if int(args.tsn_max_model_copies) <= 0 else int(args.tsn_max_model_copies)
        strategy = TSNImprovedReusePandaStrategy(
            obs_dim=obs_dim_global,
            act_dim=act_dim_global,
            seq_len=args.seq_len,
            device=device,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            p_drop=args.p_drop,
            max_ep_len=args.max_ep_len,
            rtg_scale=args.rtg_scale,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            keep_ratio=args.tsn_keep_ratio,
            include_embeddings=args.tsn_include_embeddings,
            quantize_after_task=args.tsn_quantize_after_task,
            quant_clusters=args.tsn_quant_clusters,
            freeze_non_mask_params_after_first=args.tsn_freeze_non_mask_params_after_first,
            store_task_obs_stats=args.tsn_store_task_obs_stats,
            patch_model_act=args.tsn_patch_model_act,
            expected_num_tasks=int(n),
            keep_ratio_schedule="constant",
            min_keep_ratio=1e-3,
            reuse_memory_size=int(args.tsn_reuse_memory_size),
            reuse_kl_threshold=float(args.tsn_reuse_kl_threshold),
            max_model_copies=max_model_copies,
            reuse_score_mode=str(args.tsn_reuse_score_mode),
            routing_n_batches=int(args.tsn_routing_n_batches),
            routing_batch_size=int(args.tsn_routing_batch_size),
            action_reuse_threshold=float(args.tsn_action_reuse_threshold),
            latent_reuse_threshold=float(args.tsn_latent_reuse_threshold),
            hybrid_reuse_threshold=float(args.tsn_hybrid_reuse_threshold),
            hybrid_alpha=float(args.tsn_hybrid_alpha),
            normalize_similarity_scores=bool(args.tsn_normalize_similarity_scores),
            warmstart_source_scores=bool(args.tsn_warmstart_source_scores),
            warmstart_strength=float(args.tsn_warmstart_strength),
            warmstart_noise_std=float(args.tsn_warmstart_noise_std),
            warmstart_on_new_copy=bool(args.tsn_warmstart_on_new_copy),
        )
    else:
        raise ValueError(f"Unsupported strategy: {args.strategy}")

    # run dir with model signature
    bench = "panda"
    spec_tag = "panda3"
    model_sig = _model_signature_from_model(strategy.model, fallback=_args_model_signature(args))
    run_tag = f"{(args.tag or spec_tag)}__{model_sig}{_reuse_signature(args)}"
    run_dir = build_run_dir(args.runs_root, bench, args.strategy, tag=run_tag)
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n[run_dir] {run_dir}")
    print(f"[tasks] {task_names}")
    print(f"[model_signature] {model_sig}")

    # task loop
    for i, (task_name, _env_train) in enumerate(envs.items()):
        print(f"\n[Task {i + 1}/{n}] Train on {task_name}")
        task_trajs = all_trajs[i]
        active_action_dim_mask = [1] * int(act_dims[i]) + [0] * max(0, act_dim_global - int(act_dims[i]))
        task_records[i]["active_action_dim_mask"] = list(active_action_dim_mask)

        if args.strategy == "cumulative":
            strategy.train_task(
                task_trajs,
                steps=args.steps_per_task,
                batch_size=args.batch_size,
                mix=args.mix,
            )
        elif args.strategy in ("tsn", "tsn_origin_reuse", "tsn_improved_reuse"):
            strategy.train_task(
                task_trajs,
                steps=args.steps_per_task,
                batch_size=args.batch_size,
                active_action_dim_mask=active_action_dim_mask,
            )
        else:
            strategy.train_task(
                task_trajs,
                steps=args.steps_per_task,
                batch_size=args.batch_size,
            )

        strategy.after_task(task_trajs)

        for j, (eval_name, env_eval) in enumerate(envs.items()):
            if hasattr(strategy, "has_task_mask") and hasattr(strategy, "set_eval_task"):
                if strategy.has_task_mask(j):
                    strategy.set_eval_task(j)
                elif hasattr(strategy, "clear_eval_task"):
                    strategy.clear_eval_task()
            score = evaluate_dt_panda_cl(
                model=strategy.model,
                env=env_eval,
                episodes=int(args.episodes_eval),
                device=torch.device(device),
                max_steps=int(args.max_steps),
                target_return=float(target_return_map[eval_name]),
                seed=int(args.seed + 1000 * j),
                obs_keys=PANDA_OBS_KEYS,
                obs_pad_to=obs_dim_global,
                act_pad_to=act_dim_global,
                timestep_clip_max=int(max_len_map[eval_name] - 1),
                gamma=1.0,
                clip_action=True,
            )
            P[i, j] = float(score)
            print(f"[eval] after task {i + 1} on {eval_name}: {score:.3f} (target={target_return_map[eval_name]:.6f})")

    # metrics + save
    results = BenchmarkResults(
        name=f"DT-{args.strategy}:panda3",
        task_names=task_names,
        perf_matrix=P,
    )
    metrics = StandardCLMetrics.compute(results)

    num_params_total = int(sum(p.numel() for p in strategy.model.parameters()))
    num_params_trainable = int(sum(p.numel() for p in strategy.model.parameters() if p.requires_grad))
    dt_cfg_dict = _dt_cfg_dict_from_model(strategy.model)

    copy_stats = _safe_copy_state_stats(strategy)

    model_info = {
        "signature": model_sig,
        "class": type(strategy.model).__name__,
        "obs_shape": list(obs_shape),
        "obs_dim_global": int(obs_dim_global),
        "act_dim_global": int(act_dim_global),
        "seq_len": int(args.seq_len),
        "d_model_requested": int(args.d_model),
        "n_layers_requested": int(args.n_layers),
        "n_heads_requested": int(args.n_heads),
        "p_drop_requested": float(args.p_drop),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "grad_clip": float(args.grad_clip),
        "max_ep_len_requested": int(args.max_ep_len),
        "max_ep_len_effective": int(getattr(strategy.model, "max_ep_len", args.max_ep_len)),
        "rtg_scale_requested": float(args.rtg_scale),
        "rtg_scale_effective": float(getattr(strategy.model, "rtg_scale", args.rtg_scale)),
        "num_parameters_total": num_params_total,
        "num_parameters_trainable": num_params_trainable,
        "num_model_copies": copy_stats["num_model_copies"],
        "num_parameters_all_copies_total": copy_stats["num_parameters_all_copies_total"],
        "dt_cfg": dt_cfg_dict,
    }

    extra_origin_reuse = _collect_origin_reuse_meta(strategy) if args.strategy == "tsn_origin_reuse" else {}

    is_tsn_core = args.strategy == "tsn"
    is_tsn_origin_reuse = args.strategy == "tsn_origin_reuse"
    is_tsn_improved_reuse = args.strategy == "tsn_improved_reuse"
    is_tsn_like = args.strategy in ("tsn", "tsn_origin_reuse", "tsn_improved_reuse")
    is_tsn_reuse = args.strategy in ("tsn_origin_reuse", "tsn_improved_reuse")

    save_payload = {
        "name": results.name,
        "task_names": task_names,
        "perf_matrix": P.tolist(),
        "metrics": metrics,
        "model": model_info,
        "tasks": task_records,
        "task_order": TASK_ORDER,
        "obs_dim_global": obs_dim_global,
        "act_dim_global": act_dim_global,
        "target_return_map": target_return_map,
        "seed": args.seed,
        "strategy": args.strategy,
        "seq_len": args.seq_len,
        "steps_per_task": args.steps_per_task,
        "episodes_eval": args.episodes_eval,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "mix": args.mix,
        "rehearsal_capacity": args.rehearsal_capacity,
        "target_mode": args.target_mode,
        "target_return_override": None if args.target_return is None else float(args.target_return),
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "p_drop": args.p_drop,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "max_ep_len": args.max_ep_len,
        "rtg_scale": args.rtg_scale,
        "ewc_lambda": args.ewc_lambda if args.strategy == "ewc" else None,
        "fisher_n_batches": args.fisher_n_batches if args.strategy == "ewc" else None,
        "fisher_batch_size": args.fisher_batch_size if args.strategy == "ewc" else None,
        "si_lambda": args.si_lambda if args.strategy == "si" else None,
        "si_epsilon": args.si_epsilon if args.strategy == "si" else None,
        "si_omega_max": args.si_omega_max if args.strategy == "si" else None,
        "si_clamp_min0": args.si_clamp_min0 if args.strategy == "si" else None,
        "si_debug_every": args.si_debug_every if args.strategy == "si" else None,
                "tsn_keep_ratio": args.tsn_keep_ratio if is_tsn_like else None,
        "tsn_quant_clusters": args.tsn_quant_clusters if is_tsn_like else None,
        "tsn_include_embeddings": args.tsn_include_embeddings if is_tsn_like else None,
        "tsn_allow_weight_reuse": args.tsn_allow_weight_reuse if is_tsn_core else None,
        "tsn_quantize_after_task": args.tsn_quantize_after_task if is_tsn_like else None,
        "tsn_freeze_non_mask_params_after_first": args.tsn_freeze_non_mask_params_after_first if is_tsn_like else None,
        "tsn_store_task_obs_stats": args.tsn_store_task_obs_stats if is_tsn_like else None,
        "tsn_patch_model_act": args.tsn_patch_model_act if is_tsn_like else None,

        "tsn_reuse_memory_size": int(args.tsn_reuse_memory_size) if is_tsn_reuse else None,
        "tsn_reuse_kl_threshold": float(args.tsn_reuse_kl_threshold) if is_tsn_reuse else None,
        "tsn_max_model_copies": (
            None if (not is_tsn_reuse or int(args.tsn_max_model_copies) <= 0) else int(args.tsn_max_model_copies)
        ),
        "tsn_task_to_copy": copy_stats["task_to_copy"] if is_tsn_reuse else None,
        "tsn_task_similarity": copy_stats["task_similarity"] if is_tsn_reuse else None,

        "tsn_reuse_score_mode": str(args.tsn_reuse_score_mode) if is_tsn_improved_reuse else None,
        "tsn_routing_n_batches": int(args.tsn_routing_n_batches) if is_tsn_improved_reuse else None,
        "tsn_routing_batch_size": int(args.tsn_routing_batch_size) if is_tsn_improved_reuse else None,
        "tsn_action_reuse_threshold": float(args.tsn_action_reuse_threshold) if is_tsn_improved_reuse else None,
        "tsn_latent_reuse_threshold": float(args.tsn_latent_reuse_threshold) if is_tsn_improved_reuse else None,
        "tsn_hybrid_reuse_threshold": float(args.tsn_hybrid_reuse_threshold) if is_tsn_improved_reuse else None,
        "tsn_hybrid_alpha": float(args.tsn_hybrid_alpha) if is_tsn_improved_reuse else None,
        "tsn_normalize_similarity_scores": bool(args.tsn_normalize_similarity_scores) if is_tsn_improved_reuse else None,
        "tsn_warmstart_source_scores": bool(args.tsn_warmstart_source_scores) if is_tsn_improved_reuse else None,
        "tsn_warmstart_strength": float(args.tsn_warmstart_strength) if is_tsn_improved_reuse else None,
        "tsn_warmstart_noise_std": float(args.tsn_warmstart_noise_std) if is_tsn_improved_reuse else None,
        "tsn_warmstart_on_new_copy": bool(args.tsn_warmstart_on_new_copy) if is_tsn_improved_reuse else None,
        "tsn_reuse_signature": _reuse_signature(args).lstrip("_") if is_tsn_reuse else None,
    }
    save_payload.update(extra_origin_reuse)

    save_json(os.path.join(run_dir, "results.json"), save_payload)

    save_json(os.path.join(run_dir, f"rez_{bench_short(bench)}.json"), {
        "metrics": metrics,
        "strategy": args.strategy,
        "model_signature": model_sig,
        "tsn_reuse_score_mode": str(args.tsn_reuse_score_mode) if args.strategy == "tsn_improved_reuse" else None,
        "tsn_reuse_signature": _reuse_signature(args).lstrip("_") if args.strategy in ("tsn_origin_reuse",
                                                                                       "tsn_improved_reuse") else None,
    })
    save_matrix_csv(os.path.join(run_dir, "matrix.csv"), task_names, P)

    steps = per_step_report(task_names, P)
    save_json(os.path.join(run_dir, "per_step.json"), {"per_step": steps})
    if steps:
        with open(os.path.join(run_dir, "per_step.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(steps[0].keys()))
            w.writeheader()
            w.writerows(steps)

    rez_payload = {
        "metrics": metrics,
        "strategy": args.strategy,
        "model_signature": model_sig,
    }
    rez_payload.update(extra_origin_reuse)

    for env in envs.values():
        try:
            env.close()
        except Exception:
            pass

    print("\nDone. Saved Panda CL results to:", run_dir)


if __name__ == "__main__":
    main()
