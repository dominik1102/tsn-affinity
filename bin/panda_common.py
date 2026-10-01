"""Helpers shared by the Panda Decision Transformer runner scripts."""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import gymnasium as gym
import numpy as np
import torch

from bin.config import PANDA_TASKS
from clbench.benchmark.metrics import StandardCLMetrics
from clbench.benchmark.metrics_extra import per_step_report
from clbench.benchmark.runner import BenchmarkResults
from clbench.io.run_logger import bench_short, save_json, save_matrix_csv
from dt.dataset import Trajectory
from dt.dataset_panda import load_panda_offline_pkl
from dt.utils import evaluate_dt_panda_cl
from paths import RUNS
from strategies.cumulative import PandaCumulativeReplayStrategy
from strategies.ewc import PandaEWCStrategy
from strategies.naive import PandaNaiveStrategy
from strategies.si import PandaSIStrategy
from strategies.tsn_improved_reuse_panda import TSNImprovedReusePandaStrategy
from strategies.tsn_original_reuse_panda import TSNOriginalReusePandaStrategy
from strategies.tsn_strategy_panda_dt_v1 import TSNPandaStrategy

PANDA_OBS_KEYS = ("observation", "achieved_goal", "desired_goal")

TASK_ORDER = ["PandaReach", "PandaPush", "PandaPickAndPlace"]

HARD_TARGET_RETURNS: Dict[str, float] = {
    "PandaReach": -0.000320,
    "PandaPush": -0.436000,
    "PandaPickAndPlace": -0.001000,
}

STRATEGIES = ("cumulative", "ewc", "naive", "si", "tsn", "tsn_origin_reuse", "tsn_improved_reuse")
TSN_REUSE_STRATEGIES = ("tsn_origin_reuse", "tsn_improved_reuse")
TSN_STRATEGIES = ("tsn",) + TSN_REUSE_STRATEGIES

BENCH = "panda"
SPEC_TAG = "panda3"


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

        if max_steps is None:
            max_steps = getattr(getattr(env, "spec", None), "max_episode_steps", None)
        if max_steps is None:
            max_steps = 50
        self.max_steps = max(1, int(max_steps))
        self.elapsed_steps = 0

        low = np.asarray(base.low, dtype=np.float32).reshape(-1)
        high = np.asarray(base.high, dtype=np.float32).reshape(-1)

        spaces = dict(env.observation_space.spaces)
        spaces["observation"] = gym.spaces.Box(
            low=np.append(low, np.float32(0.0)),
            high=np.append(high, np.float32(1.0)),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(spaces)

    def _time_value(self) -> np.ndarray:
        v = 1.0 - float(self.elapsed_steps) / float(self.max_steps)
        return np.array([np.clip(v, 0.0, 1.0)], dtype=np.float32)

    def _augment(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        out = dict(obs)
        base = np.asarray(out["observation"], dtype=np.float32).reshape(-1)
        out["observation"] = np.concatenate([base, self._time_value()], axis=0)
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
# Signatures / metadata
# ------------------------------------------------------------
def _max_model_copies(args) -> Optional[int]:
    """`--tsn-max-model-copies <= 0` means no limit."""
    n = int(args.tsn_max_model_copies)
    return None if n <= 0 else n


def _reuse_signature(args) -> str:
    """
    Extra suffix for run_dir so reuse runs are easy to distinguish.
    """
    if args.strategy == "tsn_origin_reuse":
        mmc = _max_model_copies(args) or "inf"
        return (
            f"__rsm-old"
            f"_mem{int(args.tsn_reuse_memory_size)}"
            f"_kl{float(args.tsn_reuse_kl_threshold):g}"
            f"_mc{mmc}"
        )

    if args.strategy != "tsn_improved_reuse":
        return ""

    mode = str(args.tsn_reuse_score_mode)
    if mode == "action":
        return f"__rsm-{mode}_athr{float(args.tsn_action_reuse_threshold):g}"
    if mode == "latent":
        return f"__rsm-{mode}_lthr{float(args.tsn_latent_reuse_threshold):g}"
    return (
        f"__rsm-{mode}"
        f"_hthr{float(args.tsn_hybrid_reuse_threshold):g}"
        f"_ha{float(args.tsn_hybrid_alpha):.2f}"
    )


def _count_params_all_copies(strategy: Any) -> Optional[int]:
    try:
        return int(sum(sum(p.numel() for p in st.model.parameters()) for st in strategy.copy_states))
    except Exception:
        return None


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

    return {
        "num_model_copies": int(len(strategy.copy_states)),
        "num_parameters_all_copies_total": _count_params_all_copies(strategy),
        "task_to_copy": dict(getattr(strategy, "task_to_copy", {})),
        "task_similarity": dict(getattr(strategy, "task_similarity", {})),
    }


def _collect_origin_reuse_meta(strategy: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if hasattr(strategy, "task_to_copy"):
        out["tsn_task_to_copy"] = {str(k): int(v) for k, v in strategy.task_to_copy.items()}
    if hasattr(strategy, "task_similarity"):
        out["tsn_task_similarity"] = {str(k): v for k, v in strategy.task_similarity.items()}
    if hasattr(strategy, "copy_states"):
        out["num_model_copies"] = int(len(strategy.copy_states))
        out["num_parameters_all_copies_total"] = _count_params_all_copies(strategy)
    return out


def _format_model_signature(d_model: Any, n_layers: Any, n_heads: Any, seq_len: Any, dropout: Any) -> str:
    return (
        f"dm{int(d_model)}"
        f"_L{int(n_layers)}"
        f"_H{int(n_heads)}"
        f"_K{int(seq_len)}"
        f"_drop{float(dropout):.2f}"
    )


def _args_model_signature(args) -> str:
    return _format_model_signature(args.d_model, args.n_layers, args.n_heads, args.seq_len, args.p_drop)


def _model_signature_from_model(model: Any, fallback: str) -> str:
    cfg = getattr(getattr(model, "dt", None), "cfg", None)
    if cfg is None:
        return fallback

    values = (
        getattr(cfg, "n_embd", None),
        getattr(cfg, "n_layer", None),
        getattr(cfg, "n_head", None),
        getattr(cfg, "K", getattr(model, "seq_len", None)),
        getattr(cfg, "dropout", None),
    )
    if None in values:
        return fallback
    return _format_model_signature(*values)


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


# ------------------------------------------------------------
# Environments
# ------------------------------------------------------------
def _close_env(env: gym.Env) -> None:
    try:
        env.close()
    except Exception:
        pass


def _raw_panda_flat_dim(env_id: str) -> int:
    env = gym.make(env_id)
    try:
        space = env.observation_space
        if not isinstance(space, gym.spaces.Dict):
            shp = getattr(space, "shape", None)
            if shp is None:
                raise ValueError(f"Unsupported obs space: {space}")
            return int(np.prod(shp))

        missing = [k for k in PANDA_OBS_KEYS if k not in space.spaces]
        if missing:
            raise ValueError(f"Missing key {missing[0]} in env obs space keys={list(space.spaces.keys())}")
        return int(sum(np.prod(space.spaces[k].shape) for k in PANDA_OBS_KEYS))
    finally:
        _close_env(env)


def _needs_time_feature(env_id: str, dataset_obs_dim: int) -> bool:
    raw_dim = _raw_panda_flat_dim(env_id)
    if int(dataset_obs_dim) == raw_dim:
        return False
    if int(dataset_obs_dim) == raw_dim + 1:
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


def _resolve_device(requested: str) -> str:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    return "cuda" if (requested == "cuda" and torch.cuda.is_available()) else "cpu"


# ------------------------------------------------------------
# Trajectories
# ------------------------------------------------------------
def _returns_to_go(rewards: np.ndarray) -> np.ndarray:
    r = np.asarray(rewards, dtype=np.float32)
    return np.flip(np.cumsum(np.flip(r, axis=0), axis=0), axis=0).astype(np.float32)


def traj_return(tr: Trajectory) -> float:
    return float(np.sum(np.asarray(tr.rewards, dtype=np.float32)))


def ensure_trajectory(obj: Any) -> Trajectory:
    if isinstance(obj, Trajectory):
        return obj
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported trajectory type: {type(obj)}")

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
        rtg = _returns_to_go(rewards)

    return Trajectory(obs=obs, actions=actions, rewards=rewards, timesteps=ts, returns_to_go=rtg)


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
    obs = pad_2d_last(np.asarray(tr.obs, dtype=np.float32), obs_dim)
    actions = pad_2d_last(np.asarray(tr.actions, dtype=np.float32), act_dim)
    rewards = np.asarray(tr.rewards, dtype=np.float32)

    if hasattr(tr, "timesteps"):
        timesteps = np.asarray(tr.timesteps, dtype=np.int64)
    else:
        timesteps = np.arange(len(rewards), dtype=np.int64)

    if getattr(tr, "returns_to_go", None) is not None:
        rtg = np.asarray(tr.returns_to_go, dtype=np.float32)
    else:
        rtg = _returns_to_go(rewards)

    return Trajectory(obs=obs, actions=actions, rewards=rewards, timesteps=timesteps, returns_to_go=rtg)


# ------------------------------------------------------------
# Task loading
# ------------------------------------------------------------
@dataclass
class PandaTask:
    name: str
    env_id: str
    dataset: Path
    env: gym.Env
    trajs: List[Trajectory]
    obs_dim: int
    act_dim: int
    max_len: int = 0
    target_return: float = 0.0
    record: Dict[str, Any] = field(default_factory=dict)


def _load_task(name: str) -> PandaTask:
    cfg = PANDA_TASKS[name]
    env_id, ds_path = cfg["env_id"], Path(cfg["dataset"])
    print(f"[TASK] {name} | env={env_id} | data={ds_path}")

    trajs = to_trajectory_list(load_panda_offline_pkl(str(ds_path), obs_keys=PANDA_OBS_KEYS))
    assert len(trajs) > 0, f"No trajectories loaded from {ds_path}"

    o0 = np.asarray(trajs[0].obs)
    a0 = np.asarray(trajs[0].actions)
    obs_dim, act_dim = int(o0.shape[-1]), int(a0.shape[-1])

    use_time = _needs_time_feature(env_id, obs_dim)
    env = make_env(env_id, add_time_feature=use_time)

    rets = np.array([traj_return(t) for t in trajs], dtype=np.float32)
    print(f"  use_time_feature={use_time}")
    print(f"  dataset obs shape: {o0.shape}, actions shape: {a0.shape}")
    print(f"  env obs_space: {env.observation_space}, act_space: {env.action_space}")
    print(f"  dataset episodes={len(trajs)} | return mean={rets.mean():.3f} min={rets.min():.3f} max={rets.max():.3f}")

    record = {
        "name": name,
        "env_id": env_id,
        "dataset": str(ds_path),
        "use_time_feature": bool(use_time),
        "dataset_episodes": int(len(trajs)),
        "dataset_obs_dim": obs_dim,
        "dataset_act_dim": act_dim,
        "dataset_return_mean": float(rets.mean()),
        "dataset_return_min": float(rets.min()),
        "dataset_return_max": float(rets.max()),
    }
    return PandaTask(
        name=name, env_id=env_id, dataset=ds_path, env=env, trajs=trajs,
        obs_dim=obs_dim, act_dim=act_dim, record=record,
    )


def load_padded_tasks() -> List[PandaTask]:
    """Load all TASK_ORDER tasks and pad their trajectories to the global obs/act dims."""
    print("\n=== Loading Panda tasks ===")
    tasks = [_load_task(name) for name in TASK_ORDER]

    # max_len per task (for timestep_clip)
    for t in tasks:
        t.max_len = max(int(len(tr.actions)) for tr in t.trajs)
        print(f"[lens] {t.name}: max_len={t.max_len} -> timestep_clip_max={t.max_len - 1}")
        t.record["max_len"] = int(t.max_len)
        t.record["timestep_clip_max"] = int(t.max_len - 1)

    obs_dims = [t.obs_dim for t in tasks]
    act_dims = [t.act_dim for t in tasks]
    obs_dim_global, act_dim_global = max(obs_dims), max(act_dims)
    print(f"\n[global dims] obs_dim={obs_dim_global} (per-task={obs_dims}), act_dim={act_dim_global} (per-task={act_dims})")

    for t in tasks:
        t.trajs = [pad_trajectory(tr, obs_dim_global, act_dim_global) for tr in t.trajs]
    return tasks


# ------------------------------------------------------------
# Strategy
# ------------------------------------------------------------
def build_strategy(
    args,
    obs_dim: int,
    act_dim: int,
    device: str,
    n_tasks: int,
    *,
    improved_reuse_extra: Optional[Dict[str, Any]] = None,
):
    """`improved_reuse_extra` holds additional kwargs passed only to TSNImprovedReusePandaStrategy."""
    obs_shape = (obs_dim,)
    s = args.strategy

    if s == "naive":
        return PandaNaiveStrategy(obs_shape, act_dim, args.seq_len, device)

    optim_kwargs = dict(
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        max_ep_len=args.max_ep_len,
        rtg_scale=args.rtg_scale,
    )
    arch_kwargs = dict(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        p_drop=args.p_drop,
    )

    if s == "cumulative":
        return PandaCumulativeReplayStrategy(
            obs_shape, act_dim, args.seq_len, device,
            **arch_kwargs, **optim_kwargs,
            rehearsal_capacity=args.rehearsal_capacity,
        )
    if s == "ewc":
        return PandaEWCStrategy(
            obs_shape, act_dim, args.seq_len, device,
            n_heads=args.n_heads,
            **optim_kwargs,
            ewc_lambda=args.ewc_lambda,
            fisher_n_batches=args.fisher_n_batches,
            fisher_batch_size=args.fisher_batch_size,
        )
    if s == "si":
        return PandaSIStrategy(
            obs_shape, act_dim, args.seq_len, device,
            **arch_kwargs, **optim_kwargs,
            si_lambda=args.si_lambda,
            si_epsilon=args.si_epsilon,
            omega_max=args.si_omega_max,
            clamp_min0=args.si_clamp_min0,
            debug_every=args.si_debug_every,
        )

    if s not in TSN_STRATEGIES:
        raise ValueError(f"Unsupported strategy: {s}")

    tsn_kwargs = dict(
        obs_dim=obs_dim,
        act_dim=act_dim,
        seq_len=args.seq_len,
        device=device,
        **arch_kwargs,
        **optim_kwargs,
        keep_ratio=args.tsn_keep_ratio,
        include_embeddings=args.tsn_include_embeddings,
        quantize_after_task=args.tsn_quantize_after_task,
        quant_clusters=args.tsn_quant_clusters,
        freeze_non_mask_params_after_first=args.tsn_freeze_non_mask_params_after_first,
        store_task_obs_stats=args.tsn_store_task_obs_stats,
        patch_model_act=args.tsn_patch_model_act,
    )
    if s == "tsn":
        return TSNPandaStrategy(**tsn_kwargs, allow_weight_reuse=args.tsn_allow_weight_reuse)

    reuse_kwargs = dict(
        expected_num_tasks=int(n_tasks),
        keep_ratio_schedule="constant",
        min_keep_ratio=1e-3,
        reuse_memory_size=int(args.tsn_reuse_memory_size),
        reuse_kl_threshold=float(args.tsn_reuse_kl_threshold),
        max_model_copies=_max_model_copies(args),
    )
    if s == "tsn_origin_reuse":
        return TSNOriginalReusePandaStrategy(**tsn_kwargs, **reuse_kwargs)

    return TSNImprovedReusePandaStrategy(
        **tsn_kwargs,
        **reuse_kwargs,
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
        **(improved_reuse_extra or {}),
    )


def _train_task(strategy: Any, args, trajs: List[Trajectory], active_action_dim_mask: List[int]) -> None:
    kwargs: Dict[str, Any] = dict(steps=args.steps_per_task, batch_size=args.batch_size)
    if args.strategy == "cumulative":
        kwargs["mix"] = args.mix
    elif args.strategy in TSN_STRATEGIES:
        kwargs["active_action_dim_mask"] = active_action_dim_mask
    strategy.train_task(trajs, **kwargs)


def _select_eval_task(strategy: Any, task_idx: int) -> None:
    if not (hasattr(strategy, "has_task_mask") and hasattr(strategy, "set_eval_task")):
        return
    if strategy.has_task_mask(task_idx):
        strategy.set_eval_task(task_idx)
    elif hasattr(strategy, "clear_eval_task"):
        strategy.clear_eval_task()


def run_continual_learning(
    strategy: Any, args, tasks: List[PandaTask], obs_dim: int, act_dim: int, device: str
) -> np.ndarray:
    n = len(tasks)
    P = np.zeros((n, n), dtype=np.float32)
    torch_device = torch.device(device)

    for i, task in enumerate(tasks):
        print(f"\n[Task {i + 1}/{n}] Train on {task.name}")
        active_action_dim_mask = [1] * task.act_dim + [0] * max(0, act_dim - task.act_dim)
        task.record["active_action_dim_mask"] = list(active_action_dim_mask)

        _train_task(strategy, args, task.trajs, active_action_dim_mask)
        strategy.after_task(task.trajs)

        for j, eval_task in enumerate(tasks):
            _select_eval_task(strategy, j)
            score = evaluate_dt_panda_cl(
                model=strategy.model,
                env=eval_task.env,
                episodes=int(args.episodes_eval),
                device=torch_device,
                max_steps=int(args.max_steps),
                target_return=float(eval_task.target_return),
                seed=int(args.seed + 1000 * j),
                obs_keys=PANDA_OBS_KEYS,
                obs_pad_to=obs_dim,
                act_pad_to=act_dim,
                timestep_clip_max=int(eval_task.max_len - 1),
                gamma=1.0,
                clip_action=True,
            )
            P[i, j] = float(score)
            print(f"[eval] after task {i + 1} on {eval_task.name}: {score:.3f} (target={eval_task.target_return:.6f})")

    return P


# ------------------------------------------------------------
# Saving
# ------------------------------------------------------------
def _if(flag: bool, value: Any) -> Any:
    return value if flag else None


def _build_model_info(strategy: Any, args, model_sig: str, obs_dim: int, act_dim: int) -> Dict[str, Any]:
    model = strategy.model
    copy_stats = _safe_copy_state_stats(strategy)
    return {
        "signature": model_sig,
        "class": type(model).__name__,
        "obs_shape": [obs_dim],
        "obs_dim_global": int(obs_dim),
        "act_dim_global": int(act_dim),
        "seq_len": int(args.seq_len),
        "d_model_requested": int(args.d_model),
        "n_layers_requested": int(args.n_layers),
        "n_heads_requested": int(args.n_heads),
        "p_drop_requested": float(args.p_drop),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "grad_clip": float(args.grad_clip),
        "max_ep_len_requested": int(args.max_ep_len),
        "max_ep_len_effective": int(getattr(model, "max_ep_len", args.max_ep_len)),
        "rtg_scale_requested": float(args.rtg_scale),
        "rtg_scale_effective": float(getattr(model, "rtg_scale", args.rtg_scale)),
        "num_parameters_total": int(sum(p.numel() for p in model.parameters())),
        "num_parameters_trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "num_model_copies": copy_stats["num_model_copies"],
        "num_parameters_all_copies_total": copy_stats["num_parameters_all_copies_total"],
        "dt_cfg": _dt_cfg_dict_from_model(model),
    }


def _strategy_hparams(strategy: Any, args) -> Dict[str, Any]:
    s = args.strategy
    is_ewc = s == "ewc"
    is_si = s == "si"
    is_tsn_core = s == "tsn"
    is_tsn_like = s in TSN_STRATEGIES
    is_tsn_reuse = s in TSN_REUSE_STRATEGIES
    is_improved = s == "tsn_improved_reuse"
    copy_stats = _safe_copy_state_stats(strategy)

    return {
        "ewc_lambda": _if(is_ewc, args.ewc_lambda),
        "fisher_n_batches": _if(is_ewc, args.fisher_n_batches),
        "fisher_batch_size": _if(is_ewc, args.fisher_batch_size),
        "si_lambda": _if(is_si, args.si_lambda),
        "si_epsilon": _if(is_si, args.si_epsilon),
        "si_omega_max": _if(is_si, args.si_omega_max),
        "si_clamp_min0": _if(is_si, args.si_clamp_min0),
        "si_debug_every": _if(is_si, args.si_debug_every),
        "tsn_keep_ratio": _if(is_tsn_like, args.tsn_keep_ratio),
        "tsn_quant_clusters": _if(is_tsn_like, args.tsn_quant_clusters),
        "tsn_include_embeddings": _if(is_tsn_like, args.tsn_include_embeddings),
        "tsn_allow_weight_reuse": _if(is_tsn_core, args.tsn_allow_weight_reuse),
        "tsn_quantize_after_task": _if(is_tsn_like, args.tsn_quantize_after_task),
        "tsn_freeze_non_mask_params_after_first": _if(is_tsn_like, args.tsn_freeze_non_mask_params_after_first),
        "tsn_store_task_obs_stats": _if(is_tsn_like, args.tsn_store_task_obs_stats),
        "tsn_patch_model_act": _if(is_tsn_like, args.tsn_patch_model_act),
        "tsn_reuse_memory_size": _if(is_tsn_reuse, int(args.tsn_reuse_memory_size)),
        "tsn_reuse_kl_threshold": _if(is_tsn_reuse, float(args.tsn_reuse_kl_threshold)),
        "tsn_max_model_copies": _if(is_tsn_reuse, _max_model_copies(args)),
        "tsn_task_to_copy": _if(is_tsn_reuse, copy_stats["task_to_copy"]),
        "tsn_task_similarity": _if(is_tsn_reuse, copy_stats["task_similarity"]),
        "tsn_reuse_score_mode": _if(is_improved, str(args.tsn_reuse_score_mode)),
        "tsn_routing_n_batches": _if(is_improved, int(args.tsn_routing_n_batches)),
        "tsn_routing_batch_size": _if(is_improved, int(args.tsn_routing_batch_size)),
        "tsn_action_reuse_threshold": _if(is_improved, float(args.tsn_action_reuse_threshold)),
        "tsn_latent_reuse_threshold": _if(is_improved, float(args.tsn_latent_reuse_threshold)),
        "tsn_hybrid_reuse_threshold": _if(is_improved, float(args.tsn_hybrid_reuse_threshold)),
        "tsn_hybrid_alpha": _if(is_improved, float(args.tsn_hybrid_alpha)),
        "tsn_normalize_similarity_scores": _if(is_improved, bool(args.tsn_normalize_similarity_scores)),
        "tsn_warmstart_source_scores": _if(is_improved, bool(args.tsn_warmstart_source_scores)),
        "tsn_warmstart_strength": _if(is_improved, float(args.tsn_warmstart_strength)),
        "tsn_warmstart_noise_std": _if(is_improved, float(args.tsn_warmstart_noise_std)),
        "tsn_warmstart_on_new_copy": _if(is_improved, bool(args.tsn_warmstart_on_new_copy)),
        "tsn_reuse_signature": _if(is_tsn_reuse, _reuse_signature(args).lstrip("_")),
    }


def save_results(
    run_dir: str,
    strategy: Any,
    args,
    tasks: List[PandaTask],
    P: np.ndarray,
    model_sig: str,
    obs_dim: int,
    act_dim: int,
    *,
    include_target_sources: bool = False,
    hparams: Optional[Dict[str, Any]] = None,
) -> None:
    """
    `include_target_sources` adds `target_return_source_map` right after `target_return_map`;
    `hparams` replaces the default `_strategy_hparams` block of results.json.
    """
    task_names = [t.name for t in tasks]
    results = BenchmarkResults(name=f"DT-{args.strategy}:{SPEC_TAG}", task_names=task_names, perf_matrix=P)
    metrics = StandardCLMetrics.compute(results)
    is_tsn_reuse = args.strategy in TSN_REUSE_STRATEGIES

    save_payload: Dict[str, Any] = {
        "name": results.name,
        "task_names": task_names,
        "perf_matrix": P.tolist(),
        "metrics": metrics,
        "model": _build_model_info(strategy, args, model_sig, obs_dim, act_dim),
        "tasks": [t.record for t in tasks],
        "task_order": TASK_ORDER,
        "obs_dim_global": obs_dim,
        "act_dim_global": act_dim,
        "target_return_map": {t.name: t.target_return for t in tasks},
    }
    if include_target_sources:
        save_payload["target_return_source_map"] = {t.name: t.record["target_return_source"] for t in tasks}
    save_payload.update({
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
        **(hparams if hparams is not None else _strategy_hparams(strategy, args)),
    })
    if args.strategy == "tsn_origin_reuse":
        save_payload.update(_collect_origin_reuse_meta(strategy))

    save_json(os.path.join(run_dir, "results.json"), save_payload)

    save_json(os.path.join(run_dir, f"rez_{bench_short(BENCH)}.json"), {
        "metrics": metrics,
        "strategy": args.strategy,
        "model_signature": model_sig,
        "tsn_reuse_score_mode": _if(args.strategy == "tsn_improved_reuse", str(args.tsn_reuse_score_mode)),
        "tsn_reuse_signature": _if(is_tsn_reuse, _reuse_signature(args).lstrip("_")),
    })
    save_matrix_csv(os.path.join(run_dir, "matrix.csv"), task_names, P)

    steps = per_step_report(task_names, P)
    save_json(os.path.join(run_dir, "per_step.json"), {"per_step": steps})
    if steps:
        with open(os.path.join(run_dir, "per_step.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(steps[0].keys()))
            w.writeheader()
            w.writerows(steps)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def build_arg_parser(
    *,
    target_modes: Sequence[str],
    target_mode_default: str,
    target_mode_help: Optional[str] = None,
) -> argparse.ArgumentParser:
    """Arguments shared by the Panda CL runners; only `--target-mode` differs between them."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=str, default=str(RUNS))
    ap.add_argument("--strategy", choices=list(STRATEGIES), default="cumulative")
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--steps-per-task", type=int, default=1_000_000)
    ap.add_argument("--episodes-eval", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=128)

    # Replay
    ap.add_argument("--mix", type=float, default=0.5, help="Replay fraction in each minibatch.")
    ap.add_argument("--rehearsal-capacity", type=int, default=5000)

    # Model / optimizer
    ap.add_argument("--target-mode", choices=list(target_modes), default=target_mode_default, help=target_mode_help)
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

    # EWC
    ap.add_argument("--ewc-lambda", type=float, default=50.0)
    ap.add_argument("--fisher-n-batches", type=int, default=50)
    ap.add_argument("--fisher-batch-size", type=int, default=32)

    # SI
    ap.add_argument("--si-lambda", type=float, default=0.5)
    ap.add_argument("--si-epsilon", type=float, default=0.1)
    ap.add_argument("--si-omega-max", type=float, default=10.0)
    ap.add_argument("--no-si-clamp-min0", dest="si_clamp_min0", action="store_false", default=True)
    ap.add_argument("--si-debug-every", type=int, default=0)

    # TSN
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

    # TSN reuse (origin + improved)
    ap.add_argument("--tsn-reuse-kl-threshold", type=float, default=0.25)
    ap.add_argument("--tsn-reuse-memory-size", type=int, default=256)
    ap.add_argument(
        "--tsn-max-model-copies",
        type=int,
        default=0,
        help="0 means no explicit limit for reuse copies.",
    )

    # TSN improved reuse
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
    return ap
