"""Shared helpers for the continual Atari DT runners
(bin/clb-run-atari-dt.py and bin/clb_run_atari_capacity_patch.py)."""
from __future__ import annotations

import csv
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from clbench.benchmark.metrics_extra import per_step_report
from clbench.benchmark.runner import describe_tasks, make_tasks
from clbench.io.run_logger import (
    bench_short,
    build_run_dir,
    save_json,
    save_matrix_csv,
    save_task_gen_json,
)
from clbench.io.serialize import load_task_specs
from dt.dataset import Trajectory
from dt.io_traj import save_trajs_pickle
from dt.utils import evaluate_dt_forward, make_minari_atari_env
from strategies.cumulative import CumulativeReplayStrategy
from strategies.ewc import EWCStrategy
from strategies.naive import NaiveStrategy
from strategies.si import SIStrategy
from strategies.tsn_improved_reuse_atari import TSNImprovedReuseAtariStrategy
from strategies.tsn_original_reuse_atari import TSNOriginalReuseStrategy
from strategies.tsn_strategy_atari_dt_v3 import TSNStrategy

STRATEGIES = ("cumulative", "ewc", "naive", "si", "tsn", "tsn_origin_reuse", "tsn_improved_reuse")
TSN_REUSE_STRATEGIES = ("tsn_origin_reuse", "tsn_improved_reuse")
TSN_STRATEGIES = ("tsn",) + TSN_REUSE_STRATEGIES
DEFAULT_TSN_SKIP_MODULES = ("dt.te",)


def _if(flag: bool, value: Any) -> Any:
    return value if flag else None


# ------------------------------------------------------------
# Signatures
# ------------------------------------------------------------
def model_signature(args) -> str:
    return (
        f"dm{int(args.d_model)}"
        f"_L{int(args.n_layers)}"
        f"_H{int(args.n_heads)}"
        f"_K{int(args.seq_len)}"
        f"_drop{float(args.p_drop):.2f}"
    )


def improved_reuse_signature(args, suffix: str = "") -> str:
    """Extra suffix for run_dir so improved-reuse runs are easy to distinguish."""
    if getattr(args, "strategy", "") != "tsn_improved_reuse":
        return ""

    mode = str(getattr(args, "tsn_reuse_score_mode", "action"))
    if mode == "action":
        return f"__rsm-{mode}_athr{float(args.tsn_action_reuse_threshold):g}{suffix}"
    if mode == "latent":
        return f"__rsm-{mode}_lthr{float(args.tsn_latent_reuse_threshold):g}{suffix}"
    return (
        f"__rsm-{mode}"
        f"_hthr{float(args.tsn_hybrid_reuse_threshold):g}"
        f"_ha{float(args.tsn_hybrid_alpha):.2f}"
        f"{suffix}"
    )


# ------------------------------------------------------------
# Returns / targets / seeds
# ------------------------------------------------------------
def traj_returns(trajs: list[Trajectory]) -> np.ndarray:
    if not trajs:
        return np.array([], dtype=np.float32)
    return np.array([float(np.sum(t.rewards)) for t in trajs], dtype=np.float32)


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


def task_target_return(args, rets: np.ndarray) -> float:
    if args.target_return is not None:
        return float(args.target_return)
    return pick_target_return(rets, args.target_mode)


def global_rtg_scale(all_returns: List[float]) -> float:
    return float(
        max(1.0, np.max(np.abs(np.asarray(all_returns, dtype=np.float32)))) if all_returns else 1.0
    )


def extract_seed(spec_obj: Any, fallback: int = 0) -> int:
    """TaskSpec fields vary between clbench versions; try several, else return fallback."""
    for k in ("seed", "random_seed", "rng_seed"):
        if hasattr(spec_obj, k):
            v = getattr(spec_obj, k)
            if v is not None:
                return int(v)
    params = getattr(spec_obj, "params", None) or {}
    if isinstance(params, dict) and ("seed" in params) and (params["seed"] is not None):
        return int(params["seed"])
    return int(fallback)


def build_seed_map(specs: list[Any], task_names: List[str]) -> Dict[str, int]:
    seed_map: Dict[str, int] = {}
    for i, s in enumerate(specs):
        name = getattr(s, "name", None) or task_names[i]
        seed_map[name] = extract_seed(s, fallback=0)
    return seed_map


def dataset_uses_action(trajs: list[Trajectory], action_id: int) -> bool:
    return any((np.asarray(t.actions).reshape(-1) == action_id).any() for t in trajs)


# ------------------------------------------------------------
# Environments
# ------------------------------------------------------------
def ensure_ale_registered() -> None:
    """Make ALE/... envs visible to gymnasium."""
    try:
        import ale_py  # noqa: F401
        import gymnasium as gym
        gym.register_envs(ale_py)
    except Exception:
        pass


def replay_actions_return(
    env,
    actions: np.ndarray,
    *,
    seed: int = 0,
    max_steps: Optional[int] = None,
) -> float:
    """Replay an action sequence (no auto-fire) and return the sum of rewards."""
    env.reset(seed=int(seed))
    total = 0.0
    for t, a in enumerate(np.asarray(actions, dtype=np.int64).reshape(-1)):
        if max_steps is not None and t >= int(max_steps):
            break
        _obs, r, terminated, truncated, _info = env.step(int(a))
        total += float(r)
        if terminated or truncated:
            break
    return float(total)


def build_envs_from_specs(
    specs: list[Any],
    *,
    bench: str,
    atari_env_mode: str,
    dqn_size_default: int,
) -> Dict[str, Any]:
    if bench != "atari":
        return make_tasks(bench, specs)

    if atari_env_mode == "clbench":
        # Legacy AtariAdapter pipeline; NOT compatible with expert_minari_dqn.npz.
        return make_tasks("atari", specs)

    ensure_ale_registered()

    envs: Dict[str, Any] = {}
    for i, s in enumerate(specs):
        name = getattr(s, "name", None) or f"task{i}"
        params = getattr(s, "params", None) or {}
        if not isinstance(params, dict):
            params = {}

        env_id = params.get("game", None)
        if not isinstance(env_id, str) or not env_id.startswith("ALE/"):
            raise ValueError(f"[atari:minari_like] spec {name} has no params.game='ALE/...', got: {env_id!r}")

        envs[name] = make_minari_atari_env(
            env_id=env_id,
            seed=None,
            frame_stack=int(params.get("frame_stack", 4)),
            dqn_size=int(params.get("dqn_size", dqn_size_default)),
            clip_rewards=bool(params.get("clip_rewards", True)),
        )

    return envs


def close_envs(envs: Dict[str, Any]) -> None:
    for e in envs.values():
        try:
            e.close()
        except Exception:
            pass


def prepare_run(args, reuse_signature: str) -> Tuple[list[Any], str, Dict[str, Any], str, str]:
    """Load specs, build envs and the run dir. Returns (specs, bench, envs, run_dir, model_sig)."""
    specs = load_task_specs(args.spec)
    is_atari = any((s.params or {}).get("game", "").startswith("ALE/") for s in specs)
    bench = "atari" if is_atari else "cartpole"

    if args.max_steps is None:
        args.max_steps = 27000 if bench == "atari" else 1000

    envs = build_envs_from_specs(
        specs,
        bench=bench,
        atari_env_mode=str(args.atari_env),
        dqn_size_default=int(args.dqn_size),
    )
    print(describe_tasks(envs, bench))

    spec_tag = os.path.splitext(os.path.basename(args.spec))[0]
    model_sig = model_signature(args)
    run_tag = f"{(args.tag or spec_tag)}__{model_sig}{reuse_signature}"
    run_dir = build_run_dir(args.runs_root, bench, args.strategy, tag=run_tag)
    print(f"[run_dir] {run_dir}")
    return specs, bench, envs, run_dir, model_sig


# ------------------------------------------------------------
# Offline dataset loader
# ------------------------------------------------------------
def load_offline_trajs_for_task(dataset_root: str, task_name: str, *, copy: bool = True) -> list[Trajectory]:
    """
    Load trajectories for one task from the last (sorted) dataset_root/task_name/*.npz.
    Expected npz keys: observations, actions, rewards, dones, episode_lengths.
    `copy=False` lets per-episode arrays be views into the loaded npz arrays.
    """
    task_dir = os.path.join(dataset_root, task_name)
    if not os.path.isdir(task_dir):
        raise FileNotFoundError(f"[offline] task directory not found: {task_dir}")

    candidates = sorted(f for f in os.listdir(task_dir) if f.endswith(".npz"))
    if not candidates:
        raise FileNotFoundError(f"[offline] no .npz files found in {task_dir}")
    npz_path = os.path.join(task_dir, candidates[-1])

    print(f"[offline] loading trajectories for task '{task_name}' from {npz_path}")
    data = np.load(npz_path)

    observations = data["observations"]
    actions = data["actions"].reshape(-1)
    rewards = data["rewards"].reshape(-1)
    dones = data["dones"].reshape(-1)
    episode_lengths = data["episode_lengths"].astype(np.int64)

    total = int(episode_lengths.sum())
    if not (observations.shape[0] == actions.shape[0] == rewards.shape[0] == dones.shape[0] == total):
        raise ValueError(
            f"[offline] inconsistent shapes vs episode_lengths: "
            f"obs={observations.shape[0]} act={actions.shape[0]} rew={rewards.shape[0]} done={dones.shape[0]} total={total}"
        )

    trajs: list[Trajectory] = []
    rets: List[float] = []
    idx = 0
    for L in episode_lengths:
        L = int(L)
        obs_ep = observations[idx:idx + L]
        act_ep = actions[idx:idx + L]
        rew_ep = rewards[idx:idx + L]
        idx += L

        r = rew_ep.astype(np.float32)
        rtg = np.flip(np.cumsum(np.flip(r, axis=0), axis=0), axis=0)
        trajs.append(Trajectory(
            obs=obs_ep.astype(np.float32, copy=copy),
            actions=act_ep.astype(np.int64, copy=copy),
            rewards=rew_ep.astype(np.float32, copy=copy),
            timesteps=np.arange(len(act_ep), dtype=np.int64),
            returns_to_go=rtg.astype(np.float32, copy=copy),
        ))
        rets.append(float(r.sum()))

    if trajs:
        rets_np = np.array(rets, dtype=np.float32)
        print(
            f"[offline] task={task_name}, episodes={len(trajs)}, "
            f"avg_return={rets_np.mean():.1f}, min={rets_np.min():.1f}, max={rets_np.max():.1f}"
        )
    else:
        print(f"[offline] task={task_name}, WARNING: no episodes reconstructed")

    return trajs


def print_offline_mode(args) -> None:
    if not bool(args.dataset_root):
        raise NotImplementedError("This runner version expects --dataset-root (offline).")
    print(f"[mode] Using OFFLINE expert trajectories from: {args.dataset_root}")


# ------------------------------------------------------------
# Strategy
# ------------------------------------------------------------
def max_model_copies(args) -> Optional[int]:
    """`--tsn-max-model-copies <= 0` means no limit."""
    n = int(args.tsn_max_model_copies)
    return None if n <= 0 else n


def tsn_skip_modules(args) -> tuple:
    return tuple(args.tsn_skip_module) if args.tsn_skip_module else DEFAULT_TSN_SKIP_MODULES


def build_strategy(
    args,
    obs_shape: Any,
    n_actions: int,
    n_tasks: int,
    *,
    rtg_scale: float,
    improved_extra_kwargs: Optional[Dict[str, Any]] = None,
):
    s = args.strategy
    head = (obs_shape, n_actions, args.seq_len, args.device)
    model_kwargs = dict(
        d_model=int(args.d_model),
        n_layers=int(args.n_layers),
        n_heads=int(args.n_heads),
        p_drop=float(args.p_drop),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        max_ep_len=int(args.max_ep_len),
        rtg_scale=rtg_scale,
    )

    if s == "naive":
        return NaiveStrategy(*head, **model_kwargs, grad_clip=float(args.grad_clip))
    if s == "cumulative":
        return CumulativeReplayStrategy(*head, **model_kwargs, grad_clip=float(args.grad_clip))
    if s == "ewc":
        return EWCStrategy(
            *head, **model_kwargs,
            grad_clip=float(args.grad_clip),
            ewc_lambda=float(args.ewc_lambda),
            fisher_n_batches=int(args.fisher_n_batches),
            fisher_batch_size=int(args.fisher_batch_size),
        )
    if s == "si":
        return SIStrategy(
            *head, **model_kwargs,
            grad_clip=float(args.grad_clip),
            si_lambda=float(args.si_lambda),
            si_epsilon=float(args.si_epsilon),
            clamp_omega=bool(args.si_clamp_min0),
        )
    if s not in TSN_STRATEGIES:
        raise ValueError(f"Unknown strategy: {s}")

    tsn_kwargs = dict(
        keep_ratio=float(args.tsn_keep_ratio),
        include_embeddings=not bool(args.tsn_no_embeddings),
        quantize_after_task=not bool(args.tsn_no_quant),
        quant_clusters=int(args.tsn_quant_clusters),
        freeze_non_mask_params_after_first=not bool(args.tsn_no_freeze_shared),
        skip_module_names=tsn_skip_modules(args),
        expected_num_tasks=int(n_tasks),
        keep_ratio_schedule=str(args.tsn_keep_schedule),
        min_keep_ratio=float(args.tsn_min_keep_ratio),
    )
    if s == "tsn":
        return TSNStrategy(
            *head, **model_kwargs, **tsn_kwargs,
            grad_clip=float(args.grad_clip),
            allow_weight_reuse=bool(args.tsn_allow_weight_reuse),
        )

    reuse_kwargs = dict(
        grad_clip=float(args.tsn_grad_clip),
        reuse_memory_size=int(args.tsn_reuse_memory_size),
        reuse_kl_threshold=float(args.tsn_reuse_kl_threshold),
        max_model_copies=max_model_copies(args),
    )
    if s == "tsn_origin_reuse":
        return TSNOriginalReuseStrategy(*head, **model_kwargs, **tsn_kwargs, **reuse_kwargs)

    return TSNImprovedReuseAtariStrategy(
        *head, **model_kwargs, **tsn_kwargs, **reuse_kwargs,
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
        **(improved_extra_kwargs or {}),
    )


def ensure_model_max_ep_len(model: Any, desired: int) -> None:
    """
    Expand the DT time embedding (model.dt.te) to `desired` entries if it is shorter,
    so long-episode timesteps are not clamped.
    """
    desired = int(desired)
    if desired <= 0:
        return

    dt = getattr(model, "dt", None)
    if dt is None or not hasattr(dt, "te"):
        return

    te = dt.te
    if not isinstance(te, torch.nn.Embedding):
        return

    old_n = int(te.num_embeddings)
    if old_n >= desired:
        return

    new_te = torch.nn.Embedding(desired, int(te.embedding_dim)).to(device=te.weight.device)
    # GPT/DT-style init: N(0, 0.02)
    torch.nn.init.normal_(new_te.weight, mean=0.0, std=0.02)
    with torch.no_grad():
        new_te.weight[:old_n].copy_(te.weight)
    dt.te = new_te

    if hasattr(model, "max_ep_len"):
        model.max_ep_len = desired
    if hasattr(dt, "cfg") and hasattr(dt.cfg, "max_ep_len"):
        dt.cfg.max_ep_len = desired

    print(f"[patch] expanded time-embedding: {old_n} -> {desired}")


def rebuild_optimizer(strategy: Any) -> None:
    """Recreate AdamW (same hparams) so it tracks the parameters after time-embedding expansion."""
    if not (hasattr(strategy, "opt") and isinstance(strategy.opt, torch.optim.Optimizer)):
        return
    pg0 = strategy.opt.param_groups[0]
    strategy.opt = torch.optim.AdamW(
        strategy.model.parameters(),
        lr=float(pg0.get("lr", 3e-4)),
        weight_decay=float(pg0.get("weight_decay", 0.0)),
        betas=pg0.get("betas", (0.9, 0.999)),
        eps=pg0.get("eps", 1e-8),
    )
    print("[patch] rebuilt optimizer after time-embedding expansion")


# ------------------------------------------------------------
# Train / eval steps
# ------------------------------------------------------------
def save_task_gen(run_dir: str, task_idx: int, name: str, trajs: list[Trajectory], args) -> None:
    traj_path = None
    if args.dump_trajs:
        os.makedirs(os.path.join(run_dir, "gen"), exist_ok=True)
        traj_path = os.path.join(run_dir, "gen", f"trajs_task{task_idx}.pkl")
        save_trajs_pickle(traj_path, trajs)

    save_task_gen_json(
        run_dir,
        task_idx + 1,
        {
            "step": task_idx + 1,
            "task_name": name,
            "n_trajectories": len(trajs),
            "traj_file": traj_path,
            "offline": True,
            "dataset_root": args.dataset_root,
        },
    )


def train_on_task(strategy: Any, args, trajs: list[Trajectory]) -> None:
    kwargs: Dict[str, Any] = dict(steps=int(args.steps_per_task), batch_size=int(args.batch_size))
    if args.strategy == "cumulative":
        kwargs["mix"] = float(args.mix)
    strategy.train_task(trajs, **kwargs)
    strategy.after_task(trajs)
    strategy.model.eval()


def select_eval_task(strategy: Any, task_idx: int) -> None:
    if hasattr(strategy, "clear_eval_task"):
        strategy.clear_eval_task()
    if hasattr(strategy, "has_task_mask") and hasattr(strategy, "set_eval_task"):
        if strategy.has_task_mask(task_idx):
            strategy.set_eval_task(task_idx)
    strategy.model.eval()


def evaluate_on_task(
    strategy: Any,
    env: Any,
    args,
    *,
    device: torch.device,
    target_return: float,
    seed: int,
    auto_fire: bool,
    auto_fire_on_life_loss: bool,
) -> float:
    return evaluate_dt_forward(
        model=strategy.model,
        env=env,
        episodes=int(args.episodes_eval),
        device=device,
        max_steps=int(args.max_steps),
        target_return=float(target_return),
        seed=int(seed),
        greedy=True,
        clamp_to_env_actions=True,
        auto_fire=auto_fire,
        auto_fire_on_life_loss=auto_fire_on_life_loss,
        debug_action_hist=False,
    )


# ------------------------------------------------------------
# Saving
# ------------------------------------------------------------
def safe_dt_cfg_dict(model: Any) -> Optional[Dict[str, Any]]:
    dt_cfg = getattr(getattr(model, "dt", None), "cfg", None)
    if dt_cfg is None:
        return None
    if is_dataclass(dt_cfg):
        return asdict(dt_cfg)
    try:
        return dict(vars(dt_cfg))
    except Exception:
        return None


def safe_copy_state_stats(strategy: Any) -> Dict[str, Any]:
    """Copy-level metadata for reuse strategies, if present."""
    if not hasattr(strategy, "copy_states"):
        return {
            "num_model_copies": None,
            "num_parameters_all_copies_total": None,
            "task_to_copy": None,
            "task_similarity": None,
        }

    copy_states = getattr(strategy, "copy_states", [])
    total_params_all_copies: Optional[int] = 0
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


def build_model_info(
    strategy: Any, args, obs_shape: Any, n_actions: int, copy_stats: Dict[str, Any]
) -> Dict[str, Any]:
    model = strategy.model
    return {
        "class": type(model).__name__,
        "obs_shape": list(obs_shape),
        "n_actions": int(n_actions),
        "seq_len": int(args.seq_len),
        "d_model": int(args.d_model),
        "n_layers": int(args.n_layers),
        "n_heads": int(args.n_heads),
        "p_drop": float(args.p_drop),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "grad_clip": float(args.grad_clip),
        "max_ep_len_requested": int(args.max_ep_len),
        "max_ep_len_effective": int(getattr(model, "max_ep_len", args.max_ep_len)),
        "rtg_scale_requested": float(args.rtg_scale),
        "rtg_scale_effective": float(getattr(model, "rtg_scale", args.rtg_scale)),
        "num_parameters_total": int(sum(p.numel() for p in model.parameters())),
        "num_parameters_trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "dt_cfg": safe_dt_cfg_dict(model),
        "num_model_copies": copy_stats["num_model_copies"],
        "num_parameters_all_copies_total": copy_stats["num_parameters_all_copies_total"],
    }


def base_results_payload(
    results_name: str,
    task_names: List[str],
    P: np.ndarray,
    model_info: Dict[str, Any],
    metrics: Any,
    args,
) -> Dict[str, Any]:
    return {
        "name": results_name,
        "task_names": task_names,
        "perf_matrix": P.tolist(),
        "model": model_info,
        "metrics": metrics,
        "atari_env": args.atari_env,
        "max_steps": int(args.max_steps),
        "steps_per_task": int(args.steps_per_task),
        "episodes_eval": int(args.episodes_eval),
        "batch_size": int(args.batch_size),
        "target_mode": str(args.target_mode),
        "target_return_override": None if args.target_return is None else float(args.target_return),
        "mix": float(args.mix),
        "replay_check": bool(args.replay_check),
        "auto_fire": bool(args.auto_fire),
        "auto_fire_on_life_loss": bool(args.auto_fire_on_life_loss),
        "dqn_size": int(args.dqn_size),
    }


def tsn_results_hparams(args, copy_stats: Dict[str, Any]) -> Dict[str, Any]:
    """TSN hyperparameters shared by both runners (None when not applicable)."""
    s = args.strategy
    is_tsn_like = s in TSN_STRATEGIES
    is_tsn_reuse = s in TSN_REUSE_STRATEGIES
    is_improved = s == "tsn_improved_reuse"
    return {
        "tsn_keep_ratio": _if(is_tsn_like, float(args.tsn_keep_ratio)),
        "tsn_keep_schedule": _if(is_tsn_like, str(args.tsn_keep_schedule)),
        "tsn_min_keep_ratio": _if(is_tsn_like, float(args.tsn_min_keep_ratio)),
        "tsn_grad_clip": _if(is_tsn_like, float(args.tsn_grad_clip)),
        "tsn_quant_clusters": _if(is_tsn_like, int(args.tsn_quant_clusters)),
        "tsn_quantize_after_task": _if(is_tsn_like, not bool(args.tsn_no_quant)),
        "tsn_include_embeddings": _if(is_tsn_like, not bool(args.tsn_no_embeddings)),
        "tsn_freeze_non_mask_params_after_first": _if(is_tsn_like, not bool(args.tsn_no_freeze_shared)),
        "tsn_skip_module_names": _if(is_tsn_like, list(tsn_skip_modules(args))),
        "tsn_allow_weight_reuse": _if(s == "tsn", bool(args.tsn_allow_weight_reuse)),
        "tsn_reuse_memory_size": _if(is_tsn_reuse, int(args.tsn_reuse_memory_size)),
        "tsn_reuse_kl_threshold": _if(s == "tsn_origin_reuse", float(args.tsn_reuse_kl_threshold)),
        "tsn_max_model_copies": _if(is_tsn_reuse, max_model_copies(args)),
        "tsn_task_to_copy": _if(is_tsn_reuse, copy_stats["task_to_copy"]),
        "tsn_task_similarity": _if(is_tsn_reuse, copy_stats["task_similarity"]),
        "tsn_reuse_score_mode": _if(is_improved, str(args.tsn_reuse_score_mode)),
        "tsn_routing_n_batches": _if(is_improved, int(args.tsn_routing_n_batches)),
        "tsn_routing_batch_size": _if(is_improved, int(args.tsn_routing_batch_size)),
        "tsn_action_reuse_threshold": _if(is_improved, float(args.tsn_action_reuse_threshold)),
        "tsn_latent_reuse_threshold": _if(is_improved, float(args.tsn_latent_reuse_threshold)),
        "tsn_hybrid_reuse_threshold": _if(is_improved, float(args.tsn_hybrid_reuse_threshold)),
        "tsn_hybrid_alpha": _if(is_improved, float(args.tsn_hybrid_alpha)),
        # Evaluated lazily: some runners do not define this CLI flag.
        "tsn_normalize_similarity_scores": (
            bool(args.tsn_normalize_similarity_scores) if is_improved else None
        ),
        "tsn_warmstart_source_scores": _if(is_improved, bool(args.tsn_warmstart_source_scores)),
        "tsn_warmstart_strength": _if(is_improved, float(args.tsn_warmstart_strength)),
        "tsn_warmstart_noise_std": _if(is_improved, float(args.tsn_warmstart_noise_std)),
        "tsn_warmstart_on_new_copy": _if(is_improved, bool(args.tsn_warmstart_on_new_copy)),
    }


def tsn_results_tail(args, reuse_signature: str) -> Dict[str, Any]:
    is_tsn_like = args.strategy in TSN_STRATEGIES
    return {
        "tsn_reuse_signature": _if(args.strategy in TSN_REUSE_STRATEGIES, reuse_signature.lstrip("_")),
        "grad_clip": float(args.tsn_grad_clip) if is_tsn_like else float(args.grad_clip),
    }


def save_run_outputs(
    run_dir: str,
    bench: str,
    task_names: List[str],
    P: np.ndarray,
    results_payload: Dict[str, Any],
    metrics: Any,
    args,
    model_sig: str,
    reuse_signature: str,
) -> None:
    save_json(os.path.join(run_dir, "results.json"), results_payload)
    save_matrix_csv(os.path.join(run_dir, "matrix.csv"), task_names, P)

    steps = per_step_report(task_names, P)
    save_json(os.path.join(run_dir, "per_step.json"), {"per_step": steps})
    if steps:
        with open(os.path.join(run_dir, "per_step.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(steps[0].keys()))
            w.writeheader()
            w.writerows(steps)

    is_improved = args.strategy == "tsn_improved_reuse"
    save_json(
        os.path.join(run_dir, f"rez_{bench_short(bench)}.json"),
        {
            "metrics": metrics,
            "strategy": args.strategy,
            "model_signature": model_sig,
            "tsn_reuse_score_mode": _if(is_improved, str(args.tsn_reuse_score_mode)),
            "tsn_reuse_signature": _if(is_improved, reuse_signature.lstrip("_")),
        },
    )


def print_final(P: np.ndarray, run_dir: str) -> None:
    print("\n=== Continual DT results (offline=True) ===")
    print(P)
    print(f"\n[artifacts] saved to: {run_dir}")
