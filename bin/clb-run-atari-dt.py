#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from dataclasses import asdict, is_dataclass
from typing import Dict, Optional, List, Any

import numpy as np
import torch

from bin.helper import _set_seed
from clbench.adapters.atari import AtariAdapter
from clbench.adapters.cartpole import CartPoleAdapter
from clbench.benchmark.metrics import StandardCLMetrics
from clbench.benchmark.metrics_extra import per_step_report
from clbench.benchmark.runner import make_tasks, describe_tasks, BenchmarkResults
from clbench.core.registry import TaskRegistry
from clbench.io.run_logger import (
    build_run_dir,
    save_json,
    save_matrix_csv,
    bench_short,
    save_task_gen_json,
)
from clbench.io.serialize import load_task_specs
from dt.dataset import Trajectory
from dt.io_traj import save_trajs_pickle
from dt.utils import make_minari_atari_env, evaluate_dt_forward

from strategies.cumulative import CumulativeReplayStrategy
from strategies.ewc import EWCStrategy
from strategies.naive import NaiveStrategy
from strategies.si import SIStrategy
from strategies.tsn_improved_reuse_atari import TSNImprovedReuseAtariStrategy
from strategies.tsn_strategy_atari_dt_v3 import TSNStrategy

# Change this import if your file has a different name.
from strategies.tsn_original_reuse_atari import TSNOriginalReuseStrategy

TaskRegistry.register("cartpole", CartPoleAdapter())
TaskRegistry.register("atari", AtariAdapter())


# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------
def _model_signature(args) -> str:
    return (
        f"dm{int(args.d_model)}"
        f"_L{int(args.n_layers)}"
        f"_H{int(args.n_heads)}"
        f"_K{int(args.seq_len)}"
        f"_drop{float(args.p_drop):.2f}"
    )

def _reuse_signature(args) -> str:
    """
    Extra suffix for run_dir so improved-reuse runs are easy to distinguish.
    """
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


def _extract_seed(spec_obj: Any, fallback: int = 0) -> int:
    """
    TaskSpec in clbench can vary between versions.
    Try several fields; if none found -> return fallback.
    """
    for k in ("seed", "random_seed", "rng_seed"):
        if hasattr(spec_obj, k):
            v = getattr(spec_obj, k)
            if v is not None:
                return int(v)
    params = getattr(spec_obj, "params", None) or {}
    if isinstance(params, dict) and ("seed" in params) and (params["seed"] is not None):
        return int(params["seed"])
    return int(fallback)


def _ensure_ale_registered() -> None:
    # So that gymnasium can see ALE/... envs
    try:
        import ale_py  # noqa: F401
        import gymnasium as gym
        gym.register_envs(ale_py)
    except Exception:
        pass


def _replay_actions_return(
    env,
    actions: np.ndarray,
    *,
    seed: int = 0,
    max_steps: Optional[int] = None,
) -> float:
    """
    Simple replay of an action sequence on the environment and return the sum of rewards.
    Used as a sanity-check: dataset_return ~ env_return.
    """
    obs, info = env.reset(seed=int(seed))
    total = 0.0
    terminated = truncated = False
    for t, a in enumerate(np.asarray(actions, dtype=np.int64).reshape(-1)):
        if max_steps is not None and t >= int(max_steps):
            break
        obs, r, terminated, truncated, info = env.step(int(a))
        total += float(r)
        if terminated or truncated:
            break
    return float(total)


def _ensure_model_max_ep_len(model: Any, desired: int) -> None:
    """
    If the model has a time-embedding (dt.te) with length < desired,
    expand it so that timesteps > 9999 are not clamped.
    Works for DecisionTransformer in dt.model.
    """
    desired = int(desired)
    if desired <= 0:
        return

    if not hasattr(model, "dt"):
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

    device = te.weight.device
    emb_dim = int(te.embedding_dim)

    new_te = torch.nn.Embedding(desired, emb_dim).to(device=device)

    # init like in GPT/DT: N(0, 0.02)
    torch.nn.init.normal_(new_te.weight, mean=0.0, std=0.02)

    with torch.no_grad():
        new_te.weight[:old_n].copy_(te.weight)

    dt.te = new_te

    # update helper fields if present
    if hasattr(model, "max_ep_len"):
        model.max_ep_len = desired
    if hasattr(dt, "cfg") and hasattr(dt.cfg, "max_ep_len"):
        dt.cfg.max_ep_len = desired

    print(f"[patch] expanded time-embedding: {old_n} -> {desired}")


# ------------------------------------------------------------
# Offline dataset loader
# ------------------------------------------------------------
def load_offline_trajs_for_task(dataset_root: str, task_name: str) -> list[Trajectory]:
    """
    Load expert/mixed trajectories for a single task from .npz and
    convert them to a list of Trajectory objects.

    Expects:
        dataset_root/task_name/*.npz
    Keys in npz:
        observations, actions, rewards, dones, episode_lengths
    """
    task_dir = os.path.join(dataset_root, task_name)
    if not os.path.isdir(task_dir):
        raise FileNotFoundError(f"[offline] task directory not found: {task_dir}")

    candidates = [f for f in os.listdir(task_dir) if f.endswith(".npz")]
    if not candidates:
        raise FileNotFoundError(f"[offline] no .npz files found in {task_dir}")
    candidates.sort()
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
    idx = 0
    rets = []

    for L in episode_lengths:
        L = int(L)
        obs_ep = observations[idx:idx + L]
        act_ep = actions[idx:idx + L]
        rew_ep = rewards[idx:idx + L]
        _done_ep = dones[idx:idx + L]
        idx += L

        r = rew_ep.astype(np.float32)
        rtg = np.flip(np.cumsum(np.flip(r, axis=0), axis=0), axis=0)
        ts = np.arange(len(act_ep), dtype=np.int64)

        traj = Trajectory(
            obs=obs_ep.astype(np.float32),
            actions=act_ep.astype(np.int64),
            rewards=rew_ep.astype(np.float32),
            timesteps=ts,
            returns_to_go=rtg.astype(np.float32),
        )
        trajs.append(traj)
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


# ------------------------------------------------------------
# Build envs (PATCH: minari_like Atari)
# ------------------------------------------------------------
def build_envs_from_specs(
    specs: list[Any],
    *,
    bench: str,
    atari_env_mode: str,
    dqn_size_default: int,
) -> Dict[str, Any]:
    if bench != "atari":
        envs = make_tasks(bench, specs)
        return envs

    if atari_env_mode == "clbench":
        # old mode (AtariAdapter) - INCOMPATIBLE with expert_minari_dqn.npz
        envs = make_tasks("atari", specs)
        return envs

    _ensure_ale_registered()

    envs: Dict[str, Any] = {}
    for i, s in enumerate(specs):
        name = getattr(s, "name", None) or f"task{i}"
        params = getattr(s, "params", None) or {}
        if not isinstance(params, dict):
            params = {}

        env_id = params.get("game", None)
        if not isinstance(env_id, str) or not env_id.startswith("ALE/"):
            raise ValueError(f"[atari:minari_like] spec {name} has no params.game='ALE/...', got: {env_id!r}")

        frame_stack = int(params.get("frame_stack", 4))
        clip_rewards = bool(params.get("clip_rewards", True))
        dqn_size = int(params.get("dqn_size", dqn_size_default))

        envs[name] = make_minari_atari_env(
            env_id=env_id,
            seed=None,
            frame_stack=frame_stack,
            dqn_size=dqn_size,
            clip_rewards=clip_rewards,
        )

    return envs


def _safe_dt_cfg_dict(model: Any) -> Optional[Dict[str, Any]]:
    dt_cfg = getattr(getattr(model, "dt", None), "cfg", None)
    if dt_cfg is None:
        return None
    if is_dataclass(dt_cfg):
        return asdict(dt_cfg)
    try:
        return dict(vars(dt_cfg))
    except Exception:
        return None


def _safe_copy_state_stats(strategy: Any) -> Dict[str, Any]:
    """
    Extract copy-level metadata for old-reuse strategies if present.
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument(
        "--strategy",
        choices=["cumulative", "ewc", "naive", "si", "tsn", "tsn_origin_reuse", "tsn_improved_reuse"],
        default="cumulative",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--episodes-eval", type=int, default=5)

    p.add_argument(
        "--steps-per-task", "--steps",
        dest="steps_per_task",
        type=int,
        default=2000,
        help="SGD steps per task (alias: --steps)",
    )

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--runs-root", type=str, default="runs")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--dump-trajs", action="store_true")

    p.add_argument(
        "--dataset-root",
        type=str,
        default="",
        help="If non-empty, use offline expert trajectories from this root.",
    )

    p.add_argument("--max-steps", type=int, default=None, help="Max env steps per episode for evaluation.")
    p.add_argument("--target-mode", choices=["max", "p90", "mean"], default="max")
    p.add_argument("--target-return", type=float, default=None, help="If set, overrides per-task target_return.")

    p.add_argument("--auto-fire", action="store_true")
    p.add_argument("--auto-fire-on-life-loss", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--atari-env",
        choices=["minari_like", "clbench"],
        default="minari_like",
        help="Atari env pipeline. Use 'minari_like' to match expert_minari_dqn.npz.",
    )
    p.add_argument("--dqn-size", type=int, default=84)
    p.add_argument("--mix", type=float, default=0.5, help="Replay mix for cumulative strategy.")

    p.add_argument(
        "--replay-check",
        action="store_true",
        help="Replay ep0 actions in env and compare dataset return vs env return.",
    )

    # Model / optimizer hyperparams
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--p-drop", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-ep-len", type=int, default=10000)
    p.add_argument("--rtg-scale", type=float, default=1000.0)

    # EWC / SI hyperparams
    p.add_argument("--ewc-lambda", type=float, default=50.0)
    p.add_argument("--fisher-n-batches", type=int, default=50)
    p.add_argument("--fisher-batch-size", type=int, default=8)
    p.add_argument("--si-lambda", type=float, default=1.0)
    p.add_argument("--si-epsilon", type=float, default=0.1)
    p.add_argument("--no-si-clamp-min0", dest="si_clamp_min0", action="store_false", default=True)

    # TSN flags
    p.add_argument("--tsn-keep-ratio", type=float, default=0.5)
    p.add_argument("--tsn-quant-clusters", type=int, default=16)
    p.add_argument("--tsn-no-quant", action="store_true")
    p.add_argument("--tsn-allow-weight-reuse", action="store_true")
    p.add_argument("--tsn-no-embeddings", action="store_true")
    p.add_argument("--tsn-no-freeze-shared", action="store_true")
    p.add_argument(
        "--tsn-keep-schedule",
        choices=["constant", "equal_remaining"],
        default="equal_remaining",
        help="How to allocate new-mask density across tasks.",
    )
    p.add_argument("--tsn-min-keep-ratio", type=float, default=1e-3)
    p.add_argument("--tsn-grad-clip", type=float, default=1.0)
    p.add_argument(
        "--tsn-skip-module",
        action="append",
        default=None,
        help="Fully-qualified module name to skip during TSN conversion. Repeatable. Default: dt.te",
    )

    # Old-reuse extras
    p.add_argument("--tsn-reuse-memory-size", type=int, default=256)
    p.add_argument("--tsn-reuse-kl-threshold", type=float, default=0.25)
    p.add_argument(
        "--tsn-max-model-copies",
        type=int,
        default=0,
        help="0 means no explicit limit for old-reuse copies.",
    )

    # Improved-reuse extras
    p.add_argument(
        "--tsn-reuse-score-mode",
        choices=["action", "latent", "hybrid"],
        default="action",
        help="Routing score used by tsn_improved_reuse.",
    )
    p.add_argument("--tsn-routing-n-batches", type=int, default=4)
    p.add_argument("--tsn-routing-batch-size", type=int, default=64)

    p.add_argument("--tsn-action-reuse-threshold", type=float, default=12.0)
    p.add_argument("--tsn-latent-reuse-threshold", type=float, default=50.0)
    p.add_argument("--tsn-hybrid-reuse-threshold", type=float, default=0.50)
    p.add_argument("--tsn-hybrid-alpha", type=float, default=0.70)

    p.add_argument(
        "--no-tsn-warmstart-source-scores",
        dest="tsn_warmstart_source_scores",
        action="store_false",
        default=True,
    )
    p.add_argument("--tsn-warmstart-strength", type=float, default=2.0)
    p.add_argument("--tsn-warmstart-noise-std", type=float, default=0.02)
    p.add_argument("--tsn-warmstart-on-new-copy", action="store_true")

    args = p.parse_args()
    _set_seed(int(args.seed))

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

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

    task_names = list(envs.keys())
    n = len(task_names)
    P = np.zeros((n, n), dtype=np.float32)

    spec_tag = os.path.splitext(os.path.basename(args.spec))[0]
    model_sig = _model_signature(args)
    run_tag = f"{(args.tag or spec_tag)}__{model_sig}{_reuse_signature(args)}"
    run_dir = build_run_dir(args.runs_root, bench, args.strategy, tag=run_tag)
    print(f"[run_dir] {run_dir}")

    env_list = list(envs.values())
    first_env = env_list[0]
    obs_shape = first_env.observation_space.shape

    common_model_kwargs = dict(
        d_model=int(args.d_model),
        n_layers=int(args.n_layers),
        n_heads=int(args.n_heads),
        p_drop=float(args.p_drop),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        max_ep_len=int(args.max_ep_len),
        rtg_scale=float(args.rtg_scale),
    )

    common_train_kwargs = dict(
        grad_clip=float(args.grad_clip),
    )

    n_actions = max(e.action_space.n for e in env_list)

    if args.strategy == "naive":
        strategy = NaiveStrategy(
            obs_shape, n_actions, args.seq_len, args.device,
            **common_model_kwargs,
            **common_train_kwargs,
        )
    elif args.strategy == "cumulative":
        strategy = CumulativeReplayStrategy(
            obs_shape, n_actions, args.seq_len, args.device,
            **common_model_kwargs,
            **common_train_kwargs,
        )
    elif args.strategy == "ewc":
        strategy = EWCStrategy(
            obs_shape, n_actions, args.seq_len, args.device,
            **common_model_kwargs,
            **common_train_kwargs,
            ewc_lambda=float(args.ewc_lambda),
            fisher_n_batches=int(args.fisher_n_batches),
            fisher_batch_size=int(args.fisher_batch_size),
        )
    elif args.strategy == "si":
        strategy = SIStrategy(
            obs_shape, n_actions, args.seq_len, args.device,
            **common_model_kwargs,
            **common_train_kwargs,
            si_lambda=float(args.si_lambda),
            si_epsilon=float(args.si_epsilon),
            clamp_omega=bool(args.si_clamp_min0),
        )
    elif args.strategy == "tsn":
        skip_modules = tuple(args.tsn_skip_module) if args.tsn_skip_module else ("dt.te",)
        strategy = TSNStrategy(
            obs_shape, n_actions, args.seq_len, args.device,
            **common_model_kwargs,
            grad_clip=float(args.grad_clip),
            keep_ratio=float(args.tsn_keep_ratio),
            include_embeddings=not bool(args.tsn_no_embeddings),
            quantize_after_task=not bool(args.tsn_no_quant),
            quant_clusters=int(args.tsn_quant_clusters),
            allow_weight_reuse=bool(args.tsn_allow_weight_reuse),
            freeze_non_mask_params_after_first=not bool(args.tsn_no_freeze_shared),
            skip_module_names=skip_modules,
            expected_num_tasks=int(n),
            keep_ratio_schedule=str(args.tsn_keep_schedule),
            min_keep_ratio=float(args.tsn_min_keep_ratio),
        )
    elif args.strategy == "tsn_origin_reuse":
        skip_modules = tuple(args.tsn_skip_module) if args.tsn_skip_module else ("dt.te",)
        max_model_copies = None if int(args.tsn_max_model_copies) <= 0 else int(args.tsn_max_model_copies)
        strategy = TSNOriginalReuseStrategy(
            obs_shape, n_actions, args.seq_len, args.device,
            **common_model_kwargs,
            grad_clip=float(args.tsn_grad_clip),
            keep_ratio=float(args.tsn_keep_ratio),
            include_embeddings=not bool(args.tsn_no_embeddings),
            quantize_after_task=not bool(args.tsn_no_quant),
            quant_clusters=int(args.tsn_quant_clusters),
            freeze_non_mask_params_after_first=not bool(args.tsn_no_freeze_shared),
            skip_module_names=skip_modules,
            expected_num_tasks=int(n),
            keep_ratio_schedule=str(args.tsn_keep_schedule),
            min_keep_ratio=float(args.tsn_min_keep_ratio),
            reuse_memory_size=int(args.tsn_reuse_memory_size),
            reuse_kl_threshold=float(args.tsn_reuse_kl_threshold),
            max_model_copies=max_model_copies,
        )
    elif args.strategy == "tsn_improved_reuse":
        skip_modules = tuple(args.tsn_skip_module) if args.tsn_skip_module else ("dt.te",)
        max_model_copies = None if int(args.tsn_max_model_copies) <= 0 else int(args.tsn_max_model_copies)

        strategy = TSNImprovedReuseAtariStrategy(
            obs_shape, n_actions, args.seq_len, args.device,
            **common_model_kwargs,
            grad_clip=float(args.tsn_grad_clip),
            keep_ratio=float(args.tsn_keep_ratio),
            include_embeddings=not bool(args.tsn_no_embeddings),
            quantize_after_task=not bool(args.tsn_no_quant),
            quant_clusters=int(args.tsn_quant_clusters),
            freeze_non_mask_params_after_first=not bool(args.tsn_no_freeze_shared),
            skip_module_names=skip_modules,
            expected_num_tasks=int(n),
            keep_ratio_schedule=str(args.tsn_keep_schedule),
            min_keep_ratio=float(args.tsn_min_keep_ratio),
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
        raise ValueError(f"Unknown strategy: {args.strategy}")

    use_offline = bool(args.dataset_root)
    if not use_offline:
        raise NotImplementedError("This runner version expects --dataset-root (offline).")

    print(f"[mode] Using OFFLINE expert trajectories from: {args.dataset_root}")

    seed_map: Dict[str, int] = {}
    for i, s in enumerate(specs):
        name = getattr(s, "name", None) or task_names[i]
        seed_map[name] = _extract_seed(s, fallback=0)

    offline_trajs: dict[str, list[Trajectory]] = {}
    target_return_map: dict[str, float] = {}
    all_returns: List[float] = []
    max_len_in_data = 0

    for name in task_names:
        trajs = load_offline_trajs_for_task(args.dataset_root, name)
        offline_trajs[name] = trajs

        rets = traj_returns(trajs)
        if rets.size:
            all_returns.extend([float(x) for x in rets.tolist()])

        if trajs:
            max_len_in_data = max(max_len_in_data, int(max(len(t.actions) for t in trajs)))

        if args.target_return is not None:
            target_return_map[name] = float(args.target_return)
        else:
            target_return_map[name] = pick_target_return(rets, args.target_mode)

        print(f"[offline] target_return[{name}] = {target_return_map[name]:.2f} (mode={args.target_mode})")

        if args.replay_check and trajs:
            ds_ret = float(np.sum(trajs[0].rewards))
            env_ret = _replay_actions_return(
                envs[name],
                np.asarray(trajs[0].actions, dtype=np.int64),
                seed=int(seed_map.get(name, 0)),
                max_steps=int(args.max_steps),
            )
            diff = abs(ds_ret - env_ret)
            ok = diff <= 1e-3
            print(f"[replay-check] {name}: dataset_ep0={ds_ret:.3f} env_ep0={env_ret:.3f} diff={diff:.3f} ok={ok}")
            if not ok:
                print(
                    "[replay-check][WARN] Env != dataset. "
                    "Make sure you use --atari-env minari_like and matching clip_rewards/frame_stack as in the export."
                )

    if hasattr(strategy, "model") and hasattr(strategy.model, "rtg_scale"):
        global_rtg_scale = float(
            max(1.0, np.max(np.abs(np.asarray(all_returns, dtype=np.float32))) if all_returns else 1.0)
        )
        strategy.model.rtg_scale = global_rtg_scale
        print(f"[patch] set model.rtg_scale = {global_rtg_scale:.3f}")

    desired_max_ep_len = int(max(int(args.max_steps) + 1, int(max_len_in_data) + 1))
    _ensure_model_max_ep_len(strategy.model, desired_max_ep_len)

    if hasattr(strategy, "opt") and isinstance(strategy.opt, torch.optim.Optimizer):
        pg0 = strategy.opt.param_groups[0]
        lr = float(pg0.get("lr", 3e-4))
        wd = float(pg0.get("weight_decay", 0.0))
        betas = pg0.get("betas", (0.9, 0.999))
        eps = pg0.get("eps", 1e-8)

        strategy.opt = torch.optim.AdamW(
            strategy.model.parameters(),
            lr=lr,
            weight_decay=wd,
            betas=betas,
            eps=eps,
        )
        print("[patch] rebuilt optimizer after time-embedding expansion")

    eval_device = torch.device(args.device)

    for i, (name, env) in enumerate(envs.items()):
        print(f"\n[Task {i + 1}/{n}] {name}")

        onpol = offline_trajs[name]

        traj_path = None
        if args.dump_trajs:
            os.makedirs(os.path.join(run_dir, "gen"), exist_ok=True)
            traj_path = os.path.join(run_dir, "gen", f"trajs_task{i}.pkl")
            save_trajs_pickle(traj_path, onpol)

        save_task_gen_json(
            run_dir,
            i + 1,
            {
                "step": i + 1,
                "task_name": name,
                "n_trajectories": len(onpol),
                "traj_file": traj_path,
                "offline": True,
                "dataset_root": args.dataset_root,
            },
        )

        bs = int(args.batch_size)

        if args.strategy == "cumulative":
            strategy.train_task(
                onpol,
                steps=int(args.steps_per_task),
                batch_size=bs,
                mix=float(args.mix),
            )
        else:
            strategy.train_task(
                onpol,
                steps=int(args.steps_per_task),
                batch_size=bs,
            )

        strategy.after_task(onpol)
        strategy.model.eval()

        for j, (n2, env2) in enumerate(envs.items()):
            if hasattr(strategy, "clear_eval_task"):
                strategy.clear_eval_task()
            if hasattr(strategy, "has_task_mask") and hasattr(strategy, "set_eval_task"):
                if strategy.has_task_mask(j):
                    strategy.set_eval_task(j)

            strategy.model.eval()

            auto_fire_eff = bool(args.auto_fire)
            auto_fire_life_eff = bool(args.auto_fire_on_life_loss)
            ...

            try:
                meanings = env2.unwrapped.get_action_meanings()
                if isinstance(meanings, (list, tuple)) and "FIRE" in meanings:
                    fire_id = int(meanings.index("FIRE"))
                    fire_used = any(
                        (np.asarray(t.actions).reshape(-1) == fire_id).any()
                        for t in offline_trajs[n2]
                    )
                    if fire_used:
                        auto_fire_eff = False
                        auto_fire_life_eff = False
                    else:
                        auto_fire_eff = True
                        auto_fire_life_eff = True
            except Exception:
                pass

            P[i, j] = evaluate_dt_forward(
                model=strategy.model,
                env=env2,
                episodes=int(args.episodes_eval),
                device=eval_device,
                max_steps=int(args.max_steps),
                target_return=float(target_return_map[n2]),
                seed=int(seed_map.get(n2, 0)),
                greedy=True,
                clamp_to_env_actions=True,
                auto_fire=auto_fire_eff,
                auto_fire_on_life_loss=auto_fire_life_eff,
                debug_action_hist=False,
            )

            print(f"[eval] after task {i + 1} on {n2}: {P[i, j]:.3f} (target={target_return_map[n2]:.1f})")

    results = BenchmarkResults(
        name=f"DT-{args.strategy}:{args.spec}",
        task_names=task_names,
        perf_matrix=P,
    )
    metrics = StandardCLMetrics.compute(results)

    num_params_total = int(sum(p.numel() for p in strategy.model.parameters()))
    num_params_trainable = int(sum(p.numel() for p in strategy.model.parameters() if p.requires_grad))
    dt_cfg_dict = _safe_dt_cfg_dict(strategy.model)
    copy_stats = _safe_copy_state_stats(strategy)

    model_info = {
        "class": type(strategy.model).__name__,
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
        "max_ep_len_effective": int(getattr(strategy.model, "max_ep_len", args.max_ep_len)),
        "rtg_scale_requested": float(args.rtg_scale),
        "rtg_scale_effective": float(getattr(strategy.model, "rtg_scale", args.rtg_scale)),
        "num_parameters_total": num_params_total,
        "num_parameters_trainable": num_params_trainable,
        "dt_cfg": dt_cfg_dict,
        "num_model_copies": copy_stats["num_model_copies"],
        "num_parameters_all_copies_total": copy_stats["num_parameters_all_copies_total"],
    }

    is_tsn_core = args.strategy == "tsn"
    is_tsn_origin_reuse = args.strategy == "tsn_origin_reuse"
    is_tsn_improved_reuse = args.strategy == "tsn_improved_reuse"
    is_tsn_like = args.strategy in ("tsn", "tsn_origin_reuse", "tsn_improved_reuse")
    is_tsn_reuse = args.strategy in ("tsn_origin_reuse", "tsn_improved_reuse")

    effective_grad_clip = float(args.tsn_grad_clip) if is_tsn_like else float(args.grad_clip)


    save_json(
        os.path.join(run_dir, "results.json"),
        {
            "name": results.name,
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

            # Common TSN-like params
                        "seed": int(args.seed),

            "tsn_keep_ratio": float(args.tsn_keep_ratio) if is_tsn_like else None,
            "tsn_keep_schedule": str(args.tsn_keep_schedule) if is_tsn_like else None,
            "tsn_min_keep_ratio": float(args.tsn_min_keep_ratio) if is_tsn_like else None,
            "tsn_grad_clip": float(args.tsn_grad_clip) if is_tsn_like else None,
            "tsn_quant_clusters": int(args.tsn_quant_clusters) if is_tsn_like else None,
            "tsn_quantize_after_task": (not bool(args.tsn_no_quant)) if is_tsn_like else None,
            "tsn_include_embeddings": (not bool(args.tsn_no_embeddings)) if is_tsn_like else None,
            "tsn_freeze_non_mask_params_after_first": (
                not bool(args.tsn_no_freeze_shared)
            ) if is_tsn_like else None,
            "tsn_skip_module_names": (
                list(args.tsn_skip_module) if args.tsn_skip_module else ["dt.te"]
            ) if is_tsn_like else None,

            "tsn_allow_weight_reuse": bool(args.tsn_allow_weight_reuse) if is_tsn_core else None,

            "tsn_reuse_memory_size": int(args.tsn_reuse_memory_size) if is_tsn_reuse else None,
            "tsn_reuse_kl_threshold": float(args.tsn_reuse_kl_threshold) if is_tsn_origin_reuse else None,
            "tsn_max_model_copies": (
                None if (not is_tsn_reuse or int(args.tsn_max_model_copies) <= 0)
                else int(args.tsn_max_model_copies)
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
            "grad_clip": effective_grad_clip,
        },
    )
    save_matrix_csv(os.path.join(run_dir, "matrix.csv"), task_names, P)

    steps = per_step_report(task_names, P)
    save_json(os.path.join(run_dir, "per_step.json"), {"per_step": steps})
    if steps:
        import csv
        with open(os.path.join(run_dir, "per_step.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(steps[0].keys()))
            w.writeheader()
            w.writerows(steps)

    save_json(
        os.path.join(run_dir, f"rez_{bench_short(bench)}.json"),
        {
            "metrics": metrics,
            "strategy": args.strategy,
            "model_signature": model_sig,
            "tsn_reuse_score_mode": str(args.tsn_reuse_score_mode) if args.strategy == "tsn_improved_reuse" else None,
            "tsn_reuse_signature": _reuse_signature(args).lstrip(
                "_") if args.strategy == "tsn_improved_reuse" else None,
        },
    )

    print("\n=== Continual DT results (offline=True) ===")
    print(P)
    print(f"\n[artifacts] saved to: {run_dir}")

    for e in envs.values():
        try:
            e.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
