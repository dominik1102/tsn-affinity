#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import os
from dataclasses import asdict, is_dataclass
from typing import Dict, Optional, List, Any

import numpy as np
import torch
import torch.nn.functional as F


def _install_sdpa_attn_mask_causal_compat() -> None:
    """Compatibility shim for PyTorch SDPA.

    Recent PyTorch versions raise:
        RuntimeError: Explicit attn_mask should not be set when is_causal=True

    Our DT implementation passes a key-padding mask together with causal
    attention.  This wrapper merges the causal constraint into the explicit
    mask and then calls scaled_dot_product_attention with is_causal=False.
    It preserves autoregressive masking and the original padding mask.
    """
    if getattr(F.scaled_dot_product_attention, "_clbench_sdpa_compat", False):
        return

    _orig_sdpa = F.scaled_dot_product_attention

    def _as_broadcastable_mask(mask: torch.Tensor, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        B = int(q.shape[0])
        T = int(q.shape[-2])
        S = int(k.shape[-2])
        m = mask.to(device=q.device)

        if m.dim() == 2:
            # [B,S] key-padding mask or [T,S] attention mask.
            if int(m.shape[0]) == B and int(m.shape[1]) == S:
                m = m[:, None, None, :]
            elif int(m.shape[0]) == T and int(m.shape[1]) == S:
                m = m[None, None, :, :]
            else:
                m = m[None, None, :, :]
        elif m.dim() == 3:
            # [B,T,S] or [B,1,S].
            if int(m.shape[0]) == B and int(m.shape[-1]) == S and int(m.shape[1]) == T:
                m = m[:, None, :, :]
            elif int(m.shape[0]) == B and int(m.shape[-1]) == S:
                m = m[:, None, None, :]
            else:
                m = m[:, None, :, :]
        elif m.dim() == 4:
            pass
        else:
            raise ValueError(f"Unsupported SDPA attn_mask shape: {tuple(mask.shape)}")
        return m

    def _merge_mask_with_causal(mask: torch.Tensor, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        T = int(q.shape[-2])
        S = int(k.shape[-2])
        causal_allowed = torch.ones((T, S), device=q.device, dtype=torch.bool).tril()
        causal_allowed = causal_allowed.view(1, 1, T, S)

        m = _as_broadcastable_mask(mask, q, k)

        if m.dtype == torch.bool:
            return m.bool() & causal_allowed

        # The DT runner commonly passes 0/1 attention masks.  Treat those as
        # allowed-key masks, not additive logits biases.
        try:
            finite = torch.isfinite(m).all().item()
            mn = float(m.min().detach().cpu().item()) if finite else -1.0
            mx = float(m.max().detach().cpu().item()) if finite else 2.0
            looks_binary = bool(finite and mn >= 0.0 and mx <= 1.0)
        except Exception:
            looks_binary = False

        if looks_binary:
            return (m > 0) & causal_allowed

        # Additive mask case: 0 for allowed, -inf/large negative for blocked.
        additive = m.to(dtype=q.dtype)
        causal_add = torch.zeros((T, S), device=q.device, dtype=q.dtype)
        causal_add.masked_fill_(~causal_allowed.view(T, S), float("-inf"))
        return additive + causal_add.view(1, 1, T, S)

    def _sdpa_compat(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
        if attn_mask is not None and bool(is_causal):
            attn_mask = _merge_mask_with_causal(attn_mask, query, key)
            is_causal = False

        kwargs = {
            "attn_mask": attn_mask,
            "dropout_p": dropout_p,
            "is_causal": is_causal,
        }
        if scale is not None:
            kwargs["scale"] = scale

        try:
            return _orig_sdpa(query, key, value, **kwargs, enable_gqa=enable_gqa)
        except TypeError:
            return _orig_sdpa(query, key, value, **kwargs)

    _sdpa_compat._clbench_sdpa_compat = True  # type: ignore[attr-defined]
    F.scaled_dot_product_attention = _sdpa_compat
    print("[patch] installed SDPA attn_mask+is_causal compatibility shim", flush=True)


# Do NOT install the SDPA shim by default.  The legacy PyTorch attention path
# was the one that learned Breakout reliably in the earlier Atari runs.
# Enable this only together with --install-sdpa-compat or by setting
# CLBENCH_INSTALL_SDPA_COMPAT=1 before starting Python.
if os.environ.get("CLBENCH_INSTALL_SDPA_COMPAT", "0") == "1":
    _install_sdpa_attn_mask_causal_compat()

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

def _short_sig_text(text: str, max_len: int = 90) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    safe = []
    for ch in text:
        if ch.isalnum():
            safe.append(ch)
        elif ch in (".", "-", "_"):
            safe.append(ch)
        else:
            safe.append("-")
    out = "".join(safe).strip("-")
    while "--" in out:
        out = out.replace("--", "-")
    return out[:max_len]


def _reuse_signature(args) -> str:
    """
    Extra suffix for run_dir so improved-reuse runs are easy to distinguish.
    """
    if getattr(args, "strategy", "") != "tsn_improved_reuse":
        return ""

    mode = str(getattr(args, "tsn_reuse_score_mode", "action"))
    suffix = ""
    manual_plan = str(getattr(args, "tsn_manual_route_plan", "") or "").strip()
    manual_thr = str(getattr(args, "tsn_manual_action_thresholds", "") or "").strip()
    if manual_plan:
        suffix += f"_manual-{_short_sig_text(manual_plan, 70)}"
    if manual_thr:
        suffix += f"_tthr-{_short_sig_text(manual_thr, 40)}"
    if bool(getattr(args, "tsn_probe_routing", False)):
        suffix += (
            f"_probeS{int(args.tsn_probe_steps)}"
            f"_top{int(args.tsn_probe_top_k)}"
            f"_grid{str(args.tsn_probe_residual_grid).replace(',', '-')}"
        )
    if str(getattr(args, "tsn_routing_policy", "")) == "dynamic_occ":
        suffix += (
            f"_dynocc"
            f"_ot{float(getattr(args, 'tsn_dynamic_occ_target', 0.80)):g}"
            f"_oo{float(getattr(args, 'tsn_dynamic_occ_open', 0.84)):g}"
            f"_m{float(getattr(args, 'tsn_dynamic_copy_margin', 6.0)):g}"
            f"_r{float(getattr(args, 'tsn_dynamic_rho_high', 0.50)):g}-"
            f"{float(getattr(args, 'tsn_dynamic_rho_mid', 0.25)):g}-"
            f"{float(getattr(args, 'tsn_dynamic_rho_low', 0.10)):g}"
        )

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
    Replay an action sequence and return sum of rewards.

    This is used only as a compatibility check between the exported NPZ actions
    and the current ALE/Gymnasium environment.  It intentionally does not use
    auto-fire, because the exported actions should already contain any actions
    actually used by the collector.
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


def _infer_action_map_small_discrete(
    env,
    actions: np.ndarray,
    dataset_return: float,
    *,
    seed: int = 0,
    max_steps: Optional[int] = None,
    tol: float = 1e-3,
    max_n: int = 6,
) -> Optional[np.ndarray]:
    """
    Try to infer a small discrete action remapping dataset_id -> env_id.

    This mirrors the single-task Breakout debug runner.  It is intentionally
    limited to small action spaces, because trying all permutations for Atari
    games with many actions would be too expensive.  This is mostly meant for
    Breakout-like minimal action spaces where a wrong RIGHT/LEFT/FIRE mapping
    can make offline CE look perfect while online rollouts fail.
    """
    try:
        import gymnasium as gym
        if not isinstance(env.action_space, gym.spaces.Discrete):
            return None
    except Exception:
        return None

    n = int(env.action_space.n)
    if n <= 1 or n > int(max_n):
        return None

    ep_actions = np.asarray(actions, dtype=np.int64).reshape(-1)
    if ep_actions.size == 0:
        return None
    if int(ep_actions.min()) < 0 or int(ep_actions.max()) >= n:
        return None

    base_ret = _replay_actions_return(env, ep_actions, seed=seed, max_steps=max_steps)
    base_diff = abs(float(base_ret) - float(dataset_return))
    if base_diff <= float(tol):
        return None

    import itertools
    best_map = None
    best_diff = base_diff
    for perm in itertools.permutations(range(n)):
        m = np.asarray(perm, dtype=np.int64)
        mapped = m[ep_actions]
        ret = _replay_actions_return(env, mapped, seed=seed, max_steps=max_steps)
        diff = abs(float(ret) - float(dataset_return))
        if diff < best_diff:
            best_diff = diff
            best_map = m
            if best_diff <= float(tol):
                break

    if best_map is not None and best_diff <= float(tol):
        return best_map
    return None


def _apply_action_map_to_trajs(trajs: list[Trajectory], action_map: np.ndarray) -> None:
    """In-place remap of Trajectory.actions using dataset_id -> env_id map."""
    m = np.asarray(action_map, dtype=np.int64)
    for tr in trajs:
        a = np.asarray(tr.actions, dtype=np.int64)
        if a.size == 0:
            continue
        if int(a.min()) < 0 or int(a.max()) >= int(len(m)):
            raise ValueError(
                f"Cannot apply action map of length {len(m)} to actions min={a.min()} max={a.max()}"
            )
        tr.actions = m[a].astype(np.int64, copy=False)


def _task_rtg_scale_from_returns(rets: np.ndarray) -> float:
    """Single-task compatible RTG scale: max absolute episodic return, at least 1."""
    if rets.size == 0:
        return 1.0
    return float(max(1.0, float(np.max(np.abs(rets.astype(np.float32))))))


def _set_strategy_rtg_scale(strategy: Any, scale: float, *, reason: str = "") -> None:
    """
    Set rtg_scale on the active model and on all stored copy models when present.

    Copy-based TSN strategies can activate different model copies for training
    or evaluation.  Setting only strategy.model can be lost when _activate_copy()
    switches to a stored copy, so we update all known copies as well.
    """
    scale = float(max(1.0, scale))
    try:
        if hasattr(strategy, "model") and hasattr(strategy.model, "rtg_scale"):
            strategy.model.rtg_scale = scale
    except Exception:
        pass

    for st in getattr(strategy, "copy_states", []) or []:
        try:
            if hasattr(st, "model") and hasattr(st.model, "rtg_scale"):
                st.model.rtg_scale = scale
        except Exception:
            pass

    if reason:
        print(f"[rtg-scale] {reason}: {scale:.3f}")


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
            obs=obs_ep.astype(np.float32, copy=False),
            actions=act_ep.astype(np.int64, copy=False),
            rewards=rew_ep.astype(np.float32, copy=False),
            timesteps=ts,
            returns_to_go=rtg.astype(np.float32, copy=False),
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
    p.add_argument(
        "--atari-action-remap",
        choices=["auto", "none"],
        default="auto",
        help=(
            "If replay-check detects a mismatch in a small discrete action space, "
            "try to infer and apply dataset-action -> env-action remapping. "
            "Use 'none' to disable. Mostly useful for Breakout-like games."
        ),
    )
    p.add_argument(
        "--force-breakout-auto-fire",
        dest="force_breakout_auto_fire",
        action="store_true",
        default=True,
        help="Force auto-fire and auto-fire-on-life-loss during Breakout evaluation.",
    )
    p.add_argument(
        "--no-force-breakout-auto-fire",
        dest="force_breakout_auto_fire",
        action="store_false",
        help="Disable the Breakout-specific auto-fire evaluation fallback.",
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
    p.add_argument(
        "--rtg-scale",
        type=float,
        default=1000.0,
        help="Used only with --rtg-scale-mode fixed; otherwise kept for compatibility.",
    )
    p.add_argument(
        "--rtg-scale-mode",
        choices=["per_task", "global", "fixed"],
        default="per_task",
        help=(
            "RTG scaling. per_task mirrors the single-task runner and is important for Breakout; "
            "global is the old continual-runner behavior; fixed uses --rtg-scale."
        ),
    )

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
    p.add_argument(
        "--tsn-no-quant",
        dest="tsn_no_quant",
        action="store_true",
        default=True,
        help="Disable TSN post-task quantization. Default for the paper experiments.",
    )
    p.add_argument(
        "--tsn-quant",
        dest="tsn_no_quant",
        action="store_false",
        help="Explicitly enable TSN post-task quantization.",
    )
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
    p.add_argument("--no-tsn-normalize-similarity-scores", dest="tsn_normalize_similarity_scores", action="store_false", default=True)

    p.add_argument(
        "--no-tsn-warmstart-source-scores",
        dest="tsn_warmstart_source_scores",
        action="store_false",
        default=True,
    )
    p.add_argument("--tsn-warmstart-strength", type=float, default=2.0)
    p.add_argument("--tsn-warmstart-noise-std", type=float, default=0.02)
    p.add_argument("--tsn-warmstart-on-new-copy", action="store_true")
    p.add_argument(
        "--tsn-routing-policy",
        choices=["affinity", "random", "round_robin", "always_reuse", "always_new", "dynamic_occ"],
        default="affinity",
    )
    p.add_argument("--tsn-copy-penalty", type=float, default=0.0)
    p.add_argument("--tsn-occupancy-penalty", type=float, default=0.0)
    p.add_argument("--tsn-reuse-margin", type=float, default=0.0)
    p.add_argument("--tsn-residual-reuse", action="store_true")
    p.add_argument("--tsn-residual-keep-ratio", type=float, default=0.10)

    # Dynamic occupancy-aware routing.  No probe training is used; the route is
    # selected from affinity scores plus current/projected copy occupancy.
    p.add_argument("--tsn-dynamic-occ-target", type=float, default=0.80)
    p.add_argument("--tsn-dynamic-occ-open", type=float, default=0.84)
    p.add_argument("--tsn-dynamic-occ-hard", type=float, default=0.90)
    p.add_argument("--tsn-dynamic-occ-lambda", type=float, default=30.0)
    p.add_argument("--tsn-dynamic-full-lambda", type=float, default=20.0)
    p.add_argument("--tsn-dynamic-copy-margin", type=float, default=6.0)
    p.add_argument("--tsn-dynamic-delay-new-until-task", type=int, default=2)
    p.add_argument("--tsn-dynamic-rho-high", type=float, default=0.50)
    p.add_argument("--tsn-dynamic-rho-mid", type=float, default=0.25)
    p.add_argument("--tsn-dynamic-rho-low", type=float, default=0.10)
    p.add_argument("--tsn-dynamic-low-occ", type=float, default=0.60)
    p.add_argument("--tsn-dynamic-mid-occ", type=float, default=0.80)

    # Probe-Adaptive Routing (PAR).  This performs a short offline inner-loop
    # route test before committing source/copy/residual decisions.
    p.add_argument("--tsn-probe-routing", action="store_true")
    p.add_argument("--tsn-probe-steps", type=int, default=0)
    p.add_argument("--tsn-probe-top-k", type=int, default=2)
    p.add_argument("--tsn-probe-batch-size", type=int, default=64)
    p.add_argument("--tsn-probe-val-batches", type=int, default=4)
    p.add_argument("--tsn-probe-residual-grid", type=str, default="0.10,0.25,0.50")
    p.add_argument("--tsn-probe-new-copy-penalty", type=float, default=0.02)
    p.add_argument("--tsn-probe-residual-penalty", type=float, default=0.0)
    p.add_argument("--tsn-probe-occupancy-penalty", type=float, default=0.0)
    p.add_argument("--tsn-probe-include-new-copy", dest="tsn_probe_include_new_copy", action="store_true", default=True)
    p.add_argument("--tsn-probe-no-new-copy", dest="tsn_probe_include_new_copy", action="store_false")
    p.add_argument("--tsn-probe-seed", type=int, default=123000)
    p.add_argument("--tsn-probe-deterministic", dest="tsn_probe_deterministic", action="store_true", default=True)
    p.add_argument("--tsn-probe-nondeterministic", dest="tsn_probe_deterministic", action="store_false")

    # Manual/oracle routing debugger for tsn_improved_reuse.
    # Examples:
    #   --tsn-manual-route-plan "1:reuse:0:rho=0.50,2:reuse:0:rho=0.50,3:new:1,4:reuse:3:rho=0.50"
    #   --tsn-manual-action-thresholds "1:25,4:10"
    p.add_argument("--tsn-manual-route-plan", type=str, default="")
    p.add_argument("--tsn-manual-action-thresholds", "--tsn-task-thresholds", dest="tsn_manual_action_thresholds", type=str, default="")

    # Anchor only the FIRST TRANSFER step: task_id=1 reuses task 0/copy 0.
    # This does not hardcode evaluation scores and does not touch task-0 training.
    p.add_argument("--tsn-anchor-first-reuse", action="store_true")
    p.add_argument("--tsn-anchor-first-reuse-rho", type=float, default=0.50)
    p.add_argument("--tsn-anchor-first-reuse-source-task", type=int, default=0)

    # Safety guard: abort immediately if the first task was not acquired.
    # This prevents wasting GPU hours on runs where Breakout already collapsed
    # before any routing decision can occur.
    p.add_argument("--abort-on-first-task-fail", action="store_true")
    p.add_argument("--min-first-task-score", type=float, default=50.0)

    # Attention/determinism switches.  Default is legacy PyTorch SDPA behavior,
    # because earlier stable Atari runs used that path.  Strict mode disables
    # flash/memory-efficient SDP and may require the compatibility shim.
    p.add_argument("--strict-determinism", action="store_true")
    p.add_argument("--install-sdpa-compat", action="store_true")

    args = p.parse_args()
    _set_seed(int(args.seed))

    if bool(args.install_sdpa_compat):
        _install_sdpa_attn_mask_causal_compat()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        if bool(args.strict_determinism):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            try:
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)
            except Exception:
                pass
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
        else:
            # Legacy path: keep PyTorch's default SDPA backend selection.
            # Earlier stable Atari runs that learned Breakout used this behavior.
            torch.backends.cudnn.benchmark = False

    print(f"[determinism] strict={bool(args.strict_determinism)} sdpa_compat={bool(args.install_sdpa_compat)} CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")
    if torch.cuda.is_available():
        try:
            print(
                "[determinism] "
                f"flash_sdp={torch.backends.cuda.flash_sdp_enabled()} "
                f"mem_efficient_sdp={torch.backends.cuda.mem_efficient_sdp_enabled()} "
                f"math_sdp={torch.backends.cuda.math_sdp_enabled()}"
            )
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
        rtg_scale=float(max(1.0, float(args.rtg_scale))),
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
            routing_policy=str(args.tsn_routing_policy),
            copy_penalty=float(args.tsn_copy_penalty),
            occupancy_penalty=float(args.tsn_occupancy_penalty),
            reuse_margin=float(args.tsn_reuse_margin),
            residual_reuse=bool(args.tsn_residual_reuse),
            residual_keep_ratio=float(args.tsn_residual_keep_ratio),
            dynamic_occ_target=float(args.tsn_dynamic_occ_target),
            dynamic_occ_open=float(args.tsn_dynamic_occ_open),
            dynamic_occ_hard=float(args.tsn_dynamic_occ_hard),
            dynamic_occ_lambda=float(args.tsn_dynamic_occ_lambda),
            dynamic_full_lambda=float(args.tsn_dynamic_full_lambda),
            dynamic_copy_margin=float(args.tsn_dynamic_copy_margin),
            dynamic_delay_new_until_task=int(args.tsn_dynamic_delay_new_until_task),
            dynamic_rho_high=float(args.tsn_dynamic_rho_high),
            dynamic_rho_mid=float(args.tsn_dynamic_rho_mid),
            dynamic_rho_low=float(args.tsn_dynamic_rho_low),
            dynamic_low_occ=float(args.tsn_dynamic_low_occ),
            dynamic_mid_occ=float(args.tsn_dynamic_mid_occ),
            probe_routing=bool(args.tsn_probe_routing),
            probe_steps=int(args.tsn_probe_steps),
            probe_top_k=int(args.tsn_probe_top_k),
            probe_batch_size=int(args.tsn_probe_batch_size),
            probe_val_batches=int(args.tsn_probe_val_batches),
            probe_residual_grid=str(args.tsn_probe_residual_grid),
            probe_new_copy_penalty=float(args.tsn_probe_new_copy_penalty),
            probe_residual_penalty=float(args.tsn_probe_residual_penalty),
            probe_occupancy_penalty=float(args.tsn_probe_occupancy_penalty),
            probe_include_new_copy=bool(args.tsn_probe_include_new_copy),
            probe_seed=int(args.tsn_probe_seed),
            probe_deterministic=bool(args.tsn_probe_deterministic),
            manual_route_plan=str(args.tsn_manual_route_plan),
            manual_action_thresholds=str(args.tsn_manual_action_thresholds),
            anchor_first_reuse=bool(args.tsn_anchor_first_reuse),
            anchor_first_reuse_rho=float(args.tsn_anchor_first_reuse_rho),
            anchor_first_reuse_source_task=int(args.tsn_anchor_first_reuse_source_task),
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
    rtg_scale_map: dict[str, float] = {}
    action_remap_map: dict[str, Optional[List[int]]] = {}
    all_returns: List[float] = []
    max_len_in_data = 0

    for name in task_names:
        trajs = load_offline_trajs_for_task(args.dataset_root, name)

        rets = traj_returns(trajs)
        if rets.size:
            all_returns.extend([float(x) for x in rets.tolist()])

        if trajs:
            max_len_in_data = max(max_len_in_data, int(max(len(t.actions) for t in trajs)))

        if args.target_return is not None:
            target_return_map[name] = float(args.target_return)
        else:
            target_return_map[name] = pick_target_return(rets, args.target_mode)

        rtg_scale_map[name] = _task_rtg_scale_from_returns(rets)
        action_remap_map[name] = None

        print(f"[offline] target_return[{name}] = {target_return_map[name]:.2f} (mode={args.target_mode})")
        print(f"[offline] rtg_scale_per_task[{name}] = {rtg_scale_map[name]:.3f}")

        # Check and optionally fix dataset-action ids vs current environment action ids.
        # The single-task Breakout runner did this; the continual runner previously only warned.
        if trajs and (bool(args.replay_check) or str(args.atari_action_remap) == "auto"):
            ds_ret = float(np.sum(trajs[0].rewards))
            ep0_actions = np.asarray(trajs[0].actions, dtype=np.int64)
            env_ret = _replay_actions_return(
                envs[name],
                ep0_actions,
                seed=int(seed_map.get(name, 0)),
                max_steps=int(args.max_steps),
            )
            diff = abs(ds_ret - env_ret)
            ok = diff <= 1e-3
            prefix = "[replay-check]" if args.replay_check else "[replay-check:auto]"
            print(f"{prefix} {name}: dataset_ep0={ds_ret:.3f} env_ep0={env_ret:.3f} diff={diff:.3f} ok={ok}")

            if (not ok) and str(args.atari_action_remap) == "auto":
                amap = _infer_action_map_small_discrete(
                    envs[name],
                    ep0_actions,
                    ds_ret,
                    seed=int(seed_map.get(name, 0)),
                    max_steps=int(args.max_steps),
                    tol=1e-3,
                    max_n=6,
                )
                if amap is not None:
                    _apply_action_map_to_trajs(trajs, amap)
                    env_ret2 = _replay_actions_return(
                        envs[name],
                        np.asarray(trajs[0].actions, dtype=np.int64),
                        seed=int(seed_map.get(name, 0)),
                        max_steps=int(args.max_steps),
                    )
                    action_remap_map[name] = [int(x) for x in amap.tolist()]
                    print(f"[action-remap] {name}: dataset->env map = {action_remap_map[name]}")
                    try:
                        print(f"[action-remap] {name}: env meanings = {envs[name].unwrapped.get_action_meanings()}")
                    except Exception:
                        pass
                    print(f"[action-remap] {name}: after remap ep0 replay {ds_ret:.3f} == {env_ret2:.3f}")
                else:
                    print(
                        f"[replay-check][WARN] {name}: Env != dataset and no small action remap was found. "
                        "This usually means env/preprocessing/frameskip/sticky/noop mismatch."
                    )
            elif (not ok) and args.replay_check:
                print(
                    "[replay-check][WARN] Env != dataset. "
                    "Make sure you use --atari-env minari_like and matching clip_rewards/frame_stack as in the export."
                )

        offline_trajs[name] = trajs

    # Decide the actual RTG scaling protocol.
    if str(args.rtg_scale_mode) == "global":
        global_rtg_scale = float(
            max(1.0, np.max(np.abs(np.asarray(all_returns, dtype=np.float32))) if all_returns else 1.0)
        )
        rtg_scale_map = {name: global_rtg_scale for name in task_names}
        print(f"[rtg-scale] mode=global scale={global_rtg_scale:.3f}")
    elif str(args.rtg_scale_mode) == "fixed":
        fixed_rtg_scale = float(max(1.0, float(args.rtg_scale)))
        rtg_scale_map = {name: fixed_rtg_scale for name in task_names}
        print(f"[rtg-scale] mode=fixed scale={fixed_rtg_scale:.3f}")
    else:
        print("[rtg-scale] mode=per_task")
        for name in task_names:
            print(f"[rtg-scale]   {name}: {rtg_scale_map[name]:.3f}")

    if hasattr(strategy, "model") and hasattr(strategy.model, "rtg_scale") and task_names:
        _set_strategy_rtg_scale(strategy, rtg_scale_map[task_names[0]], reason=f"initial/{task_names[0]}")

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

        # Match the single-task runner: train each task with its own RTG scale.
        # This is especially important for Breakout when Atlantis is in the same sequence.
        _set_strategy_rtg_scale(strategy, rtg_scale_map[name], reason=f"train/{name}")

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
            try:
                meanings = env2.unwrapped.get_action_meanings()
                if isinstance(meanings, (list, tuple)) and "FIRE" in meanings:
                    fire_id = int(meanings.index("FIRE"))
                    fire_used = any(
                        (np.asarray(t.actions).reshape(-1) == fire_id).any()
                        for t in offline_trajs[n2]
                    )
                    # Preserve explicit CLI flags. Only auto-enable when dataset never uses FIRE.
                    if not fire_used and not args.auto_fire and not args.auto_fire_on_life_loss:
                        auto_fire_eff = True
                        auto_fire_life_eff = True

                    # Breakout is special: force FIRE at reset/life-loss unless explicitly disabled in a future flag.
                    if "breakout" in str(n2).lower():
                        auto_fire_eff = True
                        auto_fire_life_eff = True
            except Exception:
                pass

            # Match the scale used by the task-specific single-task runner during evaluation.
            _set_strategy_rtg_scale(strategy, rtg_scale_map[n2], reason=f"eval/after_task_{i + 1}/{n2}")

            print(
                f"[eval-config] task={n2} rtg_scale={rtg_scale_map[n2]:.3f} "
                f"auto_fire={auto_fire_eff} auto_fire_on_life_loss={auto_fire_life_eff}"
            )

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

            if (
                bool(args.abort_on_first_task_fail)
                and i == 0
                and j == 0
                and float(P[i, j]) < float(args.min_first_task_score)
            ):
                raise RuntimeError(
                    f"First-task acquisition failed: task={n2} score={float(P[i, j]):.3f} "
                    f"< min_first_task_score={float(args.min_first_task_score):.3f}. "
                    "This happens before any reuse/dynamic routing decision; check attention/determinism path, "
                    "Breakout auto-fire, and runner version before launching the full sequence."
                )

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
            "rtg_scale_mode": str(args.rtg_scale_mode),
            "rtg_scale_map": {k: float(v) for k, v in rtg_scale_map.items()},
            "atari_action_remap": str(args.atari_action_remap),
            "action_remap_map": action_remap_map,
            "force_breakout_auto_fire": bool(args.force_breakout_auto_fire),

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
            "tsn_routing_policy": str(args.tsn_routing_policy) if is_tsn_improved_reuse else None,
            "tsn_copy_penalty": float(args.tsn_copy_penalty) if is_tsn_improved_reuse else None,
            "tsn_occupancy_penalty": float(args.tsn_occupancy_penalty) if is_tsn_improved_reuse else None,
            "tsn_reuse_margin": float(args.tsn_reuse_margin) if is_tsn_improved_reuse else None,
            "tsn_residual_reuse": bool(args.tsn_residual_reuse) if is_tsn_improved_reuse else None,
            "tsn_residual_keep_ratio": float(args.tsn_residual_keep_ratio) if is_tsn_improved_reuse else None,
            "tsn_dynamic_occ_target": float(args.tsn_dynamic_occ_target) if is_tsn_improved_reuse else None,
            "tsn_dynamic_occ_open": float(args.tsn_dynamic_occ_open) if is_tsn_improved_reuse else None,
            "tsn_dynamic_occ_hard": float(args.tsn_dynamic_occ_hard) if is_tsn_improved_reuse else None,
            "tsn_dynamic_occ_lambda": float(args.tsn_dynamic_occ_lambda) if is_tsn_improved_reuse else None,
            "tsn_dynamic_full_lambda": float(args.tsn_dynamic_full_lambda) if is_tsn_improved_reuse else None,
            "tsn_dynamic_copy_margin": float(args.tsn_dynamic_copy_margin) if is_tsn_improved_reuse else None,
            "tsn_dynamic_delay_new_until_task": int(args.tsn_dynamic_delay_new_until_task) if is_tsn_improved_reuse else None,
            "tsn_dynamic_rho_high": float(args.tsn_dynamic_rho_high) if is_tsn_improved_reuse else None,
            "tsn_dynamic_rho_mid": float(args.tsn_dynamic_rho_mid) if is_tsn_improved_reuse else None,
            "tsn_dynamic_rho_low": float(args.tsn_dynamic_rho_low) if is_tsn_improved_reuse else None,
            "tsn_dynamic_low_occ": float(args.tsn_dynamic_low_occ) if is_tsn_improved_reuse else None,
            "tsn_dynamic_mid_occ": float(args.tsn_dynamic_mid_occ) if is_tsn_improved_reuse else None,
            "tsn_probe_routing": bool(args.tsn_probe_routing) if is_tsn_improved_reuse else None,
            "tsn_probe_steps": int(args.tsn_probe_steps) if is_tsn_improved_reuse else None,
            "tsn_probe_top_k": int(args.tsn_probe_top_k) if is_tsn_improved_reuse else None,
            "tsn_probe_batch_size": int(args.tsn_probe_batch_size) if is_tsn_improved_reuse else None,
            "tsn_probe_val_batches": int(args.tsn_probe_val_batches) if is_tsn_improved_reuse else None,
            "tsn_probe_residual_grid": str(args.tsn_probe_residual_grid) if is_tsn_improved_reuse else None,
            "tsn_probe_new_copy_penalty": float(args.tsn_probe_new_copy_penalty) if is_tsn_improved_reuse else None,
            "tsn_probe_residual_penalty": float(args.tsn_probe_residual_penalty) if is_tsn_improved_reuse else None,
            "tsn_probe_occupancy_penalty": float(args.tsn_probe_occupancy_penalty) if is_tsn_improved_reuse else None,
            "tsn_probe_include_new_copy": bool(args.tsn_probe_include_new_copy) if is_tsn_improved_reuse else None,
            "tsn_probe_seed": int(args.tsn_probe_seed) if is_tsn_improved_reuse else None,
            "tsn_probe_deterministic": bool(args.tsn_probe_deterministic) if is_tsn_improved_reuse else None,
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