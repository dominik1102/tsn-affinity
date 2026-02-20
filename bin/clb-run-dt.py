#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import torch

from clbench.core.registry import TaskRegistry
from clbench.adapters.cartpole import CartPoleAdapter
from clbench.adapters.atari import AtariAdapter
from clbench.io.serialize import load_task_specs
from clbench.benchmark.runner import make_tasks, describe_tasks, BenchmarkResults
from clbench.benchmark.metrics import StandardCLMetrics
from clbench.benchmark.metrics_extra import per_step_report
from clbench.io.run_logger import (
    build_run_dir,
    save_json,
    save_matrix_csv,
    bench_short,
    save_task_gen_json,
)
from dt.dataset import Trajectory

from strategies.naive import NaiveStrategy
from strategies.cumulative import CumulativeReplayStrategy
from strategies.ewc import EWCStrategy

# ✅ minari-like env + stable eval
from dt.utils import make_minari_atari_env, evaluate_dt_forward

from dt.io_traj import save_trajs_pickle


TaskRegistry.register("cartpole", CartPoleAdapter())
TaskRegistry.register("atari", AtariAdapter())  # zostawiamy (dla opcji --atari-env clbench)


# ------------------------------------------------------------
# Small helpers
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


def _extract_seed(spec_obj: Any, fallback: int = 0) -> int:
    """
    TaskSpec w clbench bywa różny w zależności od wersji.
    Próbujemy kilka pól; jeśli brak -> fallback.
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
    # Żeby gymnasium widział ALE/... envy
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
    Prosty replay sekwencji akcji na env i zwrot sumy rewardów.
    Używamy do sanity-check: dataset_return ~ env_return.
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
    Jeśli model ma time-embedding (dt.te) o długości < desired,
    to go rozszerzamy, żeby timesteps > 9999 nie były clampowane.
    Działa dla Twojego dt.model.DecisionTransformer.
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

    # init jak w GPT/DT: N(0, 0.02)
    torch.nn.init.normal_(new_te.weight, mean=0.0, std=0.02)

    with torch.no_grad():
        new_te.weight[:old_n].copy_(te.weight)

    dt.te = new_te

    # update pola pomocnicze jeśli istnieją
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
        # stary tryb (AtariAdapter) - NIEZGODNY z expert_minari_dqn.npz
        envs = make_tasks("atari", specs)
        return envs

    # ✅ minari_like
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

        # seed zostawiamy None; w eval seedujemy per-episode
        envs[name] = make_minari_atari_env(
            env_id=env_id,
            seed=None,
            frame_stack=frame_stack,
            dqn_size=dqn_size,
            clip_rewards=clip_rewards,
        )

    return envs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument("--strategy", choices=["cumulative", "ewc", "naive"], default="cumulative")

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

    # Offline datasets root
    p.add_argument(
        "--dataset-root",
        type=str,
        default="",
        help="If non-empty, use offline expert trajectories from this root.",
    )

    # ✅ Eval controls
    p.add_argument("--max-steps", type=int, default=None, help="Max env steps per episode for evaluation.")
    p.add_argument("--target-mode", choices=["max", "p90", "mean"], default="max")
    p.add_argument("--target-return", type=float, default=None, help="If set, overrides per-task target_return.")

    # Optional Atari helpers
    p.add_argument("--auto-fire", action="store_true")
    p.add_argument("--auto-fire-on-life-loss", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    # ✅ PATCH: choose env pipeline for Atari
    p.add_argument(
        "--atari-env",
        choices=["minari_like", "clbench"],
        default="minari_like",
        help="Atari env pipeline. Use 'minari_like' to match expert_minari_dqn.npz.",
    )
    p.add_argument("--dqn-size", type=int, default=84)
    p.add_argument("--mix", type=float, default=0.5, help="Replay mix for cumulative strategy.")

    # ✅ PATCH: sanity-check dataset vs env
    p.add_argument("--replay-check", action="store_true", help="Replay ep0 actions in env and compare returns.")

    args = p.parse_args()

    specs = load_task_specs(args.spec)

    is_atari = any((s.params or {}).get("game", "").startswith("ALE/") for s in specs)
    bench = "atari" if is_atari else "cartpole"

    # default max_steps if not provided
    if args.max_steps is None:
        args.max_steps = 27000 if bench == "atari" else 1000

    # build envs (PATCH: for Atari use minari-like env)
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
    run_dir = build_run_dir(args.runs_root, bench, args.strategy, tag=args.tag or spec_tag)
    print(f"[run_dir] {run_dir}")

    env_list = list(envs.values())
    first_env = env_list[0]
    obs_shape = first_env.observation_space.shape

    # MODEL HEAD SIZE = MAX ACTIONS OVER ALL TASKS
    n_actions = max(e.action_space.n for e in env_list)


    if args.strategy == "naive":
        strategy = NaiveStrategy(obs_shape, n_actions, args.seq_len, args.device)
    elif args.strategy == "cumulative":
        strategy = CumulativeReplayStrategy(obs_shape, n_actions, args.seq_len, args.device)
    else:
        strategy = EWCStrategy(obs_shape, n_actions, args.seq_len, args.device)

    use_offline = bool(args.dataset_root)
    if not use_offline:
        raise NotImplementedError("This runner version expects --dataset-root (offline).")

    print(f"[mode] Using OFFLINE expert trajectories from: {args.dataset_root}")

    # Seed map (for deterministic eval episodes)
    seed_map: Dict[str, int] = {}
    for i, s in enumerate(specs):
        name = getattr(s, "name", None) or task_names[i]
        seed_map[name] = _extract_seed(s, fallback=0)

    # ✅ Pre-load all offline trajs + compute per-task target_return
    offline_trajs: dict[str, list[Trajectory]] = {}
    target_return_map: dict[str, float] = {}

    # Collect stats for global patches (rtg_scale, max_ep_len)
    all_returns: List[float] = []
    max_len_in_data = 0

    for name in task_names:
        trajs = load_offline_trajs_for_task(args.dataset_root, name)
        offline_trajs[name] = trajs

        # stats
        rets = traj_returns(trajs)
        if rets.size:
            all_returns.extend([float(x) for x in rets.tolist()])

        if trajs:
            max_len_in_data = max(max_len_in_data, int(max(len(t.actions) for t in trajs)))

        # target_return for DT conditioning
        if args.target_return is not None:
            target_return_map[name] = float(args.target_return)
        else:
            target_return_map[name] = pick_target_return(rets, args.target_mode)

        print(f"[offline] target_return[{name}] = {target_return_map[name]:.2f} (mode={args.target_mode})")

        # ✅ optional replay sanity-check
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
                    "Upewnij się, że używasz --atari-env minari_like i clip_rewards/frame_stack jak w eksporcie."
                )

    # ✅ PATCH: set global rtg_scale and max_ep_len (avoid timestep clamp)
    if hasattr(strategy, "model") and hasattr(strategy.model, "rtg_scale"):
        global_rtg_scale = float(max(1.0, np.max(np.abs(np.asarray(all_returns, dtype=np.float32))) if all_returns else 1.0))
        strategy.model.rtg_scale = global_rtg_scale
        print(f"[patch] set model.rtg_scale = {global_rtg_scale:.3f}")

    desired_max_ep_len = int(max(int(args.max_steps) + 1, int(max_len_in_data) + 1))
    _ensure_model_max_ep_len(strategy.model, desired_max_ep_len)
    # ✅ IMPORTANT: optimizer must be rebuilt after we replace dt.te
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

    # device for evaluation
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

        # Train strategy on current task data
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

        # ✅ Evaluate on all tasks (PATCH: stable eval_forward + minari-like auto-fire logic)
        for j, (n2, env2) in enumerate(envs.items()):
            # Auto-fire policy:
            # - if dataset NEVER uses FIRE -> enable auto-fire (typical Breakout collection)
            # - if dataset DOES use FIRE -> disable auto-fire to not shift distribution
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

            print(f"[eval] after task {i+1} on {n2}: {P[i, j]:.3f} (target={target_return_map[n2]:.1f})")

    results = BenchmarkResults(
        name=f"DT-{args.strategy}:{args.spec}",
        task_names=task_names,
        perf_matrix=P,
    )
    metrics = StandardCLMetrics.compute(results)

    save_json(
        os.path.join(run_dir, "results.json"),
        {
            "name": results.name,
            "task_names": task_names,
            "perf_matrix": P.tolist(),
            "metrics": metrics,
            "atari_env": args.atari_env,
            "max_steps": int(args.max_steps),
            "steps_per_task": int(args.steps_per_task),
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
        {"metrics": metrics, "strategy": args.strategy},
    )

    print("\n=== Continual DT results (offline=True) ===")
    print(P)
    print(f"\n[artifacts] saved to: {run_dir}")

    # close envs
    for e in envs.values():
        try:
            e.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
