#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from clbench.adapters.atari import AtariAdapter
from clbench.core.registry import TaskRegistry
from clbench.benchmark.runner import make_tasks, describe_tasks, BenchmarkResults
from clbench.io.serialize import load_task_specs
from clbench.io.run_logger import build_run_dir, save_json, save_matrix_csv, bench_short
from clbench.benchmark.metrics_extra import per_step_report
from clbench.benchmark.metrics import StandardCLMetrics

# --- IMPORTANT: correct import path for the expert code ---
try:
    # preferred (matches clbench/experts/atari_dqn.py)
    from clbench.experts.atari_dqn import (
        train_atari_dqn_expert,
        collect_atari_expert_trajectories,
        save_trajectories_npz,
    )
except ImportError:
    # fallback if you really have "expertes" as a local package
    from expertes.atari_dqn import (  # type: ignore
        train_atari_dqn_expert,
        collect_atari_expert_trajectories,
        save_trajectories_npz,
    )

# Register Atari in CLBench registry
TaskRegistry.register("atari", AtariAdapter())


def main():

    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True, help="Atari spec file, e.g. specs_atari.json")

    p.add_argument("--episodes-per-task", type=int, default=100,
                   help="Number of expert episodes to record per task")
    p.add_argument("--max-len", type=int, default=1000,
                   help="Maximum episode length when collecting trajectories")

    p.add_argument("--total-steps-expert", type=int, default=500_000,
                   help="Number of env steps for DQN expert training per task")

    p.add_argument("--warmup-steps", type=int, default=10_000,
                   help="Replay warmup steps before updates start")
    p.add_argument("--eval-every", type=int, default=50_000,
                   help="Greedy eval frequency during training (0 disables)")

    p.add_argument("--fire-reset", action="store_true",
                   help="Press FIRE after reset for games that require it (e.g., Breakout).")
    p.add_argument("--no-fire-reset", dest="fire_reset", action="store_false",
                   help="Disable FIRE-after-reset helper.")
    p.set_defaults(fire_reset=True)

    p.add_argument("--debug", action="store_true", help="Print extra debug logs")

    p.add_argument("--out-dir", type=str, default="data/atari_expert",
                   help="Root directory for saving expert datasets (.npz + meta.json)")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device for expert training (cpu / cuda)")

    p.add_argument("--expert-action-prob", type=float, default=1.0,
                   help=("Per-step probability of using expert action when collecting trajectories "
                         "(1.0 = pure expert, 0.0 = pure random)."))

    p.add_argument("--runs-root", type=str, default="runs",
                   help="Root directory for CL-style logging of expert performance")
    p.add_argument("--tag", type=str, default="",
                   help="Optional run tag used in run_dir construction")

    args = p.parse_args()

    device = torch.device(args.device)
    print(f"[device] using {device}")

    if device.type == "cuda":
        # Speeds up fixed-shape convs in many cases
        torch.backends.cudnn.benchmark = True

    # Load tasks from spec
    specs = load_task_specs(args.spec)
    bench = "atari"
    spec_by_name = {s.name: s for s in specs}

    envs = make_tasks(bench, specs)
    print(describe_tasks(envs, bench))

    # CL-style run_dir for expert runs
    spec_tag = os.path.splitext(os.path.basename(args.spec))[0]
    run_dir = build_run_dir(
        args.runs_root,
        bench,
        strategy="expert",
        tag=args.tag or spec_tag,
    )
    print(f"[run_dir] {run_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    task_avg_returns: dict[str, float] = {}

    for name, env in envs.items():
        print("\n======================================")
        env_id = getattr(getattr(env, "spec", None), "id", None)
        print(f"[Atari Expert] task={name}, env_id={env_id}")
        print("======================================")
        task_dir = os.path.join(args.out_dir, name)
        os.makedirs(task_dir, exist_ok=True)
        # Separate env for eval (so eval doesn't perturb training env)
        eval_env = AtariAdapter().create_env(spec_by_name[name])

        try:
            expert = train_atari_dqn_expert(
                env,
                device=device,
                total_steps=int(args.total_steps_expert),
                warmup_steps=int(args.warmup_steps),
                eval_env=eval_env,
                eval_every=int(args.eval_every),
                fire_reset=bool(args.fire_reset),
                debug=bool(args.debug),
                # NOTE: do NOT set reward_clip=True if your adapter already clips rewards,
                # unless you intentionally want double clipping.
                reward_clip=False,
                save_best_path=os.path.join(task_dir, "expert_best.pt"),
                save_last_path=os.path.join(task_dir, "expert_last.pt"),
                best_on="eval_avg",  # recommended for "master"
                restore_best_at_end=True,  # so collection uses BEST automatically
            )
        finally:
            try:
                eval_env.close()
            except Exception:
                pass

        # Collect expert trajectories (possibly mixed with random)
        trajectories = collect_atari_expert_trajectories(
            env,
            expert,
            device=device,
            n_episodes=int(args.episodes_per_task),
            max_len=int(args.max_len),
            expert_action_prob=float(args.expert_action_prob),
            fire_reset=bool(args.fire_reset),
        )

        # Compute per-task average return from collected trajectories
        returns = [float(tr["rewards"].sum()) for tr in trajectories]
        avg_return = float(np.mean(returns)) if returns else 0.0
        max_return = float(np.max(returns)) if returns else 0.0
        task_avg_returns[name] = avg_return

        print(
            f"[expert perf] task={name}, episodes={len(trajectories)}, "
            f"avg_return={avg_return:.2f}, max_return={max_return:.2f}"
        )

        # Save to disk: one .npz file per task
        task_dir = os.path.join(args.out_dir, name)
        os.makedirs(task_dir, exist_ok=True)

        prob_str = f"{args.expert_action_prob:.2f}"
        out_filename = f"expert_trajs_p{prob_str}.npz"
        out_path = os.path.join(task_dir, out_filename)

        save_trajectories_npz(trajectories, out_path)

        # Save metadata
        meta_path = os.path.join(task_dir, "meta.json")
        meta = {
            "task_name": name,
            "env_id": env_id,
            "expert_action_prob": float(args.expert_action_prob),
            "episodes_per_task": int(args.episodes_per_task),
            "max_len": int(args.max_len),
            "total_steps_expert": int(args.total_steps_expert),
            "warmup_steps": int(args.warmup_steps),
            "eval_every": int(args.eval_every),
            "fire_reset": bool(args.fire_reset),
            "device": args.device,
            "spec_file": os.path.abspath(args.spec),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"[save] trajectories -> {out_path}")
        print(f"[save] meta         -> {meta_path}")

        # Close training env for this task (avoid leaks)
        try:
            env.close()
        except Exception:
            pass

    # ====== CL-style metrics + logging for expert performance ======
    task_names = list(envs.keys())
    n_tasks = len(task_names)
    P = np.zeros((n_tasks, n_tasks), dtype=np.float32)

    for i, name in enumerate(task_names):
        P[i, i] = float(task_avg_returns.get(name, 0.0))

    results_obj = BenchmarkResults(
        name=f"DQN-expert:{args.spec}",
        task_names=task_names,
        perf_matrix=P,
    )
    metrics = StandardCLMetrics.compute(results_obj)

    # results.json
    save_json(
        os.path.join(run_dir, "results.json"),
        {
            "name": results_obj.name,
            "task_names": task_names,
            "perf_matrix": P.tolist(),
            "metrics": metrics,
            "mode": "expert",
        },
    )

    # matrix.csv
    save_matrix_csv(
        os.path.join(run_dir, "matrix.csv"),
        task_names,
        P,
    )

    # per_step.json / per_step.csv
    steps = per_step_report(task_names, P)
    save_json(os.path.join(run_dir, "per_step.json"), {"per_step": steps})

    if steps:
        import csv
        with open(os.path.join(run_dir, "per_step.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(steps[0].keys()))
            w.writeheader()
            w.writerows(steps)

    # rez_atari.json
    bench_name_short = bench_short(bench)
    save_json(
        os.path.join(run_dir, f"rez_{bench_name_short}.json"),
        {"metrics": metrics, "strategy": "expert"},
    )

    print("\n=== Atari DQN expert results (avg returns from datasets) ===")
    for name, r in task_avg_returns.items():
        print(f"{name}: {r:.3f}")
    print(f"\n[expert artifacts] saved to: {run_dir}")


if __name__ == "__main__":
    main()
