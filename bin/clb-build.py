#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


# =========================
#  Presety gier (CL)
# =========================

ATARI_CL_5 = [
    "Alien",
    "Atlantis",
    "Boxing",
    "Breakout",
    "Centipede",
]

ATARI_CL_10 = [
    "Alien",
    "Atlantis",
    "Boxing",
    "Breakout",
    "Centipede",
    "Double Dunk",
    "Freeway",
    "Pong",
    "Space Invaders",
    "Tennis",
]


# =========================
#  Helpers
# =========================

def _pascal_case(name: str) -> str:
    """
    "Space Invaders" -> "SpaceInvaders"
    "Double Dunk" -> "DoubleDunk"
    """
    toks = re.split(r"[^A-Za-z0-9]+", name.strip())
    toks = [t for t in toks if t]
    return "".join(t[:1].upper() + t[1:] for t in toks)


def _idx_to_prefix(i: int) -> str:
    """
    0->A, 1->B, ..., 25->Z, 26->AA...
    (na wypadek gdybyś kiedyś miał >26 tasków)
    """
    letters = []
    x = i
    while True:
        letters.append(chr(ord("A") + (x % 26)))
        x = x // 26 - 1
        if x < 0:
            break
    return "".join(reversed(letters))


def _ale_env_id(game_title: str, version: int = 5) -> str:
    """
    "Alien" -> "ALE/Alien-v5"
    "Space Invaders" -> "ALE/SpaceInvaders-v5"
    """
    return f"ALE/{_pascal_case(game_title)}-v{int(version)}"


def _minari_like_params(game_env_id: str) -> Dict[str, Any]:
    """
    Parametry zgodne z tym co zapisało Ci się przy eksporcie
    (noop_max=30, sticky_actions=True, repeat_action_prob=0.25, terminal_on_life_loss=True...).
    """
    return {
        "game": game_env_id,

        # kluczowe rzeczy, które MUSZĄ być spójne między export->train->eval:
        "frameskip": 4,
        "noop_max": 30,
        "sticky_actions": True,
        "repeat_action_prob": 0.25,
        "terminal_on_life_loss": True,
        "clip_rewards": True,

        # preprocessing:
        "grayscale_obs": True,
        "scale_obs": False,
        "frame_stack": 4,
    }


def build_atari_task_specs(
    games: List[str],
    seed: int,
    *,
    ale_version: int = 5,
) -> List[Dict[str, Any]]:
    """
    Buduje listę TaskSpecs w formacie JSON, który już używasz:
      [
        {"name":"A_Pong","env_id":null,"seed":0,"params":{...}},
        ...
      ]
    """
    specs: List[Dict[str, Any]] = []
    for i, game in enumerate(games):
        prefix = _idx_to_prefix(i)
        game_pascal = _pascal_case(game)
        env_id = _ale_env_id(game, version=ale_version)

        specs.append(
            {
                "name": f"{prefix}_{game_pascal}",
                "env_id": None,
                "seed": int(seed) + i,
                "params": _minari_like_params(env_id),
            }
        )
    return specs


def save_specs_json(path: Path, specs: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(specs, indent=2), encoding="utf-8")


# =========================
#  CLI
# =========================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate Atari CL TaskSpecs (Minari-like env params) into configs/."
    )
    ap.add_argument("--seed", type=int, default=0, help="Base seed (task i gets seed+ i)")
    ap.add_argument(
        "--out-dir",
        type=str,
        default="configs",
        help="Output directory for JSON specs (default: configs)",
    )
    ap.add_argument(
        "--ale-version",
        type=int,
        default=5,
        help="ALE env version used in env_id (default: 5 -> ALE/<Game>-v5)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    specs_5 = build_atari_task_specs(ATARI_CL_5, seed=args.seed, ale_version=args.ale_version)
    specs_10 = build_atari_task_specs(ATARI_CL_10, seed=args.seed, ale_version=args.ale_version)

    out_5 = out_dir / "specs_atari_cl_5_minari_like.json"
    out_10 = out_dir / "specs_atari_cl_10_minari_like.json"

    save_specs_json(out_5, specs_5)
    save_specs_json(out_10, specs_10)

    print(f"[OK] Saved {len(specs_5)} tasks -> {out_5}")
    print(f"[OK] Saved {len(specs_10)} tasks -> {out_10}")
    print()
    print("Task names (CL-5):  ", ", ".join([t["name"] for t in specs_5]))
    print("Task names (CL-10): ", ", ".join([t["name"] for t in specs_10]))


if __name__ == "__main__":
    main()
