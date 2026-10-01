#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

TASKS = [
    ("D_Breakout", "ALE/Breakout-v5"),
    ("A_Alien", "ALE/Alien-v5"),
    ("B_Atlantis", "ALE/Atlantis-v5"),
    ("C_Boxing", "ALE/Boxing-v5"),
    ("E_Centipede", "ALE/Centipede-v5"),
    ("F_DoubleDunk", "ALE/DoubleDunk-v5"),
    ("G_Freeway", "ALE/Freeway-v5"),
    ("H_Pong", "ALE/Pong-v5"),
    ("I_SpaceInvaders", "ALE/SpaceInvaders-v5"),
    ("J_Tennis", "ALE/Tennis-v5"),
]

specs = []
for i, (name, env_id) in enumerate(TASKS):
    specs.append({
        "name": name,
        "env_id": None,
        "seed": i,
        "params": {
            "game": env_id,
            "frame_stack": 4,
            "dqn_size": 84,
            "clip_rewards": True,
        },
    })

out = Path("configs/specs_atari_cl_10_breakout_first_minari_like.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(specs, indent=2), encoding="utf-8")
print("wrote", out)
print([x["name"] for x in specs])
