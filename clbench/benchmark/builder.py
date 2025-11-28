
from __future__ import annotations
from typing import List
from ..core.task_spec import TaskSpec

def default_cartpole_presets(seed: int = 0) -> List[TaskSpec]:
    def S(name, s, **p): return TaskSpec(name=name, env_id="CartPole-v1", seed=s, params=p or None)
    return [
        S("A_default", seed+0),
        S("B_heavier_pole", seed+1, masspole=0.2),
        S("C_stronger_gravity", seed+2, gravity=12.0),
        S("D_longer_pole", seed+3, length=1.2),
        S("E_weaker_force", seed+4, force_mag=7.5),
        S("F_faster_dynamics", seed+5, tau=0.015),
        S("G_combo", seed+6, gravity=11.0, length=1.1, force_mag=8.5),
    ]

def default_atari_presets(seed: int = 0) -> List[TaskSpec]:
    games = ["Pong", "Breakout", "Seaquest"]
    specs: List[TaskSpec] = []
    for i, g in enumerate(games):
        params = dict(game=f"ALE/{g}-v5", frameskip=4, noop_max=30,
                      sticky_actions=True, repeat_action_prob=0.25,
                      grayscale_obs=True, scale_obs=False,
                      terminal_on_life_loss=True, clip_rewards=True, frame_stack=4)
        specs.append(TaskSpec(name=f"{chr(65+i)}_{g}", seed=seed+i, params=params))
    return specs

def build_cartpole_benchmark(kind: str = "cartpole-cl-7", seed: int = 0) -> List[TaskSpec]:
    return default_cartpole_presets(seed)

def build_atari_benchmark(kind: str = "atari-cl-3", seed: int = 0) -> List[TaskSpec]:
    if kind.lower() == "atari-cl-3":
        return default_atari_presets(seed)
    if kind.lower() == "atari-pong-variants":
        specs: List[TaskSpec] = []
        variants = [
            dict(sticky_actions=True, repeat_action_prob=0.25, frameskip=4),
            dict(sticky_actions=False, repeat_action_prob=0.0, frameskip=4),
            dict(sticky_actions=True, repeat_action_prob=0.25, frameskip=2),
        ]
        for i, extra in enumerate(variants):
            params = dict(game="ALE/Pong-v5", frameskip=4, noop_max=30,
                          sticky_actions=True, repeat_action_prob=0.25,
                          grayscale_obs=True, scale_obs=False,
                          terminal_on_life_loss=True, clip_rewards=True, frame_stack=4)
            params.update(extra)
            specs.append(TaskSpec(name=f"{chr(65+i)}_Pong_var{i+1}", seed=seed+i, params=params))
        return specs
    raise ValueError(f"Unknown Atari benchmark kind: {kind}")
