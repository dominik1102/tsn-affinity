from __future__ import annotations

import re
from collections import deque
from typing import List, Tuple, Optional, Dict, Any

import gymnasium as gym
import numpy as np
import torch


# ============================================================
# Minari Atari env spec (z dokumentacji Minari expert-v0)
# ALE/<Game>-v5, obs_type=rgb, frameskip=4, repeat_action_probability=0, no wrappers
# ============================================================

def _game_slug(env_id: str) -> str:
    # "ALE/Pong-v5" -> "pong"
    m = re.match(r"^ALE/([A-Za-z0-9_]+)-v\d+$", env_id.strip())
    if not m:
        raise ValueError(f"env_id must look like ALE/Pong-v5, got {env_id!r}")
    return m.group(1).lower()


MINARI_ATARI_KWARGS_BASE: Dict[str, Any] = {
    "obs_type": "rgb",
    "frameskip": 4,
    "repeat_action_probability": 0.0,
    "full_action_space": False,
    "max_num_frames_per_episode": 108000,
}


# ============================================================
# DQN preprocess identyczny jak w Twoim eksporcie:
# RGB -> gray -> resize84 -> float[0,1], stack4 => CHW float32
# ============================================================

def _to_gray_uint8(frame_rgb: np.ndarray) -> np.ndarray:
    if frame_rgb.ndim != 3 or frame_rgb.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB, got {frame_rgb.shape}")
    if frame_rgb.dtype != np.uint8:
        frame_rgb = frame_rgb.astype(np.uint8, copy=False)

    gray = (
        0.299 * frame_rgb[..., 0].astype(np.float32)
        + 0.587 * frame_rgb[..., 1].astype(np.float32)
        + 0.114 * frame_rgb[..., 2].astype(np.float32)
    )
    return np.clip(gray, 0.0, 255.0).astype(np.uint8)


def _resize_hw_uint8(img_hw: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    oh, ow = int(out_hw[0]), int(out_hw[1])

    # Pillow (preferred)
    try:
        from PIL import Image  # type: ignore
        pil = Image.fromarray(img_hw, mode="L")
        pil = pil.resize((ow, oh), resample=Image.BILINEAR)
        return np.asarray(pil, dtype=np.uint8)
    except Exception:
        pass

    # OpenCV fallback
    try:
        import cv2  # type: ignore
        return cv2.resize(img_hw, (ow, oh), interpolation=cv2.INTER_AREA).astype(np.uint8)
    except Exception:
        pass

    # Nearest-neighbor fallback
    ys = (np.linspace(0, img_hw.shape[0] - 1, oh)).astype(np.int32)
    xs = (np.linspace(0, img_hw.shape[1] - 1, ow)).astype(np.int32)
    return img_hw[ys][:, xs].astype(np.uint8)


def _preprocess_frame_dqn_hw01(frame_rgb: np.ndarray, dqn_size: int = 84) -> np.ndarray:
    gray = _to_gray_uint8(frame_rgb)
    gray = _resize_hw_uint8(gray, (dqn_size, dqn_size))
    return gray.astype(np.float32) / 255.0  # HW float in [0,1]


class MinariDQNStackWrapper(gym.Wrapper):
    """
    Wrapper: RGB obs -> gray84 float -> stack4 => CHW float32 in [0,1].
    Nie robi frameskip (bo Minari env ma frameskip=4 w base env).
    """

    def __init__(self, env: gym.Env, frame_stack: int = 4, dqn_size: int = 84, clip_rewards: bool = True):
        super().__init__(env)
        self.frame_stack = int(frame_stack)
        self.dqn_size = int(dqn_size)
        self.clip_rewards = bool(clip_rewards)

        self._dq = deque(maxlen=self.frame_stack)

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.frame_stack, self.dqn_size, self.dqn_size),
            dtype=np.float32,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        frame = _preprocess_frame_dqn_hw01(obs, dqn_size=self.dqn_size)
        self._dq.clear()
        for _ in range(self.frame_stack):
            self._dq.append(frame.copy())
        stacked = np.stack(list(self._dq), axis=0).astype(np.float32)
        return stacked, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        if self.clip_rewards:
            reward = float(np.clip(float(reward), -1.0, 1.0))
        frame = _preprocess_frame_dqn_hw01(obs, dqn_size=self.dqn_size)
        self._dq.append(frame)
        stacked = np.stack(list(self._dq), axis=0).astype(np.float32)
        return stacked, float(reward), bool(terminated), bool(truncated), info


def make_minari_atari_env(
    env_id: str,
    seed: Optional[int],
    frame_stack: int = 4,
    dqn_size: int = 84,
    clip_rewards: bool = True,
) -> gym.Env:
    """
    Env zgodny z Minari expert-v0:
      - base ALE/<Game>-v5 z frameskip=4, repeat_action_probability=0, obs_type=rgb
      - wrapper robi tylko preprocess + stack (bez dodatkowego frame_skip!)
    """
    kwargs = dict(MINARI_ATARI_KWARGS_BASE)
    kwargs["game"] = _game_slug(env_id)

    env = gym.make(env_id, **kwargs)
    env = MinariDQNStackWrapper(env, frame_stack=frame_stack, dqn_size=dqn_size, clip_rewards=clip_rewards)

    # seed reset (opcjonalnie)
    if seed is not None:
        env.reset(seed=int(seed))

    return env


# ============================================================
# Dataset loader (twoje NPZ)
# ============================================================

def load_npz_dataset_for_task(npz_path: str):
    d = np.load(npz_path, allow_pickle=False)
    required = {"observations", "actions", "rewards", "dones", "episode_lengths"}
    if not required.issubset(set(d.files)):
        raise ValueError(f"Bad npz format. keys={list(d.files)}")

    observations = d["observations"]
    actions = d["actions"].reshape(-1)
    rewards = d["rewards"].reshape(-1)
    dones = d["dones"].reshape(-1)
    episode_lengths = d["episode_lengths"].astype(np.int64)

    total = int(episode_lengths.sum())
    if not (observations.shape[0] == actions.shape[0] == rewards.shape[0] == dones.shape[0] == total):
        raise ValueError("Inconsistent shapes vs episode_lengths")

    episodes_obs: List[np.ndarray] = []
    episodes_actions: List[np.ndarray] = []
    episodes_rewards: List[np.ndarray] = []

    idx = 0
    for L in episode_lengths:
        L = int(L)
        episodes_obs.append(observations[idx:idx + L])
        episodes_actions.append(actions[idx:idx + L])
        episodes_rewards.append(rewards[idx:idx + L])
        idx += L

    returns = np.array([float(np.sum(r)) for r in episodes_rewards], dtype=np.float32)
    return episodes_obs, episodes_actions, episodes_rewards, returns


# ============================================================
# Offline minibatches (DT training)
# ============================================================

def make_offline_minibatches(
    episodes_obs: List[np.ndarray],
    episodes_actions: List[np.ndarray],
    episodes_rewards: List[np.ndarray],
    seq_len: int,
    batch_size: int,
    device: torch.device,
):
    assert len(episodes_obs) > 0, "No episodes"
    frame_shape = episodes_obs[0].shape[1:]  # (C,H,W)

    # RTG precompute
    ep_rtg: List[np.ndarray] = []
    ep_len: List[int] = []
    for rew in episodes_rewards:
        r = np.asarray(rew, dtype=np.float32)
        rtg = np.flip(np.cumsum(np.flip(r)))  # gamma=1.0
        ep_rtg.append(rtg.astype(np.float32))
        ep_len.append(int(len(r)))

    # sample episodes proportional to length (bardziej stabilne)
    lens = np.array(ep_len, dtype=np.float64)
    probs = lens / max(lens.sum(), 1.0)

    def loader():
        while True:
            obs_batch = np.zeros((batch_size, seq_len) + frame_shape, dtype=np.float32)
            act_batch = np.full((batch_size, seq_len), -1, dtype=np.int64)
            rtg_batch = np.zeros((batch_size, seq_len, 1), dtype=np.float32)
            ts_batch = np.zeros((batch_size, seq_len), dtype=np.int64)
            mask_batch = np.zeros((batch_size, seq_len), dtype=np.bool_)

            for b in range(batch_size):
                ep_idx = int(np.random.choice(len(episodes_obs), p=probs))
                L = ep_len[ep_idx]
                if L <= 0:
                    continue

                if L >= seq_len:
                    start = np.random.randint(0, L - seq_len + 1)
                    end = start + seq_len
                    length = seq_len
                else:
                    start, end, length = 0, L, L

                obs_slice = episodes_obs[ep_idx][start:end].astype(np.float32, copy=False)
                # dataset jest już float[0,1] z exportu; ale gdyby było uint8:
                if obs_slice.dtype == np.uint8 or (obs_slice.size > 0 and obs_slice.max() > 1.5):
                    obs_slice = obs_slice.astype(np.float32) / 255.0

                obs_batch[b, :length] = obs_slice
                act_batch[b, :length] = episodes_actions[ep_idx][start:end].astype(np.int64)
                rtg_batch[b, :length, 0] = ep_rtg[ep_idx][start:end]
                ts_batch[b, :length] = np.arange(start, end, dtype=np.int64)
                mask_batch[b, :length] = True

            yield (
                torch.tensor(obs_batch, device=device, dtype=torch.float32),
                torch.tensor(act_batch, device=device, dtype=torch.long),
                torch.tensor(rtg_batch, device=device, dtype=torch.float32),
                torch.tensor(ts_batch, device=device, dtype=torch.long),
                torch.tensor(mask_batch, device=device, dtype=torch.bool),
            )

    return loader()


# ============================================================
# Eval + debug replay
# ============================================================

def _get_fire_action(env: gym.Env) -> Optional[int]:
    try:
        meanings = env.unwrapped.get_action_meanings()
        if isinstance(meanings, (list, tuple)) and "FIRE" in meanings:
            return int(meanings.index("FIRE"))
    except Exception:
        pass
    return None


def debug_replay_episode(
    env: gym.Env,
    ep_actions: np.ndarray,
    dataset_return: float,
    seed: Optional[int] = 0,
) -> float:
    """
    Replays EXACT action sequence (bez auto-fire!) i zwraca env_return.
    Jak env jest zgodny z Minari, powinno wyjść ~dataset_return (deterministycznie).
    """
    obs, info = env.reset(seed=seed)
    total = 0.0
    for a in ep_actions:
        obs, r, terminated, truncated, info = env.step(int(a))
        total += float(r)
        if terminated or truncated:
            break
    return float(total)


@torch.no_grad()
def evaluate_dt(
    model,
    env: gym.Env,
    episodes: int,
    device: torch.device,
    max_steps: int,
    target_return: float,
    auto_fire: bool = False,
    auto_fire_on_life_loss: bool = False,
) -> float:
    rets = []
    fire_a = _get_fire_action(env) if (auto_fire or auto_fire_on_life_loss) else None

    for ep in range(int(episodes)):
        model.reset_history()
        obs, info = env.reset()

        # optional: fire after reset (do not count to DT history)
        if auto_fire and fire_a is not None:
            obs, r, terminated, truncated, info = env.step(fire_a)
            if terminated or truncated:
                rets.append(float(r))
                continue

        total = 0.0
        rtg_remaining = float(target_return)
        last_lives = info.get("lives", None)

        for t in range(int(max_steps)):
            # DT expects CHW float
            a = model.act(
                obs,
                rtg_scalar=rtg_remaining,
                t=t,
                device=str(device),
                n_actions=env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else None,
            )

            obs, r, terminated, truncated, info = env.step(int(a))
            total += float(r)
            rtg_remaining -= float(r)

            # optional: fire on life loss (Breakout-style)
            if auto_fire_on_life_loss and fire_a is not None:
                lives = info.get("lives", None)
                if (lives is not None) and (last_lives is not None) and (lives < last_lives):
                    obs2, r2, term2, trunc2, info2 = env.step(fire_a)
                    obs = obs2
                    # zwykle r2=0, ale liczmy uczciwie
                    total += float(r2)
                    rtg_remaining -= float(r2)
                    terminated = terminated or term2
                    truncated = truncated or trunc2
                    info = info2
                last_lives = lives

            if terminated or truncated:
                break

        rets.append(total)

    return float(np.mean(rets)) if rets else 0.0
