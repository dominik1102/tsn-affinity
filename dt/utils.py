from __future__ import annotations

import re
from collections import deque, Counter
from typing import List, Tuple, Optional, Dict, Any, Union

import gymnasium as gym
import numpy as np
import torch

from dt.model import DecisionTransformer


# ============================================================
# Minari Atari env spec (from Minari expert-v0 docs)
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
    Does not perform frameskip (Minari base env has frameskip=4).
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
    Env compatible with Minari expert-v0:
      - base ALE/<Game>-v5 with frameskip=4, repeat_action_probability=0, obs_type=rgb
      - wrapper only does preprocess + stack (no additional frame_skip)
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
                # dataset is already float[0,1] from the export; but if it was uint8:
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
    Replays EXACT action sequence (no auto-fire!) and returns env_return.
    If the env is compatible with Minari, this should match ~dataset_return (deterministically).
    """
    obs, info = env.reset(seed=seed)
    total = 0.0
    for a in ep_actions:
        obs, r, terminated, truncated, info = env.step(int(a))
        total += float(r)
        if terminated or truncated:
            break
    return float(total)

def _stack_motion(obs) -> float:
    """
    Breakout: the ball is only a few pixels -> mean(diff) can be <1e-4.
    We use max(diff) to reliably detect motion.
    Also handles layout (S,H,W) and (H,W,S).
    """
    a = np.asarray(obs, dtype=np.float32)
    if a.ndim != 3:
        return 0.0

    # normalize if uint8
    if a.max() > 1.5:
        a = a / 255.0

    # stack first (S,H,W)
    if a.shape[0] >= 2 and a.shape[0] <= 10:
        prev, cur = a[-2], a[-1]
    # stack last (H,W,S)
    elif a.shape[-1] >= 2 and a.shape[-1] <= 10:
        prev, cur = a[..., -2], a[..., -1]
    else:
        return 0.0

    diff = np.abs(cur - prev)
    return float(diff.max())



def _robust_fire_after_reset(env: gym.Env, fire_a: int, max_tries: int = 20):
    obs, info = env.reset()

    if _stack_motion(obs) > 1e-3:
        return obs, info

    noop_a = 0  # w minimal action set NOOP zwykle 0

    for _ in range(max_tries):
        obs, r, terminated, truncated, info = env.step(int(fire_a))
        if terminated or truncated:
            obs, info = env.reset()
            continue

        # give 1 tick for 'ball movement'
        obs, r, terminated, truncated, info = env.step(int(noop_a))
        if terminated or truncated:
            obs, info = env.reset()
            continue

        if _stack_motion(obs) > 1e-3:
            break

    return obs, info




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
        if auto_fire and fire_a is not None:
            obs, info = _robust_fire_after_reset(env, fire_a, max_tries=50)
        else:
            obs, info = env.reset()

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
            if auto_fire_on_life_loss and fire_a is not None and not (terminated or truncated):
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


@torch.no_grad()
def evaluate_dt_forward(
    model: DecisionTransformer,
    env: gym.Env,
    episodes: int,
    device: torch.device,
    max_steps: int,
    target_return: float,
    *,
    seed: Optional[int] = None,
    greedy: bool = True,
    clamp_to_env_actions: bool = True,
    auto_fire: bool = False,
    auto_fire_on_life_loss: bool = False,
    debug_action_hist: bool = False,
) -> float:
    """
    Eval bez model.act() (bez ukrytej historii), ale ZGODNY z tokenizacją (R,s,a):
    - w wejściu do modelu akcje są "as-is" dla przeszłych kroków,
    - dla bieżącego kroku dokładamy dummy (0), bo a_t jest po s_t i i tak nie wpływa na predykcję.
    """
    assert isinstance(env.action_space, gym.spaces.Discrete), "Only Discrete supported here"
    n_env = int(env.action_space.n)

    # kontekst K
    K = getattr(model, "seq_len", None) or getattr(model, "K", None)
    if K is None:
        cfg = getattr(model, "config", None)
        K = getattr(cfg, "K", 20) if cfg is not None else 20
    K = int(K)

    # FIRE id (opcjonalnie)
    fire_id = None
    try:
        meanings = env.unwrapped.get_action_meanings()
        if isinstance(meanings, (list, tuple)) and "FIRE" in meanings:
            fire_id = int(meanings.index("FIRE"))
    except Exception:
        pass

    hist = Counter()
    rets: List[float] = []

    for ep in range(int(episodes)):
        # deterministyczny reset
        if seed is None:
            obs, info = env.reset()
        else:
            obs, info = env.reset(seed=int(seed + ep))

        total = 0.0
        rtg = float(target_return)

        # historia tokenów (obs/rtg/ts mają długość L)
        obs_d = deque(maxlen=K)
        rtg_d = deque(maxlen=K)
        ts_d  = deque(maxlen=K)

        # historia wykonanych akcji (ma długość L-1)
        act_d = deque(maxlen=K)  # będziemy brać last(L-1) przy budowie wejścia

        t = 0
        obs_d.append(np.asarray(obs, dtype=np.float32))
        rtg_d.append(float(rtg))
        ts_d.append(int(t))

        terminated = truncated = False
        last_lives = info.get("lives", None)

        # jeżeli auto_fire: wymuś FIRE na pierwszym kroku decyzyjnym
        force_fire_next = bool(auto_fire and (fire_id is not None))

        for step in range(int(max_steps)):
            L = len(obs_d)

            # ✅ actions_seq: (L-1) wykonanych akcji + dummy 0 na bieżący krok
            past = list(act_d)
            if len(past) > L - 1:
                past = past[-(L - 1):]
            actions_seq = past + [0]
            assert len(actions_seq) == L

            obs_t = torch.from_numpy(np.stack(list(obs_d), axis=0)).unsqueeze(0).to(device=device, dtype=torch.float32)
            act_t = torch.from_numpy(np.asarray(actions_seq, dtype=np.int64)).unsqueeze(0).to(device=device, dtype=torch.long)
            rtg_t = torch.from_numpy(np.asarray(list(rtg_d), dtype=np.float32)).reshape(1, L, 1).to(device=device, dtype=torch.float32)
            ts_t  = torch.from_numpy(np.asarray(list(ts_d), dtype=np.int64)).reshape(1, L).to(device=device, dtype=torch.long)
            mask  = torch.ones((1, L), device=device, dtype=torch.bool)

            logits = model(obs_t, act_t, rtg_t, ts_t, attention_mask=mask)  # [1,L,A]
            last_logits = logits[0, -1]  # [A]

            if clamp_to_env_actions:
                last_logits = last_logits[:n_env]

            # wybór akcji
            if force_fire_next and fire_id is not None:
                a = int(fire_id)
                force_fire_next = False
            else:
                if greedy:
                    a = int(torch.argmax(last_logits).item())
                else:
                    probs = torch.softmax(last_logits, dim=-1)
                    a = int(torch.multinomial(probs, 1).item())

            hist[a] += 1

            # krok środowiska
            obs, r, terminated, truncated, info = env.step(a)
            r = float(r)

            total += r
            rtg -= r

            # life loss -> wymuś FIRE w następnym kroku
            if auto_fire_on_life_loss and fire_id is not None and not (terminated or truncated):
                lives = info.get("lives", None)
                if (lives is not None) and (last_lives is not None) and (lives < last_lives):
                    force_fire_next = True
                last_lives = lives

            # update historii (po wykonaniu akcji)
            act_d.append(int(a))
            t += 1
            obs_d.append(np.asarray(obs, dtype=np.float32))
            rtg_d.append(float(rtg))
            ts_d.append(int(t))

            if terminated or truncated:
                break

        rets.append(total)

    if debug_action_hist:
        print("[eval] action_hist:", dict(hist))

    return float(np.mean(rets)) if rets else 0.0



# ============================================================
# Panda (continuous actions) evaluation
# ============================================================

def _pad_1d(x: np.ndarray, target_dim: Optional[int]) -> np.ndarray:
    """Pad or slice a 1D vector to target_dim (if provided)."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if target_dim is None:
        return x
    d = int(target_dim)
    if x.shape[0] == d:
        return x
    if x.shape[0] > d:
        return x[:d]
    out = np.zeros((d,), dtype=np.float32)
    out[: x.shape[0]] = x
    return out


def _get_context_len(model, default: int = 20) -> int:
    """Best-effort: get DT context length K/seq_len from various model configs."""
    for attr in ("seq_len", "K"):
        v = getattr(model, attr, None)
        if v is not None:
            return int(v)
    cfg = getattr(model, "config", None)
    if cfg is not None:
        v = getattr(cfg, "K", None)
        if v is not None:
            return int(v)
        v = getattr(cfg, "seq_len", None)
        if v is not None:
            return int(v)
    return int(default)



@torch.no_grad()
def evaluate_dt_panda(
    model,
    env: gym.Env,
    episodes: int = 5,
    device: Union[str, torch.device] = "cuda",
    max_steps: Optional[int] = None,
    target_return: float = 0.0,
    seed: Optional[int] = None,
    obs_pad_to: Optional[int] = None,
    act_pad_to: Optional[int] = None,
    clip_action: bool = True,
    gamma: float = 1.0,
    rtg_clip: Optional[Tuple[Optional[float], Optional[float]]] = None,  # (min,max)
    *,
    unwrap_flatten_observation: bool = True,
    obs_keys: Optional[Tuple[str, ...]] = None,
    use_prev_action: bool = True,
    timestep_clip_max: Optional[int] = None,
) -> float:
    """
    Panda (continuous actions) evaluation.

    Notes:
      - obs_keys can be passed from the loader/runner so train and eval use identical flatten order,
      - use_prev_action allows an ablation: whether prev_action helps or hurts,
      - timestep_clip_max prevents stepping beyond timesteps seen during training,
      - if env is FlattenObservation, we can unwrap and flatten observations ourselves.
    """

    # allow wrappers like strategy/model.model
    if (not hasattr(model, "act")) and hasattr(model, "model"):
        model = model.model
    if not hasattr(model, "act"):
        raise AttributeError("evaluate_dt_panda: `model` must implement `.act(obs, rtg_scalar, t, prev_action=...)`")

    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")

    rollout_env = env
    if unwrap_flatten_observation:
        try:
            from gymnasium.wrappers import FlattenObservation  # type: ignore
            if isinstance(env, FlattenObservation):
                rollout_env = env.env
        except Exception:
            if env.__class__.__name__ == "FlattenObservation" and hasattr(env, "env"):
                rollout_env = env.env

    if max_steps is None:
        max_steps = getattr(getattr(rollout_env, "spec", None), "max_episode_steps", None)
    if max_steps is None:
        max_steps = 50

    if not isinstance(rollout_env.action_space, gym.spaces.Box):
        raise ValueError(f"evaluate_dt_panda expects Box action space, got: {type(rollout_env.action_space)}")

    env_act_dim = int(np.prod(rollout_env.action_space.shape))
    low = np.asarray(rollout_env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(rollout_env.action_space.high, dtype=np.float32).reshape(-1)

    model_act_dim = getattr(model, "act_dim", None)
    if model_act_dim is not None:
        model_act_dim = int(model_act_dim)

    obs_keys_local = obs_keys
    if obs_keys_local is None and isinstance(rollout_env.observation_space, gym.spaces.Dict):
        obs_keys_local = tuple(rollout_env.observation_space.spaces.keys())

    if timestep_clip_max is not None:
        timestep_clip_max = max(0, int(timestep_clip_max))

    def _pad_or_trunc(x: np.ndarray, size: Optional[int]) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if size is None:
            return x
        size = int(size)
        if x.size < size:
            return np.pad(x, (0, size - x.size), mode="constant")
        if x.size > size:
            return x[:size]
        return x

    def _flatten_obs(obs) -> np.ndarray:
        if isinstance(obs, dict):
            if obs_keys_local is not None and all(k in obs for k in obs_keys_local):
                parts = [np.asarray(obs[k], dtype=np.float32).ravel() for k in obs_keys_local]
                return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

            from gymnasium.spaces.utils import flatten as gym_flatten
            return gym_flatten(rollout_env.observation_space, obs).astype(np.float32, copy=False)

        return np.asarray(obs, dtype=np.float32).reshape(-1)

    returns: List[float] = []

    try:
        model.eval()
    except Exception:
        pass

    for ep in range(int(episodes)):
        if hasattr(model, "reset_history") and callable(getattr(model, "reset_history")):
            model.reset_history()

        if seed is None:
            obs, info = rollout_env.reset()
        else:
            obs, info = rollout_env.reset(seed=int(seed + ep))

        total = 0.0
        rtg = float(target_return)
        prev_action = None

        for t in range(int(max_steps)):
            obs_vec = _flatten_obs(obs)
            obs_vec = _pad_or_trunc(obs_vec, obs_pad_to)

            t_model = int(t)
            if timestep_clip_max is not None:
                t_model = min(t_model, int(timestep_clip_max))

            a_pred = model.act(
                obs_vec,
                rtg_scalar=rtg,
                t=t_model,
                prev_action=(prev_action if use_prev_action else None),
                device=str(device),
            )
            a_pred = _pad_or_trunc(a_pred, act_pad_to)

            if a_pred.size < env_act_dim:
                a_env_flat = np.pad(a_pred, (0, env_act_dim - a_pred.size), mode="constant")
            else:
                a_env_flat = a_pred[:env_act_dim]

            if clip_action:
                a_env_flat = np.clip(a_env_flat, low[:env_act_dim], high[:env_act_dim])

            a_env = a_env_flat.astype(np.float32, copy=False).reshape(rollout_env.action_space.shape)

            obs, r, terminated, truncated, info = rollout_env.step(a_env)
            r = float(r)
            total += r

            if use_prev_action:
                pa = a_env_flat.reshape(-1).astype(np.float32, copy=False)
                if model_act_dim is not None:
                    prev_action = _pad_or_trunc(pa, model_act_dim)
                else:
                    prev_action = pa
            else:
                prev_action = None

            if gamma == 1.0:
                rtg = rtg - r
            else:
                rtg = (rtg - r) / float(gamma)

            if rtg_clip is not None:
                lo, hi = rtg_clip
                if lo is not None:
                    rtg = max(rtg, float(lo))
                if hi is not None:
                    rtg = min(rtg, float(hi))

            if terminated or truncated:
                break

        returns.append(total)

    return float(np.mean(returns)) if returns else 0.0


@torch.no_grad()
def evaluate_dt_panda_cl(
    model,
    env: gym.Env,
    *,
    episodes: int,
    device: torch.device,
    max_steps: int,
    target_return: float,
    seed: int,
    obs_keys: Tuple[str, ...],
    obs_pad_to: int,
    act_pad_to: int,
    timestep_clip_max: Optional[int],
    gamma: float = 1.0,
    clip_action: bool = True,
) -> float:
    # unwrap if strategy wrapper
    if (not hasattr(model, "act")) and hasattr(model, "model"):
        model = model.model
    if not hasattr(model, "act"):
        raise AttributeError("Model must implement .act()")

    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError("Panda eval expects Box action space")

    env_act_dim = int(np.prod(env.action_space.shape))
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)

    model_act_dim = getattr(model, "act_dim", None)
    model_act_dim = int(model_act_dim) if model_act_dim is not None else None

    def _pad_or_trunc_1d(x: np.ndarray, size: int) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size < size:
            return np.pad(x, (0, size - x.size), mode="constant")
        if x.size > size:
            return x[:size]
        return x

    def _flatten_obs(obs: Any) -> np.ndarray:
        if isinstance(obs, dict):
            parts = [np.asarray(obs[k], dtype=np.float32).ravel() for k in obs_keys]
            return np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    returns: List[float] = []
    model.eval()

    for ep in range(int(episodes)):
        if hasattr(model, "reset_history"):
            model.reset_history()

        obs, info = env.reset(seed=int(seed + ep))

        total = 0.0
        rtg = float(target_return)
        prev_action = None

        for t in range(int(max_steps)):
            obs_vec = _flatten_obs(obs)
            obs_vec = _pad_or_trunc_1d(obs_vec, int(obs_pad_to))

            t_model = int(t)
            if timestep_clip_max is not None:
                t_model = min(t_model, int(timestep_clip_max))

            a_pred = model.act(
                obs_vec,
                rtg_scalar=float(rtg),
                t=int(t_model),
                prev_action=prev_action,
                device=str(device),
            )
            a_pred = _pad_or_trunc_1d(a_pred, int(act_pad_to))

            # dopasuj do env act dim
            if a_pred.size < env_act_dim:
                a_env_flat = np.pad(a_pred, (0, env_act_dim - a_pred.size), mode="constant")
            else:
                a_env_flat = a_pred[:env_act_dim]

            if clip_action:
                a_env_flat = np.clip(a_env_flat, low[:env_act_dim], high[:env_act_dim])

            a_env = a_env_flat.astype(np.float32, copy=False).reshape(env.action_space.shape)

            obs, r, terminated, truncated, info = env.step(a_env)
            r = float(r)
            total += r

            # prev_action = executed action (after clip), adjusted to model_act_dim
            pa = a_env_flat.reshape(-1).astype(np.float32, copy=False)
            if model_act_dim is not None:
                prev_action = _pad_or_trunc_1d(pa, model_act_dim)
            else:
                prev_action = pa

            # update RTG (as in single)
            if gamma == 1.0:
                rtg = rtg - r
            else:
                rtg = (rtg - r) / float(gamma)

            if terminated or truncated:
                break

        returns.append(total)

    return float(np.mean(returns)) if returns else 0.0

