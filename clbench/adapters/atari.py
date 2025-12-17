from __future__ import annotations

import warnings
from typing import Any
import numpy as np

# Ensures ALE environments are available
import ale_py  # noqa: F401

try:
    import gymnasium as gym
    from gymnasium.spaces import Box
    GYM_IS_GYMNASIUM = True
except Exception:  # pragma: no cover
    import gym  # type: ignore
    from gym.spaces import Box  # type: ignore
    GYM_IS_GYMNASIUM = False


# -----------------------------------------------------------------------------
# Register ALE envs for gymnasium (needed in many setups)
# -----------------------------------------------------------------------------
if GYM_IS_GYMNASIUM:
    try:
        # gymnasium expects explicit registration in many versions:
        # gym.register_envs(ale_py)
        if hasattr(gym, "register_envs"):
            gym.register_envs(ale_py)  # type: ignore[attr-defined]
    except Exception:
        # Don't hard-fail; gym may already have them registered
        pass


def _box_low(old: Box) -> float:
    return float(np.min(old.low)) if np.ndim(old.low) else float(old.low)


def _box_high(old: Box) -> float:
    return float(np.max(old.high)) if np.ndim(old.high) else float(old.high)


class ChannelFirstWrapper(gym.ObservationWrapper):
    """
    Convert observations to channel-first layout (CHW) for conv nets.

    Handles:
      - HWC        -> CHW         (e.g. (84,84,3))
      - HW         -> (1,H,W)     (e.g. (84,84))
      - (S,H,W,C)  -> (S*C,H,W)   (frame-stacked RGB)
    Leaves unchanged:
      - (C,H,W) already CHW
      - (S,H,W) stacked grayscale (treat S as channels)
    """

    def __init__(self, env):
        super().__init__(env)
        old = env.observation_space

        if not isinstance(old, Box):
            self.observation_space = old
            return

        shape = old.shape
        low = _box_low(old)
        high = _box_high(old)

        # HW -> (1,H,W)
        if len(shape) == 2:
            H, W = shape
            self.observation_space = Box(low=low, high=high, shape=(1, H, W), dtype=old.dtype)
            return

        # HWC -> CHW (only if last dim looks like channels)
        if len(shape) == 3:
            H, W, C = shape
            if C in (1, 3, 4):
                self.observation_space = Box(low=low, high=high, shape=(C, H, W), dtype=old.dtype)
                return

        # (S,H,W,C) -> (S*C,H,W)
        if len(shape) == 4:
            S, H, W, C = shape
            if C in (1, 3, 4):
                self.observation_space = Box(low=low, high=high, shape=(S * C, H, W), dtype=old.dtype)
                return

        # Otherwise: keep as-is (already CHW or non-image)
        self.observation_space = old

    def observation(self, obs):
        arr = np.asarray(obs)

        # HW -> (1,H,W)
        if arr.ndim == 2:
            return arr[None, ...]

        # HWC -> CHW (channels last and looks like RGB/gray)
        if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
            return np.transpose(arr, (2, 0, 1))

        # (S,H,W,C) -> (S*C,H,W)
        if arr.ndim == 4 and arr.shape[-1] in (1, 3, 4):
            S, H, W, C = arr.shape
            arr = np.transpose(arr, (0, 3, 1, 2))  # (S,C,H,W)
            return arr.reshape(S * C, H, W)

        return arr


class AtariAdapter:
    """Adapter for ALE Atari with correct preprocessing and CHW output."""

    def _sign(self, r: float) -> float:
        return 1.0 if r > 0 else (-1.0 if r < 0 else 0.0)

    def _try_make(self, env_id: str, frameskip: int | None, rep_prob: float | None):
        """
        Try multiple kwarg combinations, because different gym/gymnasium + ALE versions
        accept different arguments. This avoids failing in fallback paths.
        """
        attempts: list[dict[str, Any]] = []

        if frameskip is not None and rep_prob is not None:
            attempts.append({"frameskip": int(frameskip), "repeat_action_probability": float(rep_prob)})
        if frameskip is not None:
            attempts.append({"frameskip": int(frameskip)})
        if rep_prob is not None:
            attempts.append({"repeat_action_probability": float(rep_prob)})
        attempts.append({})

        last_err: Exception | None = None
        for kw in attempts:
            try:
                return gym.make(env_id, **kw)
            except TypeError as e:
                last_err = e
                continue
            except Exception as e:
                # Could be "env_id not found" or other; caller will try other ids.
                last_err = e
                continue

        if last_err is not None:
            raise last_err
        raise RuntimeError(f"Failed to create env {env_id} (unknown error).")

    def _make_game(self, game: str, seed: int | None, repeat_action_prob: float):
        """
        CRITICAL:
        AtariPreprocessing must control frame_skip.
        Therefore base env should have frameskip=1 (no internal skip), if possible.
        """
        rep_prob = float(repeat_action_prob)

        # Build candidate IDs (support user passing "Pong", "Pong-v4", etc.)
        candidates: list[str] = [game]

        # If user passed something like "Pong" or "Pong-v4", try reasonable fallbacks
        if "/" not in game:
            # split version if present
            if "-v" in game:
                base = game.split("-v")[0]
                ver = "-v" + game.split("-v")[1]
            else:
                base, ver = game, ""

            # Gym classic Atari often needs NoFrameskip-v4
            if "NoFrameskip" not in base:
                if ver:
                    candidates.append(f"{base}NoFrameskip{ver}")
                candidates.append(f"{base}NoFrameskip-v4")

            # Gymnasium ALE naming
            candidates.append(f"ALE/{base}-v5")

        # De-duplicate while keeping order
        seen = set()
        uniq_candidates: list[str] = []
        for cid in candidates:
            if cid and cid not in seen:
                uniq_candidates.append(cid)
                seen.add(cid)

        env = None
        last_err: Exception | None = None

        # Try candidates with frameskip=1 first (best practice), then without if needed
        for env_id in uniq_candidates:
            try:
                env = self._try_make(env_id, frameskip=1, rep_prob=rep_prob)
                break
            except Exception as e:
                last_err = e
                env = None
                continue

        if env is None:
            # last fallback: try without frameskip kwarg (older API)
            for env_id in uniq_candidates:
                try:
                    env = self._try_make(env_id, frameskip=None, rep_prob=rep_prob)
                    break
                except Exception as e:
                    last_err = e
                    env = None
                    continue

        if env is None:
            raise RuntimeError(f"Could not create Atari env from '{game}'. Last error: {last_err}")

        # Seed (gymnasium vs gym)
        try:
            env.reset(seed=seed)
        except TypeError:
            # legacy gym
            if seed is not None:
                try:
                    env.seed(seed)
                except Exception:
                    pass

        # Also seed action_space if possible (useful for reproducibility)
        if seed is not None:
            try:
                env.action_space.seed(seed)
            except Exception:
                pass

        # Warn if the base env still uses internal frameskip != 1 (double frameskip risk)
        try:
            fs = getattr(env.unwrapped, "frameskip", None)
            if fs is not None and int(fs) != 1:
                warnings.warn(
                    f"[AtariAdapter] Base env frameskip={fs} (expected 1). "
                    "This can cause double frameskip if AtariPreprocessing also uses frame_skip. "
                    "Prefer ALE/*-v5 with frameskip=1 or *NoFrameskip-v4."
                )
        except Exception:
            pass

        return env

    def _require_84x84(self, env) -> None:
        """Fail fast if AtariPreprocessing did not produce 84x84-like output."""
        out = env.reset()
        o = out[0] if isinstance(out, tuple) else out
        o = np.asarray(o)

        ok = (
            (o.ndim == 2 and o.shape == (84, 84)) or
            (o.ndim == 3 and o.shape[:2] == (84, 84))  # (84,84,1) or (84,84,3)
        )
        if not ok:
            raise RuntimeError(
                f"AtariPreprocessing did not produce 84x84. Got obs shape={o.shape}. "
                "This usually means AtariPreprocessing failed (often missing opencv-python). "
                "Install opencv-python and gymnasium[atari] (and ale-py)."
            )

    def create_env(self, spec):
        p = spec.params or {}
        game = p.get("game")
        if not game:
            raise ValueError("AtariAdapter requires params['game'] (e.g., 'ALE/Pong-v5').")

        # Standard knobs
        frameskip = int(p.get("frameskip", 4))          # used only by AtariPreprocessing
        noop_max = int(p.get("noop_max", 30))
        sticky = bool(p.get("sticky_actions", True))
        rep_prob = float(p.get("repeat_action_prob", 0.25 if sticky else 0.0))

        grayscale = bool(p.get("grayscale_obs", True))
        scale_obs = bool(p.get("scale_obs", False))     # usually keep False, normalize in model/code
        term_on_life = bool(p.get("terminal_on_life_loss", True))
        clip_rewards = bool(p.get("clip_rewards", True))
        frame_stack = int(p.get("frame_stack", 4))

        # 0) Base env (try to ensure frameskip=1)
        env = self._make_game(game, spec.seed, rep_prob)

        # 1) AtariPreprocessing
        try:
            try:
                from gymnasium.wrappers import AtariPreprocessing as AP
            except Exception:
                AP = gym.wrappers.AtariPreprocessing  # type: ignore

            env = AP(
                env,
                noop_max=noop_max,
                frame_skip=frameskip,
                screen_size=84,
                grayscale_obs=grayscale,
                scale_obs=scale_obs,
                terminal_on_life_loss=term_on_life,
            )
        except Exception as e:
            raise RuntimeError(
                f"AtariPreprocessing failed: {e}. "
                "Install opencv-python and gymnasium[atari], and ensure base env uses frameskip=1."
            )

        # 2) Strict check (fail fast if preprocessing didn't work)
        self._require_84x84(env)

        # 3) Frame stacking
        if frame_stack and frame_stack > 1:
            fs_applied = False

            if GYM_IS_GYMNASIUM:
                # gymnasium >= 1.0 (or some versions)
                try:
                    from gymnasium.wrappers import FrameStackObservation
                    env = FrameStackObservation(env, stack_size=frame_stack)
                    fs_applied = True
                except Exception:
                    # gymnasium <= 0.29
                    try:
                        from gymnasium.wrappers import FrameStack as FS
                        env = FS(env, num_stack=frame_stack)
                        fs_applied = True
                    except Exception as e:
                        warnings.warn(f"[AtariAdapter] Frame stacking (gymnasium) failed: {e}")

            if not fs_applied:
                # old gym fallback
                try:
                    env = gym.wrappers.FrameStack(env, num_stack=frame_stack)  # type: ignore
                    fs_applied = True
                except Exception as e:
                    warnings.warn(f"[AtariAdapter] FrameStack failed: {e}")

        # 4) CHW (channel-first)
        env = ChannelFirstWrapper(env)

        # 5) Reward clipping
        if clip_rewards:
            try:
                env = gym.wrappers.TransformReward(env, lambda r: self._sign(float(r)))
            except Exception as e:
                warnings.warn(f"[AtariAdapter] TransformReward failed: {e}")

        return env

    def describe(self, env) -> dict:
        info: dict = {"env": str(getattr(env, "spec", None))}
        cur = env
        wrappers = []
        while hasattr(cur, "env"):
            wrappers.append(cur.__class__.__name__)
            cur = cur.env
        info["wrappers"] = wrappers

        try:
            out = env.reset()
            o = out[0] if isinstance(out, tuple) else out
            arr = np.asarray(o)
            info["obs_shape_sample"] = arr.shape
            info["obs_dtype_sample"] = str(arr.dtype)
            info["obs_minmax_sample"] = (float(arr.min()), float(arr.max()))
        except Exception:
            pass

        return info
