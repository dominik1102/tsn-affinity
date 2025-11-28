from __future__ import annotations
import warnings
import numpy as np
import ale_py

try:
    import gymnasium as gym
except Exception:
    import gym  # type: ignore

try:
    import gymnasium as gym
    from gymnasium.spaces import Box
    GYM_IS_GYMNASIUM = True
except Exception:
    import gym  # type: ignore
    from gym.spaces import Box  # type: ignore
    GYM_IS_GYMNASIUM = False


class ChannelFirstWrapper(gym.ObservationWrapper):
    """
    Ensure channel-first layout for image-like observations.

    Cases handled:
      - HWC       -> CHW
      - HW        -> (1, H, W)
      - (S, H, W, C) -> (S * C, H, W)  (e.g. frame-stacked RGB)

    If the observation is already channel-first or non-image (e.g. (C, H, W) or
    (S, H, W) for stacked grayscale), the observation space is left unchanged.
    """

    def __init__(self, env):
        super().__init__(env)
        old = env.observation_space

        if isinstance(old, Box):
            shape = old.shape

            # HWC image
            if len(shape) == 3 and shape[-1] in (1, 2, 3, 4, 12):
                H, W, C = shape
                low = np.min(old.low) if np.ndim(old.low) else old.low
                high = np.max(old.high) if np.ndim(old.high) else old.high
                self.observation_space = Box(
                    low=low, high=high, shape=(C, H, W), dtype=old.dtype
                )

            # HW -> (1, H, W)
            elif len(shape) == 2:
                H, W = shape
                low = np.min(old.low) if np.ndim(old.low) else old.low
                high = np.max(old.high) if np.ndim(old.high) else old.high
                self.observation_space = Box(
                    low=low, high=high, shape=(1, H, W), dtype=old.dtype
                )

            # FrameStacked RGB: (S, H, W, C) -> (S * C, H, W)
            elif len(shape) == 4 and shape[-1] in (1, 2, 3, 4, 12):
                S, H, W, C = shape
                low = np.min(old.low) if np.ndim(old.low) else old.low
                high = np.max(old.high) if np.ndim(old.high) else old.high
                self.observation_space = Box(
                    low=low, high=high, shape=(S * C, H, W), dtype=old.dtype
                )

            else:
                # Already channel-first or a non-image observation; keep as is
                self.observation_space = old
        else:
            self.observation_space = old

    def observation(self, obs):
        arr = np.asarray(obs)

        # HWC -> CHW
        if arr.ndim == 3 and arr.shape[-1] in (1, 2, 3, 4, 12):
            return np.transpose(arr, (2, 0, 1))

        # HW -> (1, H, W)
        if arr.ndim == 2:
            return arr[None, ...]

        # (S, H, W, C) -> (S * C, H, W)
        if arr.ndim == 4 and arr.shape[-1] in (1, 2, 3, 4, 12):
            S, H, W, C = arr.shape
            arr = np.transpose(arr, (0, 3, 1, 2))  # (S, C, H, W)
            return arr.reshape(S * C, H, W)

        # Anything else (e.g. already CHW / (S, H, W)) – leave unchanged
        return arr


class AtariAdapter:
    """Adapter for ALE with safe preprocessing and CHW-compatible output."""

    def _sign(self, r: float) -> float:
        return 1.0 if r > 0 else (-1.0 if r < 0 else 0.0)

    def _make_game(self, game: str, seed: int | None, repeat_action_prob: float):
        # Try sticky actions; fall back to different IDs (ALE/<game>-v5, NoFrameskip, etc.)
        try:
            env = gym.make(game, repeat_action_probability=repeat_action_prob)
        except Exception:
            alt = None
            if "/" not in game and "NoFrameskip" not in game:
                alt = f"ALE/{game}-v5"
            elif game.endswith("NoFrameskip-v4"):
                alt = game
            env = gym.make(alt or game)

        # Seed, handling both gym and gymnasium APIs
        try:
            env.reset(seed=seed)
        except TypeError:
            if seed is not None:
                env.seed(seed)
        return env

    def create_env(self, spec):
        p = spec.params or {}
        game = p.get("game")
        if not game:
            raise ValueError("AtariAdapter requires params['game'] (e.g., 'ALE/Pong-v5').")

        frameskip = int(p.get("frameskip", 4))
        noop_max = int(p.get("noop_max", 30))
        sticky = bool(p.get("sticky_actions", True))
        rep_prob = float(p.get("repeat_action_prob", 0.25 if sticky else 0.0))
        grayscale = bool(p.get("grayscale_obs", True))
        # usually False; you can normalize inside the model instead
        scale_obs = bool(p.get("scale_obs", False))
        term_on_life = bool(p.get("terminal_on_life_loss", True))
        clip_rewards = bool(p.get("clip_rewards", True))
        frame_stack = int(p.get("frame_stack", 4))

        env = self._make_game(game, spec.seed, rep_prob)

        # 1) Try official AtariPreprocessing (resizes to 84x84, grayscale, etc.)
        try:
            # Different import paths for gymnasium vs old gym
            try:
                from gymnasium.wrappers import AtariPreprocessing as AP
            except Exception:
                AP = gym.wrappers.AtariPreprocessing  # old gym

            env = AP(
                env,
                noop_max=noop_max,
                frame_skip=frameskip,
                screen_size=84,
                grayscale_obs=grayscale,
                # grayscale_newaxis left as default
                scale_obs=scale_obs,
                terminal_on_life_loss=term_on_life,
            )
        except Exception as e:
            # Most common reason: no opencv / no gymnasium[atari] installed
            warnings.warn(f"AtariPreprocessing unavailable/failed (OK, fallback): {e}")

        # 3) Frame stacking
        if frame_stack and frame_stack > 1:
            fs_applied = False

            if GYM_IS_GYMNASIUM:
                # New API in gymnasium >= 1.0: FrameStackObservation
                try:
                    from gymnasium.wrappers import FrameStackObservation

                    env = FrameStackObservation(env, stack_size=frame_stack)
                    fs_applied = True
                except Exception:
                    # Older gymnasium (<= 0.29) still has FrameStack
                    try:
                        from gymnasium.wrappers import FrameStack as FS

                        env = FS(env, num_stack=frame_stack)
                        fs_applied = True
                    except Exception as e:
                        warnings.warn(f"Frame stacking (gymnasium) failed: {e}")

            if not fs_applied:
                # Fallback to classic gym FrameStack
                try:
                    env = gym.wrappers.FrameStack(env, num_stack=frame_stack)
                    fs_applied = True
                except Exception as e:
                    warnings.warn(f"FrameStack failed: {e}")

        # 4) Channel-first (CHW) — our wrapper, without TransformObservation
        env = ChannelFirstWrapper(env)

        # 5) Reward clipping (optional)
        try:
            if clip_rewards:
                env = gym.wrappers.TransformReward(env, lambda r: self._sign(r))
        except Exception as e:
            warnings.warn(f"TransformReward failed: {e}")

        return env

    def describe(self, env) -> dict:
        info: dict = {"env": str(getattr(env, "spec", None))}
        cur = env
        wrappers = []
        while hasattr(cur, "env"):
            wrappers.append(cur.__class__.__name__)
            cur = cur.env
        info["wrappers"] = wrappers

        # Best-effort probe of observation shape
        try:
            o, _ = env.reset()
            info["obs_shape_sample"] = np.asarray(o).shape
        except Exception:
            pass

        return info
