# clbench/experts/atari_dqn.py
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Helper: Gym vs Gymnasium compatibility
# =============================================================================

def env_reset(env):
    """Reset environment and return only the observation (Gymnasium returns (obs, info))."""
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def env_step(env, action: int):
    """Step environment and return (obs, reward, done, info) with unified API."""
    out = env.step(int(action))
    # Gymnasium: (obs, reward, terminated, truncated, info)
    if isinstance(out, tuple) and len(out) == 5:
        next_obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
    else:
        # Legacy Gym: (obs, reward, done, info)
        next_obs, reward, done, info = out
        done = bool(done)
    return next_obs, float(reward), done, info


def reset_env_processed(env, fire_reset: bool = True) -> np.ndarray:
    """
    Reset env and return a PREPROCESSED observation (float32 CHW in [0,1]).
    Optionally presses FIRE once after reset for games that require it (e.g., Breakout).

    Safe no-op if the env doesn't support get_action_meanings().
    """
    obs = preprocess_obs(env_reset(env))

    if not fire_reset:
        return obs

    try:
        unwrapped = env.unwrapped
        if hasattr(unwrapped, "get_action_meanings"):
            meanings = unwrapped.get_action_meanings()
            if "FIRE" in meanings:
                fire_action = meanings.index("FIRE")
                obs2, _, done, _ = env_step(env, fire_action)
                obs = preprocess_obs(env_reset(env) if done else obs2)
    except Exception:
        # Never fail training because of FIRE helper.
        pass

    return obs


def preprocess_obs(obs: np.ndarray) -> np.ndarray:
    """
    Ensure Atari obs is:
      - np.float32
      - scaled to [0,1] if looks like pixels
      - CHW layout for PyTorch Conv2d
    Works with LazyFrames too (np.asarray).
    """
    x = np.asarray(obs)

    # if grayscale without channel: (H,W) -> (1,H,W)
    if x.ndim == 2:
        x = x[None, :, :]

    # convert dtype
    if x.dtype != np.float32:
        x = x.astype(np.float32)

    # normalize if pixel range
    # (Using numpy max: CPU-side and cheap compared to GPU sync)
    if x.size and x.max() > 1.0:
        x = x / 255.0

    # ensure CHW if HWC
    if x.ndim == 3:
        # if last dim looks like channels and first dim does NOT
        if x.shape[-1] in (1, 3, 4) and x.shape[0] not in (1, 3, 4):
            x = np.transpose(x, (2, 0, 1))

    return x


def obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert a preprocessed observation (float32 CHW in [0,1]) to [1,C,H,W] tensor.
    IMPORTANT: no `.max().item()` here (avoids GPU sync per step).
    """
    x = np.ascontiguousarray(obs)
    return torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)


# =============================================================================
# DQN network for Atari
# =============================================================================

class AtariQNetwork(nn.Module):
    """
    Simple convolutional Q-network for Atari-like inputs.

    Expects observations of shape (C, H, W), where C is number of stacked frames.
    """

    def __init__(self, obs_shape: Tuple[int, int, int], n_actions: int):
        super().__init__()
        c, h, w = obs_shape

        # Standard DQN-style conv encoder (similar to Nature DQN)
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        # Compute conv output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out = self.conv(dummy)
            conv_out_dim = conv_out.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W] float tensor in [0,1]
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# =============================================================================
# Replay buffer (IMPORTANT FIX: store images as uint8 to save RAM and speed up)
# =============================================================================

class ReplayBufferAtari:
    """
    Replay buffer for Atari images.
    Stores obs / next_obs as uint8 in [0,255] for memory efficiency,
    converts to float32 [0,1] on sampling.
    """

    def __init__(self, capacity: int, obs_shape: Tuple[int, int, int]):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity,) + obs_shape, dtype=np.uint8)
        self.next_obs = np.zeros((self.capacity,) + obs_shape, dtype=np.uint8)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.idx = 0
        self.full = False

    def size(self) -> int:
        return self.capacity if self.full else self.idx

    @staticmethod
    def _to_uint8(x: np.ndarray) -> np.ndarray:
        """
        Accepts either float32 [0,1] or uint8 [0,255] and returns uint8 [0,255].
        """
        x = np.asarray(x)
        if x.dtype == np.uint8:
            return x
        # assume float32 [0,1]
        x = np.clip(x * 255.0, 0.0, 255.0).astype(np.uint8)
        return x

    def add(self, obs, action, reward, next_obs, done):
        """Store one transition in the buffer."""
        self.obs[self.idx] = self._to_uint8(obs)
        self.next_obs[self.idx] = self._to_uint8(next_obs)
        self.actions[self.idx] = int(action)
        self.rewards[self.idx] = float(reward)
        self.dones[self.idx] = float(done)

        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(self, batch_size: int, device: torch.device):
        """
        Sample a batch of transitions and return tensors on the given device.
        """
        max_idx = self.size()
        idxs = np.random.randint(0, max_idx, size=int(batch_size))

        batch_obs = torch.from_numpy(self.obs[idxs]).to(device).float().div_(255.0)
        batch_next_obs = torch.from_numpy(self.next_obs[idxs]).to(device).float().div_(255.0)

        batch_actions = torch.from_numpy(self.actions[idxs]).to(device)
        batch_rewards = torch.from_numpy(self.rewards[idxs]).to(device)
        batch_dones = torch.from_numpy(self.dones[idxs]).to(device)

        return batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def eval_greedy(env, q_net, device, episodes=5, max_len=10_000, fire_reset: bool = True, return_hist: bool = True):
    q_net.eval()
    rets = []
    hist = Counter()

    for _ in range(int(episodes)):
        obs = reset_env_processed(env, fire_reset=fire_reset)
        ep_ret = 0.0

        for _t in range(int(max_len)):
            obs_t = obs_to_tensor(obs, device)
            action = int(q_net(obs_t).argmax(dim=1).item())
            hist[action] += 1

            next_obs_raw, r, done, _ = env_step(env, action)
            obs = preprocess_obs(next_obs_raw)
            ep_ret += float(r)
            if done:
                break

        rets.append(ep_ret)

    q_net.train()
    if return_hist:
        return float(np.mean(rets)), float(np.max(rets)), dict(hist)
    return float(np.mean(rets)), float(np.max(rets))


# =============================================================================
# Atari DQN expert training
# =============================================================================

def train_atari_dqn_expert(
    env,
    device: torch.device,
    total_steps: int = 500_000,
    batch_size: int = 32,
    gamma: float = 0.99,
    lr: float = 1e-4,
    buffer_capacity: int = 100_000,
    warmup_steps: int = 100_000,
    target_update_every: int = 10_000,
    eps_start: float = 1.0,
    eps_end: float = 0.01,
    eps_decay: int = 250_000,
    eval_env=None,
    eval_every: int = 50_000,
    resync_after_eval: bool = True,
    fire_reset: bool = True,
    reward_clip: bool = False,
    debug: bool = True,
):
    """
    Train a DQN expert on a single Atari task.

    Key fixes included:
      - Replay buffer stores uint8 frames (avoids huge RAM usage and slowdowns)
      - obs_to_tensor no longer uses x.max().item() (avoids GPU sync)
      - obs_shape taken from PREPROCESSED observation (prevents HWC/CHW mismatch)
      - epsilon schedule decays after warmup (more standard)
      - optional FIRE reset helper for games requiring it
    """
    # Use real preprocessed observation shape (important if env gives HWC)
    obs = reset_env_processed(env, fire_reset=fire_reset)
    obs_shape = obs.shape
    n_actions = env.action_space.n

    q_net = AtariQNetwork(obs_shape, n_actions).to(device)
    target_net = AtariQNetwork(obs_shape, n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr, eps=1e-4)
    replay = ReplayBufferAtari(buffer_capacity, obs_shape)

    if debug:
        print("[debug] obs shape:", obs.shape, "dtype:", obs.dtype, "min/max:", float(obs.min()), float(obs.max()))
        print("[debug] env obs_space:", getattr(env.observation_space, "shape", None), "actions:", n_actions)

    episode_return = 0.0
    all_returns = []
    best_return = -float("inf")

    for step in range(1, int(total_steps) + 1):
        # Epsilon schedule (decay starts after warmup)
        t = max(0, step - int(warmup_steps))
        eps = eps_end + (eps_start - eps_end) * math.exp(-t / float(eps_decay))

        if step == warmup_steps and debug:
            print(f"[DQN expert Atari] STARTING UPDATES at step={step}")

        # --- Greedy eval (epsilon=0) ---
        if eval_every and (step % int(eval_every) == 0):
            e_env = eval_env if eval_env is not None else env
            avg_eval, best_eval, hist = eval_greedy(
                e_env, q_net, device, episodes=5, fire_reset=fire_reset, return_hist=True
            )
            print(f"[eval greedy] step={step} avg={avg_eval:.1f} best={best_eval:.1f} hist={hist}")

            if (eval_env is None) and resync_after_eval:
                obs = reset_env_processed(env, fire_reset=fire_reset)
                episode_return = 0.0

        # Action selection
        if random.random() < eps:
            action = int(env.action_space.sample())
        else:
            with torch.no_grad():
                obs_t = obs_to_tensor(obs, device)
                action = int(q_net(obs_t).argmax(dim=1).item())

        # Environment step
        next_obs_raw, reward, done, _ = env_step(env, action)
        if reward_clip:
            reward = float(np.clip(reward, -1.0, 1.0))

        next_obs = preprocess_obs(next_obs_raw)

        # Store transition
        replay.add(obs, action, reward, next_obs, float(done))

        episode_return += reward
        obs = next_obs

        # Quick sanity check after warmup starts
        if step == warmup_steps and debug and replay.size() >= batch_size:
            b_obs, _, _, _, _ = replay.sample(batch_size, device)
            print("[debug] batch_obs:", tuple(b_obs.shape), "min/max:", float(b_obs.min()), float(b_obs.max()))

        # DQN updates after warmup
        if step >= warmup_steps and replay.size() >= batch_size:
            batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = replay.sample(batch_size, device)

            q_values = q_net(batch_obs).gather(1, batch_actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q_values = target_net(batch_next_obs).max(dim=1)[0]
                target_q = batch_rewards + gamma * (1.0 - batch_dones) * next_q_values

            loss = F.smooth_l1_loss(q_values, target_q)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            optimizer.step()

            if debug and (step % 20_000 == 0):
                print(
                    f"[debug] step={step} loss={float(loss.item()):.4f} "
                    f"q_mean={float(q_values.mean().item()):.3f} r_mean={float(batch_rewards.mean().item()):.3f} eps={eps:.3f}"
                )

        # Periodically update target network
        if step % int(target_update_every) == 0:
            target_net.load_state_dict(q_net.state_dict())

        if done:
            all_returns.append(episode_return)
            best_return = max(best_return, episode_return)

            if len(all_returns) % 10 == 0:
                avg_last10 = float(np.mean(all_returns[-10:]))
                print(
                    f"[DQN expert Atari] steps={step}, episodes={len(all_returns)}, "
                    f"avg_return(last10)={avg_last10:.1f}, best={best_return:.1f}, eps={eps:.3f}"
                )

            episode_return = 0.0
            obs = reset_env_processed(env, fire_reset=fire_reset)

    print(f"[DQN expert Atari] finished training, best_return={best_return:.1f}")
    return q_net


# =============================================================================
# Expert trajectory collection (with expert/random mixing)
# =============================================================================

def collect_atari_expert_trajectories(
    env,
    q_net: AtariQNetwork,
    device: torch.device,
    n_episodes: int,
    max_len: int,
    expert_action_prob: float = 1.0,
    fire_reset: bool = True,
) -> List[Dict[str, Any]]:
    """
    Collect trajectories using a mixture of expert and random actions.

    Per-step behavior:
      - with probability `expert_action_prob` we use the expert (greedy Q-network),
      - otherwise we sample a random action.

    Returns float32 observations in [0,1], CHW (compatible with your old pipeline).
    """
    expert_action_prob = float(np.clip(expert_action_prob, 0.0, 1.0))
    trajectories: List[Dict[str, Any]] = []

    for ep in range(int(n_episodes)):
        obs = reset_env_processed(env, fire_reset=fire_reset)

        obs_buf = []
        act_buf = []
        rew_buf = []
        done_buf = []

        ep_return = 0.0

        for _t in range(int(max_len)):
            obs_buf.append(obs.copy())  # float32 CHW [0,1]

            use_expert = (random.random() < expert_action_prob)

            if use_expert and q_net is not None:
                with torch.no_grad():
                    obs_t = obs_to_tensor(obs, device)
                    action = int(q_net(obs_t).argmax(dim=1).item())
            else:
                action = int(env.action_space.sample())

            act_buf.append(action)

            next_obs, reward, done, _ = env_step(env, action)
            rew_buf.append(float(reward))
            done_buf.append(bool(done))

            ep_return += reward
            obs = preprocess_obs(next_obs)

            if done:
                break

        traj = {
            "observations": np.stack(obs_buf, axis=0),       # [T, C, H, W] float32
            "actions": np.array(act_buf, dtype=np.int64),    # [T]
            "rewards": np.array(rew_buf, dtype=np.float32),
            "dones": np.array(done_buf, dtype=np.bool_),
        }
        trajectories.append(traj)

        print(
            f"[collect Atari expert-mix] episode {ep+1}/{n_episodes}, "
            f"return={ep_return:.1f}, T={len(obs_buf)}, expert_action_prob={expert_action_prob:.2f}"
        )

    return trajectories


# =============================================================================
# Saving trajectories to disk (.npz)
# =============================================================================

def save_trajectories_npz(trajectories: List[Dict[str, Any]], out_path: str) -> None:
    """
    Save multiple episodes into a single .npz file.

    Stored keys:
      - observations:    [N, C, H, W]
      - actions:         [N]
      - rewards:         [N]
      - dones:           [N]
      - episode_lengths: [n_episodes]
    """
    obs_list = [tr["observations"] for tr in trajectories]
    act_list = [tr["actions"] for tr in trajectories]
    rew_list = [tr["rewards"] for tr in trajectories]
    done_list = [tr["dones"] for tr in trajectories]

    observations = np.concatenate(obs_list, axis=0)
    actions = np.concatenate(act_list, axis=0)
    rewards = np.concatenate(rew_list, axis=0)
    dones = np.concatenate(done_list, axis=0)
    episode_lengths = np.array([len(tr["observations"]) for tr in trajectories], dtype=np.int32)

    np.savez_compressed(
        out_path,
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
        episode_lengths=episode_lengths,
    )
    print(f"[save] saved {len(trajectories)} episodes to {out_path}")
