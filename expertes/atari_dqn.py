# clbench/experts/atari_dqn.py
from __future__ import annotations
import math
import random
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==== Helper: Gym vs Gymnasium compatibility ====

def preprocess_obs(obs: np.ndarray) -> np.ndarray:
    """
    Ensure Atari obs is:
      - np.float32
      - scaled to [0,1] if looks like pixels
      - CHW layout for PyTorch Conv2d
    Works with LazyFrames too (np.asarray).
    """
    x = np.asarray(obs)

    # convert dtype
    if x.dtype != np.float32:
        x = x.astype(np.float32)

    # normalize if pixel range
    if x.max() > 1.0:
        x = x / 255.0

    # ensure CHW if HWC
    if x.ndim == 3:
        # if last dim looks like channels and first dim does NOT
        if x.shape[-1] in (1, 3, 4) and x.shape[0] not in (1, 3, 4):
            x = np.transpose(x, (2, 0, 1))

    return x

def obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # [1,C,H,W]
    # normalize if uint8-like
    if x.max().item() > 1.0:
        x = x / 255.0
    return x


def env_reset(env):
    """Reset environment and return only the observation (handles Gym vs Gymnasium)."""
    out = env.reset()
    # Gymnasium: (obs, info), legacy Gym: obs
    if isinstance(out, tuple):
        return out[0]
    return out


def env_step(env, action):
    """Step environment and return (obs, reward, done, info) with unified API."""
    out = env.step(action)
    # Gymnasium: (obs, reward, terminated, truncated, info)
    if isinstance(out, tuple) and len(out) == 5:
        next_obs, reward, terminated, truncated, info = out
        done = terminated or truncated
    else:
        # Legacy Gym: (obs, reward, done, info)
        next_obs, reward, done, info = out
    return next_obs, reward, done, info


# ==== DQN network for Atari ====


class AtariQNetwork(nn.Module):
    """
    Simple convolutional Q-network for Atari-like inputs.

    Expects observations of shape (C, H, W), where C is number of stacked frames.
    """

    def __init__(self, obs_shape, n_actions: int):
        super().__init__()
        c, h, w = obs_shape

        # Standard DQN-style conv encoder (similar to Nature DQN)
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Compute conv output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out = self.conv(dummy)
            conv_out_dim = conv_out.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W] float tensor (range can be [0,1] or [0,255]).
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ReplayBufferAtari:
    """Replay buffer for Atari images."""

    def __init__(self, capacity: int, obs_shape):
        self.capacity = capacity
        self.obs = np.zeros((capacity,) + obs_shape, dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity,) + obs_shape, dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.idx = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        """Store one transition in the buffer."""
        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_obs[self.idx] = next_obs
        self.dones[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def size(self) -> int:
        """Current number of valid transitions in the buffer."""
        return self.capacity if self.full else self.idx

    def sample(self, batch_size: int, device: torch.device):
        """
        Sample a batch of transitions and return tensors on the given device.
        """
        max_idx = self.size()
        idxs = np.random.randint(0, max_idx, size=batch_size)

        batch_obs = torch.from_numpy(self.obs[idxs]).to(device)
        batch_actions = torch.from_numpy(self.actions[idxs]).to(device)
        batch_rewards = torch.from_numpy(self.rewards[idxs]).to(device)
        batch_next_obs = torch.from_numpy(self.next_obs[idxs]).to(device)
        batch_dones = torch.from_numpy(self.dones[idxs]).to(device)

        return batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones


def eval_greedy(env, q_net, device, episodes=5, max_len=10_000):
    """
    Greedy evaluation (epsilon=0) on a given env.

    IMPORTANT:
    - Uses the same preprocess_obs() and env_reset/env_step wrappers as training,
      so the network sees the same input distribution.
    """
    q_net.eval()
    rets = []

    for _ in range(episodes):
        obs = preprocess_obs(env_reset(env))
        ep_ret = 0.0

        for _t in range(max_len):
            with torch.no_grad():
                obs_t = obs_to_tensor(obs, device)  # [1,C,H,W] float32
                a = int(q_net(obs_t).argmax(dim=1).item())

            next_obs_raw, r, done, _ = env_step(env, a)
            obs = preprocess_obs(next_obs_raw)

            ep_ret += float(r)
            if done:
                break

        rets.append(ep_ret)

    q_net.train()
    return float(np.mean(rets)), float(np.max(rets))



# ==== Atari DQN expert training ====


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
    eval_env=None,                 # NEW: separate env for evaluation (recommended)
    eval_every: int = 50_000,      # NEW
    resync_after_eval: bool = True # NEW: if eval_env is None, resync training env state
):

    """
    Train a DQN expert on a single Atari task.

    Args:
        env: Atari environment (from AtariAdapter / make_tasks).
        device: torch device.
        total_steps: total environment steps for training.
        batch_size: mini-batch size.
        gamma: discount factor.
        lr: learning rate for Adam.
        buffer_capacity: replay buffer capacity.
        warmup_steps: number of steps with replay filling before updates start.
        target_update_every: frequency of target network updates.
        eps_start/eps_end/eps_decay: epsilon-greedy exploration schedule.

    Returns:
        A trained AtariQNetwork instance.
    """
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    q_net = AtariQNetwork(obs_shape, n_actions).to(device)
    target_net = AtariQNetwork(obs_shape, n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    replay = ReplayBufferAtari(buffer_capacity, obs_shape)

    obs = preprocess_obs(env_reset(env))
    print("[debug] obs shape:", obs.shape, "dtype:", obs.dtype, "min/max:", float(obs.min()), float(obs.max()))
    print("[debug] env obs_space:", env.observation_space.shape, "actions:", env.action_space.n)

    episode_return = 0.0
    all_returns = []
    best_return = -float("inf")

    for step in range(1, total_steps + 1):
        # Epsilon-greedy exploration with exponential decay
        eps = eps_end + (eps_start - eps_end) * math.exp(-step / eps_decay)
        if step == warmup_steps:
            print(f"[DQN expert Atari] STARTING UPDATES at step={step}")

        # --- Greedy eval (epsilon=0) ---
        if eval_every and (step % eval_every == 0):
            e_env = eval_env if eval_env is not None else env
            avg_eval, best_eval = eval_greedy(e_env, q_net, device, episodes=5)
            print(f"[eval greedy] step={step} avg={avg_eval:.1f} best={best_eval:.1f}")

            # CRITICAL:
            # If we evaluated on the SAME env instance, eval_greedy() reset/stepped it.
            # That would desync 'obs' vs actual env state and corrupt replay transitions.
            # Resync after eval to keep training consistent.
            if (eval_env is None) and resync_after_eval:
                obs = preprocess_obs(env_reset(env))
                episode_return = 0.0

        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = obs_to_tensor(obs, device)
                q_values = q_net(obs_t)
                action = int(q_values.argmax(dim=1).item())

        next_obs_raw, reward, done, _ = env_step(env, action)
        next_obs = preprocess_obs(next_obs_raw)

        replay.add(obs, action, reward, next_obs, float(done))

        if step == warmup_steps:
            b_obs, b_act, b_rew, b_next, b_done = replay.sample(batch_size, device)
            print("[debug] batch_obs:", tuple(b_obs.shape), "min/max:", float(b_obs.min()), float(b_obs.max()))

        episode_return += reward
        obs = next_obs

        # DQN updates after warmup
        if step >= warmup_steps and replay.size() >= batch_size:
            (
                batch_obs,
                batch_actions,
                batch_rewards,
                batch_next_obs,
                batch_dones,
            ) = replay.sample(batch_size, device)

            q_values = q_net(batch_obs).gather(1, batch_actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q_values = target_net(batch_next_obs).max(dim=1)[0]
                target_q = batch_rewards + gamma * (1.0 - batch_dones) * next_q_values

            loss = F.mse_loss(q_values, target_q)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            optimizer.step()

        # Periodically update target network
        if step % target_update_every == 0:
            target_net.load_state_dict(q_net.state_dict())

        if done:
            all_returns.append(episode_return)
            best_return = max(best_return, episode_return)
            if len(all_returns) % 10 == 0:
                avg_last10 = np.mean(all_returns[-10:])
                print(
                    f"[DQN expert Atari] steps={step}, episodes={len(all_returns)}, "
                    f"avg_return(last10)={avg_last10:.1f}, best={best_return:.1f}, eps={eps:.3f}"
                )
            episode_return = 0.0
            obs = preprocess_obs(env_reset(env))

    print(f"[DQN expert Atari] finished training, best_return={best_return:.1f}")
    return q_net


# ==== Expert trajectory collection (with expert/random mixing) ====


def collect_atari_expert_trajectories(
    env,
    q_net: AtariQNetwork,
    device: torch.device,
    n_episodes: int,
    max_len: int,
    expert_action_prob: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Collect trajectories using a mixture of expert and random actions.

    Per-step behavior:
        - with probability `expert_action_prob` we use the expert (greedy Q-network),
        - otherwise we sample a random action from env.action_space.

    Args:
        env: Atari environment.
        q_net: trained AtariQNetwork expert.
        device: torch device.
        n_episodes: number of episodes to collect.
        max_len: maximum episode length.
        expert_action_prob: probability in [0, 1] of using expert action at each step.
                            1.0 -> pure expert, 0.0 -> pure random, 0.5 -> ~50/50 mix.

    Returns:
        List of episode dicts with keys:
          - "observations": [T, C, H, W]
          - "actions":      [T]
          - "rewards":      [T]
          - "dones":        [T]
    """
    expert_action_prob = max(0.0, min(1.0, float(expert_action_prob)))
    trajectories: List[Dict[str, Any]] = []

    for ep in range(n_episodes):
        obs = preprocess_obs(env_reset(env))
        obs_buf = []
        act_buf = []
        rew_buf = []
        done_buf = []

        ep_return = 0.0

        for t in range(max_len):
            obs_buf.append(obs.copy())  # already float32 CHW [0,1]

            use_expert = (random.random() < expert_action_prob)

            if use_expert and q_net is not None:
                with torch.no_grad():
                    obs_t = obs_to_tensor(obs, device)
                    q_values = q_net(obs_t)
                    action = int(q_values.argmax(dim=1).item())
            else:
                action = env.action_space.sample()

            act_buf.append(action)

            next_obs, reward, done, _ = env_step(env, action)
            rew_buf.append(float(reward))
            done_buf.append(bool(done))

            ep_return += reward
            obs = preprocess_obs(next_obs)

            if done:
                break

        traj = {
            "observations": np.stack(obs_buf, axis=0),     # [T, C, H, W]
            "actions": np.array(act_buf, dtype=np.int64),  # [T]
            "rewards": np.array(rew_buf, dtype=np.float32),
            "dones": np.array(done_buf, dtype=np.bool_),
        }
        trajectories.append(traj)

        print(
            f"[collect Atari expert-mix] episode {ep+1}/{n_episodes}, "
            f"return={ep_return:.1f}, T={len(obs_buf)}, "
            f"expert_action_prob={expert_action_prob:.2f}"
        )

    return trajectories


# ==== Saving trajectories to disk (.npz) ====


def save_trajectories_npz(trajectories: List[Dict[str, Any]], out_path: str) -> None:
    """
    Save multiple episodes into a single .npz file.

    Stored keys:
        - observations:    [N, C, H, W]
        - actions:         [N]
        - rewards:         [N]
        - dones:           [N]
        - episode_lengths: [n_episodes]

    You can later reconstruct episode boundaries from episode_lengths.
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
