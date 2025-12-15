# clbench/experts/cartpole_dqn.py
from __future__ import annotations
import math
import random
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==== Helper: Gym vs Gymnasium compatibility ====


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


# ==== Q-network and replay buffer ====


class QNetwork(nn.Module):
    """Simple MLP-based Q-network for CartPole."""

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Basic replay buffer for off-policy DQN training."""

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
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


# ==== DQN expert training ====


def train_cartpole_dqn_expert(
    env,
    device: torch.device,
    total_steps: int = 50000,
    batch_size: int = 64,
    gamma: float = 0.99,
    lr: float = 1e-3,
    buffer_capacity: int = 100000,
    warmup_steps: int = 1000,
    target_update_every: int = 1000,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay: int = 25000,
) -> QNetwork:
    """
    Train a simple DQN expert on a single CartPole task.

    Returns:
        A trained QNetwork instance.
    """
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_net = QNetwork(obs_dim, n_actions).to(device)
    target_net = QNetwork(obs_dim, n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    replay = ReplayBuffer(buffer_capacity, obs_dim)

    obs = env_reset(env)
    episode_return = 0.0
    all_returns = []
    best_return = -float("inf")

    for step in range(1, total_steps + 1):
        # Epsilon-greedy exploration with exponential decay
        eps = eps_end + (eps_start - eps_end) * math.exp(-step / eps_decay)
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = q_net(obs_t)
                action = int(q_values.argmax(dim=1).item())

        next_obs, reward, done, _ = env_step(env, action)
        replay.add(obs, action, reward, next_obs, float(done))

        episode_return += reward
        obs = next_obs

        # DQN update
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
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
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
                    f"[DQN expert] steps={step}, episodes={len(all_returns)}, "
                    f"avg_return(last10)={avg_last10:.1f}, best={best_return:.1f}"
                )
            episode_return = 0.0
            obs = env_reset(env)

    print(f"[DQN expert] finished training, best_return={best_return:.1f}")
    return q_net


# ==== Expert trajectory collection ====


def collect_expert_trajectories(
    env,
    q_net: QNetwork,
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
        env: CartPole-like environment.
        q_net: trained QNetwork expert.
        device: torch device.
        n_episodes: number of episodes to collect.
        max_len: maximum episode length.
        expert_action_prob: probability in [0, 1] of using expert action at each step.
                            1.0 -> pure expert, 0.0 -> pure random, 0.5 -> ~50/50 mix.

    Returns:
        List of episode dicts with keys:
          - "observations": [T, obs_dim]
          - "actions":      [T]
          - "rewards":      [T]
          - "dones":        [T]
    """
    # clamp probability to sane range
    expert_action_prob = max(0.0, min(1.0, float(expert_action_prob)))

    trajectories: List[Dict[str, Any]] = []

    obs_dim = env.observation_space.shape[0]

    for ep in range(n_episodes):
        obs = env_reset(env)
        obs_buf = []
        act_buf = []
        rew_buf = []
        done_buf = []

        ep_return = 0.0

        for t in range(max_len):
            obs_buf.append(np.array(obs, dtype=np.float32))

            # decide whether to use expert or random at this step
            use_expert = (random.random() < expert_action_prob)

            if use_expert and q_net is not None:
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    q_values = q_net(obs_t)
                    action = int(q_values.argmax(dim=1).item())
            else:
                action = env.action_space.sample()

            act_buf.append(action)

            next_obs, reward, done, _ = env_step(env, action)
            rew_buf.append(float(reward))
            done_buf.append(bool(done))

            ep_return += reward
            obs = next_obs

            if done:
                break

        traj = {
            "observations": np.stack(obs_buf, axis=0),     # [T, obs_dim]
            "actions": np.array(act_buf, dtype=np.int64),  # [T]
            "rewards": np.array(rew_buf, dtype=np.float32),
            "dones": np.array(done_buf, dtype=np.bool_),
        }
        trajectories.append(traj)

        print(
            f"[collect expert-mix] episode {ep+1}/{n_episodes}, "
            f"return={ep_return:.1f}, T={len(obs_buf)}, "
            f"expert_action_prob={expert_action_prob:.2f}"
        )

    return trajectories


# ==== Disk saving (simple .npz schema) ====


def save_trajectories_npz(trajectories: List[Dict[str, Any]], out_path: str) -> None:
    """
    Save multiple episodes into a single .npz file.

    Stored keys:
        - observations:    [N, obs_dim]
        - actions:         [N]
        - rewards:         [N]
        - dones:           [N]
        - episode_lengths: [n_episodes]

    You can later reconstruct episode boundaries from episode_lengths.
    """
    # Concatenate along time dimension
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
