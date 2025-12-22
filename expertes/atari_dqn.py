# clbench/experts/atari_dqn.py
from __future__ import annotations

import os
import math
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

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

    # normalize if pixel range (CPU-side and cheap compared to GPU sync)
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


def _state_dict_cpu(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Return a CPU-cloned state_dict (safe to keep as 'best' without holding GPU memory)."""
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


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
# Replay buffer (stores images as uint8 to save RAM and speed up)
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
        """Accepts either float32 [0,1] or uint8 [0,255] and returns uint8 [0,255]."""
        x = np.asarray(x)
        if x.dtype == np.uint8:
            return x
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
        """Sample a batch of transitions and return tensors on the given device."""
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
def eval_greedy(
    env,
    q_net: nn.Module,
    device: torch.device,
    episodes: int = 5,
    max_len: int = 10_000,
    fire_reset: bool = True,
    return_hist: bool = True,
):
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
# Atari DQN expert training (MASTER: keeps/saves BEST + Double DQN)
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
    # -------- MASTER additions --------
    save_best_path: Optional[str] = None,
    save_last_path: Optional[str] = None,
    best_on: str = "eval_avg",          # "eval_avg" | "eval_best" | "train_episode" | "auto"
    restore_best_at_end: bool = True,
    eval_episodes: int = 20,            # more stable than 5 for selecting "master"
    eval_max_len: int = 20_000,         # allow full games in eval (especially Pong)
    stop_when_eval_avg_ge: Optional[float] = None,  # early stop criterion (optional)
):
    """
    Train a DQN expert on a single Atari task.

    MASTER behavior:
      - Tracks BEST weights (CPU snapshot) and can save them to disk
      - Can restore BEST weights before returning, so trajectory collection uses BEST
      - Uses Double DQN target (usually more stable and better than vanilla DQN)
      - Eval defaults are more stable (eval_episodes=20, longer eval_max_len)
    """

    def _ensure_parent_dir(path: str) -> None:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

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

    # -------- BEST tracking --------
    if best_on == "auto":
        best_on_eff = "eval_avg" if (eval_every and int(eval_every) > 0) else "train_episode"
    else:
        best_on_eff = best_on

    if best_on_eff in ("eval_avg", "eval_best") and not (eval_every and int(eval_every) > 0):
        if debug:
            print(f"[warn] best_on='{best_on_eff}' but eval_every=0 -> falling back to best_on='train_episode'")
        best_on_eff = "train_episode"

    best_score = -float("inf")
    best_step = -1
    best_state_dict_cpu: Optional[Dict[str, torch.Tensor]] = None

    best_train_episode_return = -float("inf")

    def _maybe_update_best(score: float, step_i: int, reason: str) -> None:
        nonlocal best_score, best_step, best_state_dict_cpu
        if score > best_score + 1e-8:
            best_score = float(score)
            best_step = int(step_i)
            best_state_dict_cpu = _state_dict_cpu(q_net)

            if save_best_path:
                _ensure_parent_dir(save_best_path)
                torch.save(
                    {
                        "kind": "best",
                        "best_on": best_on_eff,
                        "score": best_score,
                        "step": best_step,
                        "obs_shape": tuple(obs_shape),
                        "n_actions": int(n_actions),
                        "state_dict": best_state_dict_cpu,
                    },
                    save_best_path,
                )
                print(
                    f"[checkpoint] saved BEST ({best_on_eff}) score={best_score:.2f} "
                    f"step={best_step} reason={reason} -> {save_best_path}"
                )
            else:
                print(f"[checkpoint] updated BEST ({best_on_eff}) score={best_score:.2f} step={best_step} reason={reason}")

    episode_return = 0.0
    all_returns: List[float] = []

    last_step = 0

    for step in range(1, int(total_steps) + 1):
        last_step = step

        # Epsilon schedule (decay starts after warmup)
        t = max(0, step - int(warmup_steps))
        eps = eps_end + (eps_start - eps_end) * math.exp(-t / float(eps_decay))

        if step == warmup_steps and debug:
            print(f"[DQN expert Atari] STARTING UPDATES at step={step}")

        # --- Greedy eval (epsilon=0) ---
        if eval_every and (step % int(eval_every) == 0):
            e_env = eval_env if eval_env is not None else env
            avg_eval, best_eval, hist = eval_greedy(
                e_env,
                q_net,
                device,
                episodes=int(eval_episodes),
                max_len=int(eval_max_len),
                fire_reset=fire_reset,
                return_hist=True,
            )
            print(f"[eval greedy] step={step} avg={avg_eval:.2f} best={best_eval:.2f} hist={hist}")

            if best_on_eff == "eval_avg":
                _maybe_update_best(avg_eval, step, reason="eval_avg")
            elif best_on_eff == "eval_best":
                _maybe_update_best(best_eval, step, reason="eval_best")

            if stop_when_eval_avg_ge is not None and avg_eval >= float(stop_when_eval_avg_ge):
                print(f"[stop] reached avg_eval={avg_eval:.2f} >= {float(stop_when_eval_avg_ge):.2f} at step={step}")
                break

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

            # Double DQN target:
            # - action selection by online net (q_net)
            # - action evaluation by target net (target_net)
            with torch.no_grad():
                next_actions = q_net(batch_next_obs).argmax(dim=1)
                next_q_values = target_net(batch_next_obs).gather(1, next_actions.unsqueeze(1)).squeeze(1)
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

        # Episode ended
        if done:
            all_returns.append(float(episode_return))
            best_train_episode_return = max(best_train_episode_return, float(episode_return))

            if best_on_eff == "train_episode":
                _maybe_update_best(float(episode_return), step, reason="train_episode")

            if len(all_returns) % 10 == 0:
                avg_last10 = float(np.mean(all_returns[-10:]))
                print(
                    f"[DQN expert Atari] steps={step}, episodes={len(all_returns)}, "
                    f"avg_return(last10)={avg_last10:.2f}, best_train_ep={best_train_episode_return:.2f}, eps={eps:.3f}"
                )

            episode_return = 0.0
            obs = reset_env_processed(env, fire_reset=fire_reset)

    # Save LAST checkpoint (optional)
    if save_last_path:
        _ensure_parent_dir(save_last_path)
        torch.save(
            {
                "kind": "last",
                "step": int(last_step),
                "obs_shape": tuple(obs_shape),
                "n_actions": int(n_actions),
                "state_dict": _state_dict_cpu(q_net),
            },
            save_last_path,
        )
        print(f"[checkpoint] saved LAST -> {save_last_path}")

    # Restore BEST into q_net before returning (recommended for master trajectory collection)
    if restore_best_at_end and best_state_dict_cpu is not None:
        q_net.load_state_dict(best_state_dict_cpu)
        target_net.load_state_dict(q_net.state_dict())

    print(
        f"[DQN expert Atari] finished training. "
        f"best_train_ep={best_train_episode_return:.2f}. "
        f"best({best_on_eff})={best_score:.2f} at step={best_step}"
    )

    # Attach info for downstream saving / debugging (optional)
    try:
        q_net.best_on = best_on_eff
        q_net.best_score = float(best_score)
        q_net.best_step = int(best_step)
        q_net.best_state_dict_cpu = best_state_dict_cpu
    except Exception:
        pass

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

    Observations are stored as float32 CHW in [0,1] (as produced by preprocess_obs()).

    IMPORTANT NOTE:
      If many episodes hit `max_len` without done=True, your returns are truncated.
      For Pong "master" datasets, increase `max_len` substantially (e.g., 20000+).
    """
    expert_action_prob = float(np.clip(expert_action_prob, 0.0, 1.0))
    trajectories: List[Dict[str, Any]] = []

    hit_limit_count = 0

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

        # Count truncations (hit max_len without done=True)
        hit_limit = (len(obs_buf) >= int(max_len)) and (len(done_buf) == 0 or done_buf[-1] is False)
        if hit_limit:
            hit_limit_count += 1

        traj = {
            "observations": np.stack(obs_buf, axis=0),       # [T, C, H, W] float32
            "actions": np.array(act_buf, dtype=np.int64),    # [T]
            "rewards": np.array(rew_buf, dtype=np.float32),  # [T]
            "dones": np.array(done_buf, dtype=np.bool_),     # [T]
        }
        trajectories.append(traj)

        print(
            f"[collect Atari expert-mix] episode {ep+1}/{n_episodes}, "
            f"return={ep_return:.2f}, T={len(obs_buf)}, expert_action_prob={expert_action_prob:.2f}"
        )

    if n_episodes > 0:
        frac = hit_limit_count / float(n_episodes)
        print(f"[collect summary] hit_max_len_without_done: {hit_limit_count}/{n_episodes} ({frac*100:.1f}%)")
        if frac > 0.2:
            print(
                "[warn] Many episodes hit max_len without done=True. Returns are truncated. "
                "Increase --max-len (e.g. 20000/50000) or check if the env has a strict TimeLimit wrapper."
            )

    return trajectories


# =============================================================================
# Saving trajectories to disk (.npz)
# =============================================================================

def save_trajectories_npz(trajectories: List[Dict[str, Any]], out_path: str) -> None:
    """
    Save multiple episodes into a single .npz file.

    IMPORTANT: This uses a concatenated representation:
      - observations/actions/rewards/dones are concatenated over time across episodes
      - episode_lengths stores the per-episode T so you can reconstruct boundaries

    Stored keys:
      - observations:     [sum_T, C, H, W]
      - actions:          [sum_T]
      - rewards:          [sum_T]
      - dones:            [sum_T]
      - episode_lengths:  [n_episodes]
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
