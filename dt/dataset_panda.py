from __future__ import annotations

from typing import Any, Dict, List, Iterator, Optional, Sequence, Tuple
import pickle
import numpy as np
import torch

from dt.dataset import Trajectory, discount_cumsum


# -----------------------------------------------------------------------------
# Obs "extractor" (as in the working pipeline): flatten according to observation_space keys
# -----------------------------------------------------------------------------
_DEFAULT_PANDA_KEYS: Tuple[str, ...] = ("observation", "achieved_goal", "desired_goal")


def _resolve_obs_keys_from_dict(obs_dict: Dict[str, Any], obs_keys: Optional[Sequence[str]]) -> Tuple[str, ...]:
    """
    - if obs_keys provided -> use them
    - otherwise:
      - if the dict has the standard 3 keys -> use _DEFAULT_PANDA_KEYS
      - else use the dict insertion order
    """
    if obs_keys is not None:
        return tuple(obs_keys)

    if all(k in obs_dict for k in _DEFAULT_PANDA_KEYS):
        return _DEFAULT_PANDA_KEYS

    # fallback: zachowaj kolejność insertion-order
    return tuple(obs_dict.keys())


def flatten_panda_step(obs: Dict[str, np.ndarray], obs_keys: Optional[Sequence[str]] = None) -> np.ndarray:
    """
    Flatten single Panda obs dict into 1D vector using obs_keys order.

    This is the equivalent of an "extractor" based on observation_space.
    """
    keys = _resolve_obs_keys_from_dict(obs, obs_keys)
    parts: List[np.ndarray] = []
    for k in keys:
        if k not in obs:
            raise KeyError(f"Missing key {k!r} in obs dict. Available keys: {list(obs.keys())}")
        parts.append(np.asarray(obs[k], dtype=np.float32).ravel())
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


def _slice_obs(obs_all: Any, start: int, end: int) -> Any:
    """
    Slice obs_all for 2 formats:
      - dict-of-arrays: {key: np.ndarray[N,...]}
      - array/list: np.ndarray[N,...] or list
    """
    if isinstance(obs_all, dict):
        return {k: np.asarray(v[start:end]) for k, v in obs_all.items()}
    return obs_all[start:end]


def _parse_episode_ends(episode_ends: Any, N: int) -> np.ndarray:
    """
    Support different episode_ends formats:
      - boolean mask of length N (True at episode ends)
      - 0/1 mask of length N
      - end indices: inclusive (max==N-1) or exclusive (max==N)
    Returns increasing EXCLUSIVE indices (i.e. end = t+1).
    """
    ends_raw = np.asarray(episode_ends)

    # bool mask -> indeksy
    if ends_raw.dtype == np.bool_:
        ends = np.nonzero(ends_raw)[0] + 1
    else:
        ends_raw = ends_raw.astype(np.int64).reshape(-1)

        # 0/1 mask o długości N -> indeksy
        if ends_raw.size == N and ends_raw.max(initial=0) <= 1:
            ends = np.nonzero(ends_raw.astype(bool))[0] + 1
        else:
            ends = ends_raw

        # inclusive -> exclusive
        if ends.size > 0 and ends.max(initial=0) == (N - 1):
            ends = ends + 1

    ends = np.asarray(ends, dtype=np.int64)
    ends = ends[(ends > 0) & (ends <= N)]

    if ends.size == 0:
        return np.array([N], dtype=np.int64)

    ends = np.unique(ends)
    ends.sort()

    if ends[-1] != N:
        ends = np.append(ends, N)

    return ends


def _make_trajectory(
    obs_seq: Any,
    actions_seq: Any,
    rewards_seq: Any,
    gamma: float = 1.0,
    obs_keys: Optional[Sequence[str]] = None,
) -> Trajectory:
    """
    Build Trajectory for Panda with continuous actions.

    obs_seq:
      - dict of arrays: keys e.g. "observation", "desired_goal", "achieved_goal",
        each [T, dim], OR
      - list of dicts / list of vectors [T, ...], OR
      - array [T, obs_dim] already flattened.
    """
    # ---- OBS ----
    if isinstance(obs_seq, dict):
        keys = _resolve_obs_keys_from_dict(obs_seq, obs_keys)
        for k in keys:
            if k not in obs_seq:
                raise KeyError(f"Missing key {k!r} in obs_seq dict. Available: {list(obs_seq.keys())}")

        parts = [np.asarray(obs_seq[k], dtype=np.float32) for k in keys]
        obs = np.concatenate(parts, axis=-1).astype(np.float32, copy=False)

    else:
        # list of dicts / vectors
        if (
            isinstance(obs_seq, (list, tuple))
            and len(obs_seq) > 0
            and isinstance(obs_seq[0], dict)
        ):
            obs = np.stack([flatten_panda_step(o, obs_keys=obs_keys) for o in obs_seq], axis=0).astype(np.float32)
        else:
            obs = np.asarray(obs_seq, dtype=np.float32)

    # ---- ACTIONS / REWARDS ----
    actions = np.asarray(actions_seq, dtype=np.float32)
    rewards = np.asarray(rewards_seq, dtype=np.float32).reshape(-1)

    T = int(rewards.shape[0])

    # adjust lengths (sometimes the dataset has obs length T+1)
    if obs.shape[0] == T + 1:
        obs = obs[:-1]
    if actions.shape[0] == T + 1:
        actions = actions[:-1]

    if obs.shape[0] != T or actions.shape[0] != T:
        raise ValueError(
            f"Length mismatch in Panda trajectory: obs={obs.shape[0]}, actions={actions.shape[0]}, rewards={T}"
        )

    timesteps = np.arange(T, dtype=np.int64)
    returns_to_go = discount_cumsum(rewards, gamma=float(gamma))

    return Trajectory(
        obs=obs,
        actions=actions,
        rewards=rewards,
        timesteps=timesteps,
        returns_to_go=returns_to_go,
    )


def load_panda_offline_pkl(
    path: str,
    gamma: float = 1.0,
    obs_keys: Optional[Sequence[str]] = None,
) -> List[Trajectory]:
    """
    Load panda-gym(-offline) dataset and convert to list[Trajectory].

    Notes:
      - obs_keys allows matching flatten order to env.observation_space (extractor-style)
      - episode_ends can be a mask or a list of indices (we support both)
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    trajs: List[Trajectory] = []

    # Case 1: list of episodes
    if isinstance(data, list):
        for ep in data:
            obs_seq = ep["observations"]
            actions_seq = ep["actions"]
            rewards_seq = ep["rewards"]
            trajs.append(_make_trajectory(obs_seq, actions_seq, rewards_seq, gamma=float(gamma), obs_keys=obs_keys))
        return trajs

    # Case 2: dict of arrays
    if isinstance(data, dict) and "observations" in data:
        obs_all = data["observations"]
        actions_all = np.asarray(data["actions"], dtype=np.float32)
        rewards_all = np.asarray(data["rewards"], dtype=np.float32).reshape(-1)

        N = int(len(rewards_all))

        # episode boundaries
        if "episode_ends" in data:
            ends = _parse_episode_ends(data["episode_ends"], N)
            start = 0
            for end in ends:
                end = int(end)
                if end <= start:
                    continue
                obs_seq = _slice_obs(obs_all, start, end)
                trajs.append(
                    _make_trajectory(
                        obs_seq,
                        actions_all[start:end],
                        rewards_all[start:end],
                        gamma=float(gamma),
                        obs_keys=obs_keys,
                    )
                )
                start = end
            return trajs

        # fallback: dones/terminals
        if "dones" in data:
            dones = np.asarray(data["dones"], dtype=bool).reshape(-1)
        elif "terminals" in data:
            dones = np.asarray(data["terminals"], dtype=bool).reshape(-1)
        else:
            raise ValueError("Cannot find episode boundaries (no 'episode_ends', 'dones' or 'terminals').")

        start = 0
        for t in range(N):
            if dones[t] or t == N - 1:
                end = t + 1
                if end <= start:
                    start = end
                    continue
                obs_seq = _slice_obs(obs_all, start, end)
                trajs.append(
                    _make_trajectory(
                        obs_seq,
                        actions_all[start:end],
                        rewards_all[start:end],
                        gamma=float(gamma),
                        obs_keys=obs_keys,
                    )
                )
                start = end
        return trajs

    raise ValueError(f"Unsupported panda dataset format in {path}: type {type(data)}")


# -----------------------------------------------------------------------------
# Minibatch sampler
# -----------------------------------------------------------------------------
def make_minibatches_panda(
    trajs: List[Trajectory],
    *,
    seq_len: int,
    batch_size: int,
    device: str,
    act_dim: int,
    obs_dim: int,
) -> Iterator:
    seq_len = int(seq_len)
    batch_size = int(batch_size)
    dev = torch.device(device)

    n_windows = np.array([max(1, len(t.actions) - seq_len + 1) for t in trajs], dtype=np.int64)
    cdf = np.cumsum(n_windows / n_windows.sum())

    while True:
        u = np.random.rand(batch_size)
        idxs = np.searchsorted(cdf, u, side="right")

        obs_b = np.zeros((batch_size, seq_len, obs_dim), dtype=np.float32)
        act_b = np.zeros((batch_size, seq_len, act_dim), dtype=np.float32)
        rtg_b = np.zeros((batch_size, seq_len, 1), dtype=np.float32)
        ts_b  = np.zeros((batch_size, seq_len), dtype=np.int64)
        mask_b = np.zeros((batch_size, seq_len), dtype=np.bool_)

        for b, idx in enumerate(idxs):
            tr = trajs[int(idx)]
            L = int(len(tr.actions))

            if L <= seq_len:
                si = 0
            else:
                si = np.random.randint(0, L - seq_len + 1)

            end = min(si + seq_len, L)
            tlen = end - si
            pad = seq_len - tlen

            obs = np.asarray(tr.obs, dtype=np.float32).reshape(-1, obs_dim)[si:end]
            act = np.asarray(tr.actions, dtype=np.float32).reshape(-1, act_dim)[si:end]
            rtg = np.asarray(tr.returns_to_go, dtype=np.float32).reshape(-1)[si:end]

            obs_b[b, pad:, :] = obs
            act_b[b, pad:, :] = act
            rtg_b[b, pad:, 0] = rtg
            ts_b[b, pad:] = np.arange(si, si + tlen, dtype=np.int64)
            mask_b[b, pad:] = True

        yield (
            torch.from_numpy(obs_b).to(dev),
            torch.from_numpy(act_b).to(dev),
            torch.from_numpy(rtg_b).to(dev),
            torch.from_numpy(ts_b).to(dev),
            torch.from_numpy(mask_b).to(dev),
        )