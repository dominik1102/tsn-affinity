from __future__ import annotations
from typing import Any, Dict, List
import pickle
import numpy as np

from dt.dataset import Trajectory, discount_cumsum


def flatten_panda_step(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Flatten single Panda obs dict into 1D vector.

    obs:
      - "observation": [...],
      - "desired_goal": [...],
      - "achieved_goal": [...]
    """
    o = np.asarray(obs["observation"], dtype=np.float32).ravel()
    dg = np.asarray(obs["desired_goal"], dtype=np.float32).ravel()
    ag = np.asarray(obs["achieved_goal"], dtype=np.float32).ravel()
    return np.concatenate([o, dg, ag], axis=0)


def _make_trajectory(
    obs_seq: Any,
    actions_seq: Any,
    rewards_seq: Any,
    gamma: float = 1.0,
) -> Trajectory:
    """
    Build Trajectory for Panda with continuous actions.

    obs_seq:
      - dict of arrays: keys "observation", "desired_goal", "achieved_goal",
        each [T, dim], OR
      - list of dicts / list of vectors [T, ...].
    actions_seq:
      - array [T, act_dim] (continuous)
    rewards_seq:
      - array/list [T]
    """
    # ---- OBS ----
    if isinstance(obs_seq, dict) and "observation" in obs_seq:
        # dict of arrays, shape [T, dim] each
        obs_arr = np.asarray(obs_seq["observation"], dtype=np.float32)
        dg_arr = np.asarray(obs_seq["desired_goal"], dtype=np.float32)
        ag_arr = np.asarray(obs_seq["achieved_goal"], dtype=np.float32)
        obs = np.concatenate([obs_arr, dg_arr, ag_arr], axis=-1).astype(np.float32)
    else:
        # list of dicts / vectors
        if (
            isinstance(obs_seq, (list, tuple))
            and len(obs_seq) > 0
            and isinstance(obs_seq[0], dict)
        ):
            obs = np.stack([flatten_panda_step(o) for o in obs_seq], axis=0).astype(
                np.float32
            )
        else:
            obs = np.asarray(obs_seq, dtype=np.float32)

    # ---- ACTIONS (ciągłe) ----
    actions = np.asarray(actions_seq, dtype=np.float32)
    rewards = np.asarray(rewards_seq, dtype=np.float32).reshape(-1)

    T = rewards.shape[0]

    # dopasowanie długości
    if obs.shape[0] == T + 1:
        obs = obs[:-1]
    if actions.shape[0] == T + 1:
        actions = actions[:-1]

    if obs.shape[0] != T or actions.shape[0] != T:
        raise ValueError(
            f"Length mismatch in Panda trajectory: "
            f"obs={obs.shape[0]}, actions={actions.shape[0]}, rewards={T}"
        )

    timesteps = np.arange(T, dtype=np.int64)
    returns_to_go = discount_cumsum(rewards, gamma=gamma)

    return Trajectory(
        obs=obs,
        actions=actions,
        rewards=rewards,
        timesteps=timesteps,
        returns_to_go=returns_to_go,
    )


def load_panda_offline_pkl(path: str, gamma: float = 1.0) -> List[Trajectory]:
    """
    Load panda-gym(-offline) dataset and convert to list[Trajectory].
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
            trajs.append(
                _make_trajectory(obs_seq, actions_seq, rewards_seq, gamma=gamma)
            )
        return trajs

    # Case 2: dict of arrays
    if isinstance(data, dict) and "observations" in data:
        obs_all = data["observations"]
        actions_all = data["actions"]
        rewards_all = data["rewards"]

        # ep boundaries
        if "episode_ends" in data:
            ends = np.asarray(data["episode_ends"], dtype=np.int64)
            start = 0
            for end in ends:
                trajs.append(
                    _make_trajectory(
                        obs_all[start:end],
                        actions_all[start:end],
                        rewards_all[start:end],
                        gamma=gamma,
                    )
                )
                start = end
        else:
            if "dones" in data:
                dones = np.asarray(data["dones"], dtype=bool)
            elif "terminals" in data:
                dones = np.asarray(data["terminals"], dtype=bool)
            else:
                raise ValueError(
                    "Cannot find episode boundaries in panda dataset "
                    "(no 'episode_ends', 'dones' or 'terminals')."
                )
            N = len(rewards_all)
            start = 0
            for t in range(N):
                if dones[t] or t == N - 1:
                    end = t + 1
                    trajs.append(
                        _make_trajectory(
                            obs_all[start:end],
                            actions_all[start:end],
                            rewards_all[start:end],
                            gamma=gamma,
                        )
                    )
                    start = end
        return trajs

    raise ValueError(f"Unsupported panda dataset format in {path}: type {type(data)}")


def make_minibatches_panda(
    trajs: List[Trajectory],
    seq_len: int,
    batch_size: int,
    device: str,
):
    """
    Minibatch generator dla Pandy z ciągłymi akcjami.
    Zwraca:
      obs:    [B, L, obs_dim]
      actions:[B, L, act_dim]
      rtg:    [B, L, 1]
      ts:     [B, L]
      mask:   [B, L]  (1 dla prawdziwych kroków, 0 dla paddingu)
    """
    import numpy as _np
    import torch

    while True:
        B_obs, B_actions, B_rtg, B_ts, B_mask = [], [], [], [], []

        for _ in range(batch_size):
            tr = _np.random.choice(trajs)
            T = tr.actions.shape[0]
            start = 0 if T <= seq_len else _np.random.randint(0, T - seq_len + 1)
            end = min(start + seq_len, T)

            o = tr.obs[start:end]          # [L, obs_dim]
            a = tr.actions[start:end]      # [L, act_dim]
            rtg = tr.returns_to_go[start:end]
            ts = tr.timesteps[start:end]

            L = a.shape[0]
            pad = seq_len - L

            mask = _np.zeros(seq_len, dtype=_np.float32)

            if pad > 0:
                # pad at END
                o = _np.pad(o, ((0, pad), (0, 0)), mode="constant")
                a = _np.pad(a, ((0, pad), (0, 0)), mode="constant")
                rtg = _np.pad(rtg, (0, pad), mode="constant")
                ts = _np.pad(ts, (0, pad), mode="constant")
                mask[:L] = 1.0
            else:
                mask[:] = 1.0

            B_obs.append(o)
            B_actions.append(a)
            B_rtg.append(rtg[:, None])
            B_ts.append(ts)
            B_mask.append(mask)

        obs = torch.tensor(_np.stack(B_obs), dtype=torch.float32, device=device)
        actions = torch.tensor(_np.stack(B_actions), dtype=torch.float32, device=device)
        rtg = torch.tensor(_np.stack(B_rtg), dtype=torch.float32, device=device)
        ts = torch.tensor(_np.stack(B_ts), dtype=torch.long, device=device)
        mask = torch.tensor(_np.stack(B_mask), dtype=torch.float32, device=device)

        yield obs, actions, rtg, ts, mask
