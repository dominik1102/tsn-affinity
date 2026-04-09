from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import torch

from dt.dataset import Trajectory
from dt.dataset_panda import make_minibatches_panda


# ============================================================
# Common batch helpers
# ============================================================

def unpack_batch_discrete(batch):
    """
    Normalize discrete-action batches (Atari) to one common format.

    Supported layouts:
      - (obs, actions, rtg, ts, mask)
      - (obs, actions, rtg, ts)
      - dict-like batch with the same logical fields

    If the loader does not provide `mask`, it is reconstructed from padded
    actions. In `dt.dataset.make_minibatches(...)` padded actions are -1,
    so valid positions are exactly `actions != -1`.
    """
    if isinstance(batch, dict):
        obs = batch.get("obs", batch.get("observations"))
        actions = batch.get("actions")
        rtg = batch.get("rtg", batch.get("returns_to_go"))
        ts = batch.get("timesteps", batch.get("ts"))
        mask = batch.get("mask", batch.get("attention_mask"))
    elif isinstance(batch, (tuple, list)):
        if len(batch) == 5:
            obs, actions, rtg, ts, mask = batch
        elif len(batch) == 4:
            obs, actions, rtg, ts = batch
            mask = None
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    obs = obs.float()
    actions = actions.long()
    rtg = rtg.float()
    ts = ts.long()

    if rtg.dim() == 2:
        rtg = rtg.unsqueeze(-1)

    if mask is None:
        mask = (actions != -1)
    else:
        mask = mask.bool()

    return obs, actions, rtg, ts, mask


def _unpack_batch(batch):
    """
    Backward-compatible alias used by existing Atari/TSN code.
    """
    return unpack_batch_discrete(batch)


def unpack_batch_continuous(batch):
    """
    Normalize continuous-action batches (Panda) to one common format.

    Supported layouts:
      - (obs, actions, rtg, ts, mask)
      - (obs, actions, rtg, ts)
      - dict-like batch with the same logical fields

    If the loader does not provide `mask`, we assume all sequence positions are
    valid and create an all-ones mask.
    """
    if isinstance(batch, dict):
        obs = batch.get("obs", batch.get("observations"))
        actions = batch.get("actions")
        rtg = batch.get("rtg", batch.get("returns_to_go"))
        ts = batch.get("timesteps", batch.get("ts"))
        mask = batch.get("mask", batch.get("attention_mask"))
    elif isinstance(batch, (tuple, list)):
        if len(batch) == 5:
            obs, actions, rtg, ts, mask = batch
        elif len(batch) == 4:
            obs, actions, rtg, ts = batch
            mask = None
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    obs = obs.float()
    actions = actions.float()
    rtg = rtg.float()
    ts = ts.long()

    if rtg.dim() == 2:
        rtg = rtg.unsqueeze(-1)

    if mask is None:
        B, L = actions.shape[:2]
        mask = torch.ones((B, L), device=obs.device, dtype=torch.bool)
    else:
        mask = mask.bool()

    return obs, actions, rtg, ts, mask


# ============================================================
# Common losses
# ============================================================

def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    action_dim_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Masked MSE over sequence positions and optionally over action dimensions.

    Args:
        pred:  [B, L, D]
        target:[B, L, D]
        mask:  [B, L] bool
        action_dim_mask: optional [D] tensor in {0,1}
    """
    diff2 = (pred - target) ** 2  # [B, L, D]

    if action_dim_mask is not None:
        adm = action_dim_mask.to(device=diff2.device, dtype=diff2.dtype).view(1, 1, -1)
        diff2 = diff2 * adm
        denom_d = adm.sum().clamp(min=1.0)
        mse_per_step = diff2.sum(dim=-1) / denom_d
    else:
        mse_per_step = diff2.mean(dim=-1)

    mask_f = mask.to(dtype=mse_per_step.dtype)
    denom_t = mask_f.sum().clamp(min=1.0)
    return (mse_per_step * mask_f).sum() / denom_t


# ============================================================
# Panda trajectory / loader helpers
# ============================================================

def pad_traj_obs_dim(tr: Trajectory, target_obs_dim: int) -> Trajectory:
    obs = np.asarray(tr.obs, dtype=np.float32)
    cur_obs_dim = int(obs.shape[-1])

    if cur_obs_dim > int(target_obs_dim):
        raise ValueError(
            f"Trajectory obs_dim={cur_obs_dim} is larger than target obs_dim={target_obs_dim}."
        )

    if cur_obs_dim == int(target_obs_dim):
        return tr

    pad = np.zeros((obs.shape[0], int(target_obs_dim) - cur_obs_dim), dtype=np.float32)
    obs_new = np.concatenate([obs, pad], axis=-1).astype(np.float32, copy=False)

    return Trajectory(
        obs=obs_new,
        actions=np.asarray(tr.actions, dtype=np.float32),
        rewards=np.asarray(tr.rewards, dtype=np.float32),
        timesteps=np.asarray(tr.timesteps, dtype=np.int64),
        returns_to_go=np.asarray(tr.returns_to_go, dtype=np.float32),
    )


def pad_traj_act_dim(tr: Trajectory, target_act_dim: int) -> Trajectory:
    actions = np.asarray(tr.actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[:, None]
    cur_act_dim = int(actions.shape[-1])

    if cur_act_dim > int(target_act_dim):
        raise ValueError(
            f"Trajectory act_dim={cur_act_dim} is larger than target act_dim={target_act_dim}."
        )

    if cur_act_dim == int(target_act_dim):
        return Trajectory(
            obs=np.asarray(tr.obs, dtype=np.float32),
            actions=actions.astype(np.float32, copy=False),
            rewards=np.asarray(tr.rewards, dtype=np.float32),
            timesteps=np.asarray(tr.timesteps, dtype=np.int64),
            returns_to_go=np.asarray(tr.returns_to_go, dtype=np.float32),
        )

    pad = np.zeros((actions.shape[0], int(target_act_dim) - cur_act_dim), dtype=np.float32)
    act_new = np.concatenate([actions, pad], axis=-1).astype(np.float32, copy=False)

    return Trajectory(
        obs=np.asarray(tr.obs, dtype=np.float32),
        actions=act_new,
        rewards=np.asarray(tr.rewards, dtype=np.float32),
        timesteps=np.asarray(tr.timesteps, dtype=np.int64),
        returns_to_go=np.asarray(tr.returns_to_go, dtype=np.float32),
    )


def prepare_panda_trajs(
    trajs: List[Trajectory],
    *,
    obs_dim: int,
    act_dim: int,
) -> List[Trajectory]:
    out: List[Trajectory] = []
    for tr in trajs:
        x = pad_traj_obs_dim(tr, obs_dim)
        x = pad_traj_act_dim(x, act_dim)
        out.append(x)
    return out


def infer_obs_dim(trajs: List[Trajectory]) -> int:
    if not trajs:
        return 0
    return int(np.asarray(trajs[0].obs).shape[-1])


def infer_act_dim(trajs: List[Trajectory]) -> int:
    if not trajs:
        return 0
    a = np.asarray(trajs[0].actions)
    return int(1 if a.ndim == 1 else a.shape[-1])


def compute_obs_stats(trajs: List[Trajectory], obs_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    if not trajs:
        return np.zeros(obs_dim, dtype=np.float32), np.ones(obs_dim, dtype=np.float32)
    xs = [np.asarray(t.obs, dtype=np.float32).reshape(-1, obs_dim) for t in trajs]
    x = np.concatenate(xs, axis=0)
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std = np.clip(std, 1e-6, None)
    return mean, std


def make_panda_loader(
    trajs: List[Trajectory],
    seq_len: int,
    batch_size: int,
    device: Any,
    *,
    obs_dim: Optional[int] = None,
    act_dim: Optional[int] = None,
):
    """
    Wrapper around dt.dataset_panda.make_minibatches_panda with compatibility
    for both the newer keyword-based and older positional calling conventions.
    """
    if obs_dim is not None:
        for tr in trajs:
            cur = int(np.asarray(tr.obs).shape[-1])
            if cur != int(obs_dim):
                raise ValueError(
                    f"make_panda_loader got trajectory with obs_dim={cur}, expected obs_dim={obs_dim}."
                )

    try:
        return make_minibatches_panda(
            trajs,
            seq_len=seq_len,
            batch_size=batch_size,
            device=device,
            obs_dim=obs_dim,
            act_dim=act_dim,
        )
    except TypeError:
        return make_minibatches_panda(trajs, seq_len, batch_size, device)


def infer_active_action_mask(
    task_trajs: List[Trajectory],
    act_dim: int,
    *,
    max_samples: int = 20000,
) -> torch.Tensor:
    """
    Infer which action dimensions are truly active in a Panda task.

    This is useful when several tasks share a padded global action head, e.g.:
      - Reach / Push: 3D actions padded to 4D
      - PickAndPlace: real 4D actions

    Returns a float tensor [act_dim] in {0,1}.
    """
    chunks = []
    n = 0
    for tr in task_trajs:
        a = np.asarray(tr.actions, dtype=np.float32)
        if a.ndim != 2:
            continue
        chunks.append(a)
        n += a.shape[0]
        if n >= max_samples:
            break

    if not chunks:
        return torch.ones(act_dim, dtype=torch.float32)

    A = np.concatenate(chunks, axis=0)
    if A.shape[1] != act_dim:
        return torch.ones(act_dim, dtype=torch.float32)

    std = A.std(axis=0)
    mx = np.abs(A).max(axis=0)
    active = (std > 1e-8) | (mx > 1e-8)

    m = torch.zeros(act_dim, dtype=torch.float32)
    m[torch.from_numpy(active.astype(np.bool_))] = 1.0
    if float(m.sum().item()) < 1.0:
        m[:] = 1.0
    return m


# ------------------------------------------------------------
# Backward-compatible aliases used by older files
# ------------------------------------------------------------
_pad_traj_obs_dim = pad_traj_obs_dim
_pad_traj_act_dim = pad_traj_act_dim
_prepare_panda_trajs = prepare_panda_trajs
_infer_obs_dim = infer_obs_dim
_infer_act_dim = infer_act_dim
_compute_obs_stats = compute_obs_stats
_make_panda_loader = make_panda_loader
_infer_active_action_mask = infer_active_action_mask
_masked_mse = masked_mse
