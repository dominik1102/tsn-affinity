
from __future__ import annotations
import numpy as np, torch
from typing import List

class Trajectory:
    def __init__(self, obs, actions, rewards, timesteps, returns_to_go):
        self.obs = obs; self.actions = actions; self.rewards = rewards
        self.timesteps = timesteps; self.returns_to_go = returns_to_go

def discount_cumsum(x: np.ndarray, gamma: float=1.0) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32); run = 0.0
    for t in reversed(range(len(x))): run = x[t] + gamma * run; out[t] = run
    return out

def make_minibatches(trajs: List[Trajectory], seq_len: int, batch_size: int, device: str):
    while True:
        B = []
        for _ in range(batch_size):
            import numpy as _np
            tr = _np.random.choice(trajs); T = len(tr.actions)
            start = 0 if T <= seq_len else _np.random.randint(0, T-seq_len+1)
            end = min(start+seq_len, T)
            o = tr.obs[start:end]; a = tr.actions[start:end]; rtg = tr.returns_to_go[start:end]; ts = tr.timesteps[start:end]
            pad = seq_len - len(a)
            if o.ndim == 2:
                if pad>0: o = _np.pad(o, ((0,pad),(0,0)))
            else:
                if pad>0: o = _np.pad(o, ((0,pad),(0,0),(0,0),(0,0)))
            if pad>0:
                a = _np.pad(a, (0,pad), constant_values=-1); rtg = _np.pad(rtg, (0,pad)); ts = _np.pad(ts, (0,pad))
            B.append((o, a, rtg[:,None], ts))
        obs = torch.tensor(np.stack([b[0] for b in B]), dtype=torch.float32, device=device)
        actions = torch.tensor(np.stack([b[1] for b in B]), dtype=torch.long, device=device)
        rtg = torch.tensor(np.stack([b[2] for b in B]), dtype=torch.float32, device=device)
        ts = torch.tensor(np.stack([b[3] for b in B]), dtype=torch.long, device=device)
        yield obs, actions, rtg, ts
