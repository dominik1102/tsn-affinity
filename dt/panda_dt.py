from __future__ import annotations

from typing import Optional
import numpy as np
import torch
import torch.nn as nn

from dt.model import DTConfig, ObsEncoder, DTBackbone


class PandaDecisionTransformer(nn.Module):
    """
    Decision Transformer for continuous Panda actions (DTBackbone).

    TRAINING: actions aligned => actions[:, t] = a_t (NO SHIFT).
    INFERENCE: last action token is a zero placeholder; previous are past actions.

    Adds:
      - observation normalization (mean/std) like in your old working project,
      - rtg scaling via self.rtg_scale (Atari-style).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        seq_len: int = 20,
        p_drop: float = 0.1,
        max_ep_len: int = 256,
        act_tanh: bool = False,
        obs_mean: Optional[np.ndarray] = None,
        obs_std: Optional[np.ndarray] = None,
        rtg_scale: float = 1.0,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        self.act_dim = int(act_dim)
        self.obs_dim = int(obs_dim)
        self.max_ep_len = int(max_ep_len)

        # rtg_scale: keep as float attribute (safe with your other code style)
        self.rtg_scale = float(max(1e-6, rtg_scale))

        # --- obs normalization buffers (saved with checkpoint) ---
        self.register_buffer("obs_mean", torch.zeros(self.obs_dim, dtype=torch.float32), persistent=True)
        self.register_buffer("obs_std", torch.ones(self.obs_dim, dtype=torch.float32), persistent=True)

        if obs_mean is not None:
            m = torch.as_tensor(obs_mean, dtype=torch.float32).reshape(-1)
            if m.numel() != self.obs_dim:
                raise ValueError(f"obs_mean has dim={m.numel()} but obs_dim={self.obs_dim}")
            self.obs_mean.copy_(m)

        if obs_std is not None:
            s = torch.as_tensor(obs_std, dtype=torch.float32).reshape(-1)
            if s.numel() != self.obs_dim:
                raise ValueError(f"obs_std has dim={s.numel()} but obs_dim={self.obs_dim}")
            self.obs_std.copy_(torch.clamp(s, min=1e-6))

        self.obs_enc = ObsEncoder((self.obs_dim,), self.d_model)

        cfg = DTConfig(
            n_layer=n_layers,
            n_head=n_heads,
            n_embd=self.d_model,
            dropout=p_drop,
            bias=False,
            K=self.seq_len,
            max_ep_len=self.max_ep_len,
            state_dim=self.d_model,
            act_dim=self.act_dim,
            act_discrete=False,
            act_vocab_size=1,
            act_tanh=bool(act_tanh),
            tanh_embeddings=False,
        )
        self.dt = DTBackbone(cfg)

        self.reset_history()

    def _norm_obs(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [B,L,obs_dim]
        mean = self.obs_mean.view(1, 1, -1).to(device=obs.device, dtype=obs.dtype)
        std = self.obs_std.view(1, 1, -1).to(device=obs.device, dtype=obs.dtype)
        return (obs - mean) / std

    def forward(
        self,
        obs: torch.Tensor,                  # [B,L,obs_dim]
        actions: torch.Tensor,              # [B,L,act_dim] aligned (a_t)
        rtg: torch.Tensor,                  # [B,L,1] in *raw* reward units
        timesteps: torch.Tensor,            # [B,L]
        attention_mask: Optional[torch.Tensor] = None,  # [B,L] bool
    ) -> torch.Tensor:
        if obs.dim() != 3:
            raise ValueError(f"Expected obs [B,L,obs_dim], got {obs.shape}")

        B, L, _ = obs.shape
        device = obs.device

        obs = obs.to(device=device, dtype=torch.float32)
        actions = actions.to(device=device, dtype=torch.float32)
        rtg = rtg.to(device=device, dtype=torch.float32)

        # scale RTG like Atari runner does
        rtg = rtg / float(self.rtg_scale)

        timesteps = timesteps.to(device=device, dtype=torch.long)
        timesteps = torch.clamp(timesteps, max=self.max_ep_len - 1)

        obs = self._norm_obs(obs)

        # Encode obs -> states [B,L,d_model]
        obs_flat = obs.reshape(B * L, -1)
        s_tok = self.obs_enc(obs_flat).reshape(B, L, -1)

        if attention_mask is None:
            attn_mask = torch.ones((B, L), dtype=torch.bool, device=device)
        else:
            attn_mask = attention_mask.to(device=device, dtype=torch.bool)

        return self.dt(
            states=s_tok,
            actions=actions,
            rtgs=rtg,
            tsteps=timesteps,
            attn_mask=attn_mask,
        )

    def reset_history(self):
        self._hist_obs: list[np.ndarray] = []
        self._hist_actions: list[np.ndarray] = []
        self._hist_rtgs: list[float] = []
        self._hist_t: list[int] = []

    @staticmethod
    def _pad_or_trunc_1d(x: np.ndarray, size: int) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size < size:
            return np.pad(x, (0, size - x.size), mode="constant")
        if x.size > size:
            return x[:size]
        return x

    @torch.no_grad()
    def act(
            self,
            obs,
            rtg_scalar: float,
            t: int,
            prev_action=None,  # <-- NOW USED (previous action executed at t-1)
            device: str = "cpu",
    ) -> np.ndarray:
        self.eval()
        dev = next(self.parameters()).device

        # 1) append executed prev_action (a_{t-1}) if present
        if prev_action is not None:
            pa = self._pad_or_trunc_1d(np.asarray(prev_action, dtype=np.float32), self.act_dim)
            self._hist_actions.append(pa)

        # 2) append current observation and RTG (s_t, R_t)
        o = self._pad_or_trunc_1d(np.asarray(obs, dtype=np.float32), self.obs_dim)
        self._hist_obs.append(o)
        self._hist_rtgs.append(float(rtg_scalar))
        self._hist_t.append(int(t))

        # 3) trim to context K (important: keep obs vs actions aligned)
        if len(self._hist_obs) > self.seq_len:
            n_drop = len(self._hist_obs) - self.seq_len
            self._hist_obs = self._hist_obs[n_drop:]
            self._hist_rtgs = self._hist_rtgs[n_drop:]
            self._hist_t = self._hist_t[n_drop:]
            # actions correspond to obs[0..-2], so drop the same amount
            if n_drop > 0:
                self._hist_actions = self._hist_actions[n_drop:] if len(self._hist_actions) >= n_drop else []

        L = len(self._hist_obs)  # number of states in history (including current)
        K = self.seq_len
        start = K - L  # left pad

        obs_seq = torch.zeros((1, K, self.obs_dim), dtype=torch.float32, device=dev)
        act_seq = torch.zeros((1, K, self.act_dim), dtype=torch.float32, device=dev)
        rtg_seq = torch.zeros((1, K, 1), dtype=torch.float32, device=dev)
        ts_seq = torch.zeros((1, K), dtype=torch.long, device=dev)
        mask = torch.zeros((1, K), dtype=torch.bool, device=dev)

        obs_seq[0, start:, :] = torch.from_numpy(np.stack(self._hist_obs, axis=0)).to(dev)
        rtg_seq[0, start:, 0] = torch.tensor(self._hist_rtgs, dtype=torch.float32, device=dev)
        ts_seq[0, start:] = torch.tensor(self._hist_t, dtype=torch.long, device=dev).clamp(max=self.max_ep_len - 1)
        mask[0, start:] = True

        # actions history should have length L-1 (a_0..a_{t-1})
        past = self._hist_actions
        if len(past) > (L - 1):
            past = past[-(L - 1):]

        for i, a in enumerate(past):
            act_seq[0, start + i, :] = torch.from_numpy(self._pad_or_trunc_1d(a, self.act_dim)).to(dev)

        # last action token (for current step) remains 0 (placeholder)

        pred = self.forward(obs_seq, act_seq, rtg_seq, ts_seq, attention_mask=mask)  # [1,K,act_dim]
        a_t = pred[0, -1, :].detach().cpu().numpy().astype(np.float32, copy=False)

        return a_t