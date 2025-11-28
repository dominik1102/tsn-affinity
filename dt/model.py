
from __future__ import annotations
import torch, torch.nn as nn

class ObsEncoder(nn.Module):
    """Encodes observation into d_model. Supports vector and image inputs."""
    def __init__(self, obs_shape, d_model: int):
        super().__init__()
        if len(obs_shape) == 1:
            self.kind = "mlp"
            self.enc = nn.Sequential(
                nn.Linear(obs_shape[0], d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
        else:
            self.kind = "cnn"
            c, h, w = obs_shape
            self.enc = nn.Sequential(
                nn.Conv2d(c, 32, 8, stride=4),
                nn.GELU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.GELU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.GELU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, *obs_shape)
                out = self.enc[:-1](dummy)
                flat = out.view(1, -1).size(1)
            self.proj = nn.Sequential(nn.Linear(flat, d_model), nn.GELU(), nn.LayerNorm(d_model))
    def forward(self, x):
        if self.kind == "mlp":
            return self.enc(x)
        z = self.enc(x)
        return self.proj(z.view(z.size(0), -1))

class DecisionTransformer(nn.Module):
    def __init__(self, obs_shape, n_actions: int, d_model: int=128, n_layers: int=3, n_heads: int=4, seq_len: int=20, p_drop: float=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.obs_enc = ObsEncoder(obs_shape, d_model)
        self.embed_action = nn.Embedding(n_actions, d_model)
        self.embed_rtg = nn.Linear(1, d_model)
        self.embed_t = nn.Embedding(2048, d_model)

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model, dropout=p_drop, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_actions))

    def forward(self, obs, actions, rtg, timesteps):
        # obs: [B,L,*], actions: [B,L], rtg: [B,L,1], timesteps: [B,L]
        B, L = actions.shape
        if obs.dim() == 5:  # images [B,L,C,H,W]
            B, L, C, H, W = obs.shape
            obs = obs.view(B*L, C, H, W)
            s_tok = self.obs_enc(obs).view(B, L, -1)
        else:  # vectors [B,L,D]
            s_tok = self.obs_enc(obs.view(B*L, -1)).view(B, L, -1)

        a_tok = self.embed_action(torch.clamp(actions, min=0)) * (actions.unsqueeze(-1) >= 0).float()
        r_tok = self.embed_rtg(rtg)
        t_tok = self.embed_t(timesteps)

        tokens = torch.stack([r_tok, s_tok, a_tok], dim=2).reshape(B, L*3, self.d_model)
        tokens = tokens + t_tok.repeat_interleave(3, dim=1)

        z = self.transformer(tokens)
        pos = torch.arange(1, L*3, 3, device=obs.device)
        z_state = z[:, pos, :]
        logits = self.head(z_state)
        return logits

    @torch.no_grad()
    def act(
            self,
            obs,
            rtg_scalar: float,
            t: int,
            prev_action: int,
            device: str = "cpu",
            n_actions: int | None = None,
    ):
        """
        Greedy action selection for discrete-action environments.

        This is used only at evaluation / trajectory collection time.
        Observations can be flat vectors or images (CHW / HWC).
        """
        self.eval()
        import torch
        import numpy as np

        # ---- Convert observation to a [1, 1, ...] tensor on the correct device ----
        if isinstance(obs, (list, tuple)):
            obs = torch.tensor(obs, dtype=torch.float32)

        if torch.is_tensor(obs):
            if obs.dim() == 1:
                # Vector observation: [D] -> [B=1, T=1, D]
                o = obs.view(1, 1, -1).to(device)
            elif obs.dim() == 3:
                # Image observation: [C, H, W] or [H, W, C] already converted upstream
                # -> [B=1, T=1, C, H, W]
                o = obs.unsqueeze(0).unsqueeze(0).to(device)
            else:
                raise ValueError(f"Unsupported obs shape for act(): {obs.shape}")
        else:
            oarr = np.array(obs, dtype=np.float32)
            if oarr.ndim == 1:
                o = torch.tensor(oarr).view(1, 1, -1).to(device)
            elif oarr.ndim == 3:
                o = torch.tensor(oarr).unsqueeze(0).unsqueeze(0).to(device)
            else:
                raise ValueError(f"Unsupported obs shape for act(): {oarr.shape}")

        # Previous action and return-to-go
        a = torch.tensor([[prev_action]], dtype=torch.long, device=device)
        r = torch.tensor([[[rtg_scalar]]], dtype=torch.float32, device=device)

        # Use a safe timestep index during evaluation to avoid out-of-range
        # indices in the timestep embedding.
        #
        # If your model exposes e.g. self.max_timestep, you can clamp:
        #   t_for_model = min(t, self.max_timestep - 1)
        # For now we just use 0, which is always in range.
        t_for_model = 0
        ts = torch.tensor([[t_for_model]], dtype=torch.long, device=device)

        # Forward pass: logits over actions at the last time step
        logits = self.forward(o, a, r, ts)[:, -1, :]  # [1, n_actions_model]

        # Optionally mask logits to the valid action range of the current env
        if n_actions is not None:
            n_model = logits.shape[-1]
            if n_actions > n_model:
                raise ValueError(
                    f"Requested n_actions={n_actions}, but model head has only {n_model}"
                )
            # Set logits for invalid actions to a very negative value
            logits[..., n_actions:] = -1e9

        action = int(torch.argmax(logits, dim=-1).item())
        return action




