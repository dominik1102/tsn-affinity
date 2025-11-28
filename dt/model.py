from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class ObsEncoder(nn.Module):
    """
    Encodes observations into a d_model-sized embedding.

    - For vector observations (shape [D]), uses an MLP.
    - For image observations (shape [C, H, W]), uses a small CNN + linear head.
    """

    def __init__(self, obs_shape, d_model: int):
        super().__init__()
        if len(obs_shape) == 1:
            # Vector observations, e.g. CartPole
            self.kind = "mlp"
            self.enc = nn.Sequential(
                nn.Linear(obs_shape[0], d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
        else:
            # Image observations, e.g. Atari
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
            # Determine flattened conv output size
            with torch.no_grad():
                dummy = torch.zeros(1, *obs_shape)
                out = self.enc[:-1](dummy)
                flat = out.view(1, -1).size(1)
            self.proj = nn.Sequential(
                nn.Linear(flat, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D] for vectors or [B, C, H, W] for images.

        Returns:
            Tensor of shape [B, d_model].
        """
        if self.kind == "mlp":
            return self.enc(x)
        z = self.enc(x)
        z = z.view(z.size(0), -1)
        return self.proj(z)


class DecisionTransformer(nn.Module):
    """
    Decision Transformer for discrete-action environments (CartPole, Atari).

    - Tokens per timestep: [return-to-go, state, action] -> 3 * L tokens.
    - Uses a causal mask so tokens can only attend to past and current tokens.
    - act() keeps a rolling context of up to `seq_len` steps.
    """

    def __init__(
        self,
        obs_shape,
        n_actions: int,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        seq_len: int = 20,
        p_drop: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # Observation encoder (vector or image)
        self.obs_enc = ObsEncoder(obs_shape, d_model)

        # Embeddings for action, return-to-go, timestep
        self.embed_action = nn.Embedding(n_actions, d_model)
        self.embed_rtg = nn.Linear(1, d_model)
        self.embed_t = nn.Embedding(2048, d_model)  # maximum timestep index

        # Transformer encoder (we supply an explicit causal mask)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=p_drop,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Head mapping state embeddings -> action logits
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_actions),
        )

        # Precompute a causal mask for up to seq_len timesteps (3 tokens per step)
        max_tokens = 3 * seq_len
        causal_mask = torch.triu(
            torch.ones(max_tokens, max_tokens, dtype=torch.bool), diagonal=1
        )
        # True = masked position (cannot attend there)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        # Internal buffers used by act()
        self.reset_history()

    def forward(
            self,
            obs: torch.Tensor,
            actions: torch.Tensor,
            rtg: torch.Tensor,
            timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass over a sequence.

        Args:
            obs:       [B, L, D] for vectors or [B, L, C, H, W] for images
            actions:   [B, L] with -1 used for padding
            rtg:       [B, L, 1] return-to-go per timestep
            timesteps: [B, L] integer timestep indices

        Returns:
            logits: [B, L, n_actions] — action distribution for each state_t.
        """
        B, L = actions.shape

        # Encode observations into state tokens
        if obs.dim() == 5:
            # Images: [B, L, C, H, W] -> [B*L, C, H, W]
            B, L, C, H, W = obs.shape
            obs_flat = obs.view(B * L, C, H, W)
            s_tok = self.obs_enc(obs_flat).view(B, L, -1)  # [B, L, d_model]
        else:
            # Vectors: [B, L, D] -> [B*L, D]
            obs_flat = obs.view(B * L, -1)
            s_tok = self.obs_enc(obs_flat).view(B, L, -1)

        # Action embeddings (use -1 as padding index)
        a_mask = (actions.unsqueeze(-1) >= 0).float()  # 1.0 where valid
        a_clamped = torch.clamp(actions, min=0)
        a_tok = self.embed_action(a_clamped) * a_mask  # [B, L, d_model]

        # Return-to-go and timestep embeddings
        r_tok = self.embed_rtg(rtg)  # [B, L, d_model]
        max_t_embed = self.embed_t.num_embeddings
        ts_clamped = timesteps.clamp(max=max_t_embed - 1)
        t_tok = self.embed_t(ts_clamped)  # [B, L, d_model]

        # Stack tokens as [R_t, S_t, A_t] -> [B, 3L, d_model]
        tokens = torch.stack([r_tok, s_tok, a_tok], dim=2).reshape(
            B, L * 3, self.d_model
        )

        # Add timestep embedding to all three tokens per timestep
        t_tok_expanded = t_tok.repeat_interleave(3, dim=1)  # [B, 3L, d_model]
        tokens = tokens + t_tok_expanded

        # Apply causal mask so each token sees only past and current tokens
        T = L * 3
        mask = self.causal_mask[:T, :T]  # [T, T]
        z = self.transformer(tokens, mask=mask)  # [B, 3L, d_model]

        # Use only state positions: indices 1, 4, 7, ... = 3*t + 1
        pos = torch.arange(1, T, 3, device=obs.device)
        z_state = z[:, pos, :]  # [B, L, d_model]

        logits = self.head(z_state)  # [B, L, n_actions]
        return logits

    # ------------------------------------------------------------------
    # History management for act()
    # ------------------------------------------------------------------
    def reset_history(self):
        """
        Reset internal history buffers used by act().

        Call this at the beginning of every new episode.
        """
        self._hist_obs: list[torch.Tensor] = []
        self._hist_actions: list[int] = []

    @torch.no_grad()
    def act(
            self,
            obs,
            rtg_scalar: float,
            t: int,
            prev_action: int,
            device: str = "cpu",
            n_actions: int | None = None,
    ) -> int:
        """
        Greedy action selection with a rolling context of up to `seq_len` steps.

        IMPORTANT:
          All tensors are moved to the same device as the model parameters.
          The `device` argument is treated only as a hint; the actual device
          is taken from `next(self.parameters()).device`.
        """
        self.eval()

        # 1) Determine the true device from model params
        param_device = next(self.parameters()).device
        device = torch.device(param_device)

        # 2) Convert observation to tensor on that device
        if isinstance(obs, (list, tuple)):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        elif isinstance(obs, torch.Tensor):
            obs_t = obs.to(device=device, dtype=torch.float32)
        else:
            obs_arr = np.asarray(obs, dtype=np.float32)
            obs_t = torch.from_numpy(obs_arr).to(device=device)

        # ---- history update ----
        self._hist_obs.append(obs_t)
        if len(self._hist_obs) > self.seq_len:
            self._hist_obs.pop(0)
            if len(self._hist_actions) > 0:
                self._hist_actions.pop(0)

        L = len(self._hist_obs)
        dummy_action = -1
        actions_list = list(self._hist_actions) + [dummy_action]
        assert len(actions_list) == L

        actions = torch.tensor(
            actions_list, dtype=torch.long, device=device
        ).unsqueeze(0)  # [1, L]
        rtg = torch.full(
            (1, L, 1), float(rtg_scalar), dtype=torch.float32, device=device
        )

        # timesteps [1, L]
        start_t = max(0, t - L + 1)
        t_seq = torch.arange(
            start_t, start_t + L, dtype=torch.long, device=device
        ).unsqueeze(0)
        max_t_embed = self.embed_t.num_embeddings
        t_seq = torch.clamp(t_seq, max=max_t_embed - 1)

        # Build observation batch: [1, L, ...]
        if obs_t.dim() == 1:
            obs_batch = torch.stack(self._hist_obs, dim=0).unsqueeze(0)  # [1, L, D]
        elif obs_t.dim() == 3:
            obs_batch = torch.stack(self._hist_obs, dim=0).unsqueeze(0)  # [1, L, C,H,W]
        else:
            raise ValueError(f"Unsupported obs shape for act(): {obs_t.shape}")

        # ---- forward ----
        logits = self.forward(obs_batch, actions, rtg, t_seq)  # [1, L, n_actions]
        logits_last = logits[:, -1, :]

        if n_actions is not None:
            n_model = logits_last.shape[-1]
            if n_actions > n_model:
                raise ValueError(
                    f"Requested n_actions={n_actions}, but model head has only {n_model}"
                )
            logits_last = logits_last.clone()
            logits_last[..., n_actions:] = -1e9

        action = int(torch.argmax(logits_last, dim=-1).item())

        # update history
        self._hist_actions.append(action)
        if len(self._hist_actions) > self.seq_len:
            self._hist_actions.pop(0)

        return action




