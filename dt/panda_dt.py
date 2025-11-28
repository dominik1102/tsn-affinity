from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class ObsEncoder(nn.Module):
    """
    Simple MLP encoder for vector observations.
    """

    def __init__(self, obs_dim: int, d_model: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B*L, obs_dim]

        Returns:
            Tensor of shape [B*L, d_model].
        """
        return self.enc(x)


class PandaDecisionTransformer(nn.Module):
    """
    Decision Transformer for continuous Panda actions.

    During training:
        obs:       [B, L, obs_dim]
        actions:   [B, L, act_dim]  (previous actions, shifted)
        rtg:       [B, L, 1]
        timesteps: [B, L]

    Returns:
        [B, L, act_dim] — predicted continuous actions.
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
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.act_dim = act_dim
        self.obs_dim = obs_dim

        # Encoders
        self.obs_enc = ObsEncoder(obs_dim, d_model)
        self.embed_action = nn.Linear(act_dim, d_model)
        self.embed_rtg = nn.Linear(1, d_model)
        self.embed_t = nn.Embedding(2048, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=p_drop,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, act_dim),
        )

        # Causal mask for up to seq_len timesteps (3 tokens per step)
        max_tokens = seq_len * 3
        causal_mask = torch.triu(
            torch.ones(max_tokens, max_tokens, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask)

        # Internal history buffers for act()
        self.reset_history()

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rtg: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            obs:       [B, L, obs_dim]
            actions:   [B, L, act_dim]  (previous actions, shifted)
            rtg:       [B, L, 1]
            timesteps: [B, L]

        Returns:
            Tensor of shape [B, L, act_dim] with predicted actions.
        """
        B, L, _ = obs.shape

        # Encode observations
        s_tok = self.obs_enc(obs.view(B * L, -1)).view(B, L, -1)
        a_tok = self.embed_action(actions)
        r_tok = self.embed_rtg(rtg)
        t_tok = self.embed_t(
            timesteps.clamp(max=self.embed_t.num_embeddings - 1)
        )

        # Stack tokens [r, s, a] per timestep -> [B, 3L, d_model]
        tokens = torch.stack([r_tok, s_tok, a_tok], dim=2).reshape(
            B, L * 3, self.d_model
        )
        tokens = tokens + t_tok.repeat_interleave(3, dim=1)

        # Causal mask: prevent attending to future tokens
        T = tokens.size(1)
        if T <= self.causal_mask.size(0):
            attn_mask = self.causal_mask[:T, :T]
        else:
            attn_mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=tokens.device),
                diagonal=1,
            )

        z = self.transformer(tokens, mask=attn_mask)

        # Take only "state" positions: indices 1, 4, 7, ... = 3*t + 1
        pos = torch.arange(1, L * 3, 3, device=obs.device)
        z_state = z[:, pos, :]  # [B, L, d_model]

        pred_actions = self.head(z_state)  # [B, L, act_dim]
        return pred_actions

    # ------------------------------------------------------------------
    # History handling for act()
    # ------------------------------------------------------------------
    def reset_history(self):
        """
        Reset internal history buffers used by act().

        Call this method at the beginning of each Panda episode.
        """
        self._hist_obs: list[torch.Tensor] = []
        self._hist_actions: list[np.ndarray] = []
        self._hist_rtgs: list[float] = []
        self._hist_t: list[int] = []

    @torch.no_grad()
    def act(
        self,
        obs,
        rtg_scalar: float,
        t: int,
        prev_action=None,  # kept for API compatibility
        device: str = "cpu",
    ):
        """
        Greedy action selection with a rolling context for continuous actions.

        This method:
          - appends the current observation and RTG to internal history,
          - builds a sequence of up to `seq_len` past (obs, action, rtg, t),
          - runs a forward pass and returns the last predicted action.

        Args:
            obs: flat observation vector (obs_dim,). If your env returns a dict,
                 flatten it first using the same logic as in the offline dataset.
            rtg_scalar: scalar return-to-go (placeholder is fine).
            t: environment timestep.
            prev_action: unused; history is tracked inside the model.
            device: 'cpu' or 'cuda'.

        Returns:
            action as a numpy array of shape [act_dim].
        """
        self.eval()
        device = torch.device(device)

        if isinstance(obs, dict):
            raise ValueError(
                "PandaDecisionTransformer.act expects a flat observation vector, "
                "got a dict. Flatten obs before calling this method."
            )

        # Convert current observation to tensor and store in history
        o_np = np.asarray(obs, dtype=np.float32).ravel()
        o_t = torch.from_numpy(o_np).to(device=device)
        self._hist_obs.append(o_t)
        self._hist_rtgs.append(float(rtg_scalar))
        self._hist_t.append(int(t))

        # Keep only the last `seq_len` steps
        if len(self._hist_obs) > self.seq_len:
            self._hist_obs = self._hist_obs[-self.seq_len:]
            self._hist_rtgs = self._hist_rtgs[-self.seq_len:]
            self._hist_t = self._hist_t[-self.seq_len:]
            if self._hist_actions:
                self._hist_actions = self._hist_actions[-self.seq_len:]

        L = len(self._hist_obs)
        B = 1

        # Build observation batch: [1, L, obs_dim]
        obs_seq = torch.stack(self._hist_obs, dim=0).unsqueeze(0)

        # Build actions batch: previous actions per step [1, L, act_dim]
        zero_action = torch.zeros(self.act_dim, dtype=torch.float32, device=device)
        actions_tensors = []
        for i in range(L):
            if i == 0:
                # First step has no previous action
                actions_tensors.append(zero_action)
            else:
                idx = i - 1
                if idx < len(self._hist_actions):
                    a_prev_np = np.asarray(self._hist_actions[idx], dtype=np.float32)
                    a_prev_t = torch.from_numpy(a_prev_np).to(
                        device=device, dtype=torch.float32
                    )
                    actions_tensors.append(a_prev_t)
                else:
                    actions_tensors.append(zero_action)

        actions_seq = torch.stack(actions_tensors, dim=0).unsqueeze(0)  # [1, L, act_dim]

        # Build RTG and timestep sequences
        rtg_seq = torch.tensor(
            self._hist_rtgs, dtype=torch.float32, device=device
        ).view(B, L, 1)

        ts_seq = torch.tensor(self._hist_t, dtype=torch.long, device=device)
        max_t = self.embed_t.num_embeddings
        ts_seq = ts_seq.clamp(max=max_t - 1).view(B, L)

        # Forward pass over the full history
        pred_seq = self.forward(obs_seq, actions_seq, rtg_seq, ts_seq)  # [1, L, act_dim]
        action_t = pred_seq[:, -1, :]  # [1, act_dim]

        # Store and return action
        action_np = action_t[0].detach().cpu().numpy()
        self._hist_actions.append(action_np)
        if len(self._hist_actions) > self.seq_len:
            self._hist_actions = self._hist_actions[-self.seq_len:]

        return action_np
