from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from dt.model import DecisionTransformerConfig, ObsEncoder, DTBackbone


class PandaDecisionTransformer(nn.Module):
    """
    Decision Transformer for continuous Panda actions, built on top of DTBackbone.

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
        max_ep_len: int = 2048,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.max_ep_len = max_ep_len

        # Reuse the generic ObsEncoder from above (vector observations)
        self.obs_enc = ObsEncoder((obs_dim,), d_model)

        # DTBackbone configuration (continuous actions)
        config = DecisionTransformerConfig(
            n_layer=n_layers,
            n_head=n_heads,
            n_embd=d_model,
            dropout=p_drop,
            bias=False,
            K=seq_len,
            max_ep_len=max_ep_len,
            state_dim=d_model,   # state = encoded observation
            act_dim=act_dim,     # continuous actions
            act_discrete=False,  # <-- continuous
            act_vocab_size=1,    # unused for continuous
            act_tanh=False,
            tanh_embeddings=False,
        )
        self.dt = DTBackbone(config)

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
        if obs.dim() != 3:
            raise ValueError(
                f"PandaDecisionTransformer.forward expects obs of shape [B, L, obs_dim], "
                f"got {obs.shape}"
            )

        B, L, _ = obs.shape
        device = obs.device

        # --- encode observations -> states [B, L, d_model] ---
        obs_flat = obs.view(B * L, -1)                # [B*L, obs_dim]
        s_tok = self.obs_enc(obs_flat).view(B, L, -1) # [B, L, d_model]

        # actions, rtg, timesteps
        actions = actions.to(device)
        rtg = rtg.to(device)
        timesteps = timesteps.to(device).long()
        timesteps = torch.clamp(timesteps, max=self.max_ep_len - 1)

        # full attention mask (no padding for now)
        attn_mask = torch.ones(B, L, dtype=torch.bool, device=device)

        # Call DTBackbone in "training mode" (targets != None) to get predictions for all L
        # For continuous actions DTBackbone uses MSE internally, but we ignore the loss here
        pred_actions, _ = self.dt(
            states=s_tok,
            actions=actions,
            rtgs=rtg,
            tsteps=timesteps,
            attn_mask=attn_mask,
            targets=actions,  # just to get [B, L, act_dim] outputs
        )
        # pred_actions: [B, L, act_dim]
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
            device: kept for API compatibility, actual device is taken from model params.

        Returns:
            action as a numpy array of shape [act_dim].
        """
        self.eval()

        # Use the real device of the model parameters
        param_device = next(self.parameters()).device
        device = torch.device(param_device)

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
        obs_seq = torch.stack(self._hist_obs, dim=0).unsqueeze(0).to(device=device)

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
        ts_seq = ts_seq.clamp(max=self.max_ep_len - 1).view(B, L)

        # Forward pass over the full history
        pred_seq = self.forward(obs_seq, actions_seq, rtg_seq, ts_seq)  # [1, L, act_dim]
        action_t = pred_seq[:, -1, :]  # [1, act_dim]

        # Store and return action
        action_np = action_t[0].detach().cpu().numpy()
        self._hist_actions.append(action_np)
        if len(self._hist_actions) > self.seq_len:
            self._hist_actions = self._hist_actions[-self.seq_len:]

        return action_np
