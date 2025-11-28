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
        # x: [B*L, obs_dim]
        return self.enc(x)


class PandaDecisionTransformer(nn.Module):
    """
    Decision Transformer for continuous Panda actions.

    obs:   [B, L, obs_dim]
    acts:  [B, L, act_dim]   (continuous)
    rtg:   [B, L, 1]
    time:  [B, L]
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

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rtg: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        obs:       [B, L, obs_dim]
        actions:   [B, L, act_dim]  (previous actions, shifted)
        rtg:       [B, L, 1]
        timesteps: [B, L]
        returns:   [B, L, act_dim]  (predicted actions)
        """
        B, L, _ = obs.shape

        # encode observations
        s_tok = self.obs_enc(obs.view(B * L, -1)).view(B, L, -1)
        a_tok = self.embed_action(actions)
        r_tok = self.embed_rtg(rtg)
        t_tok = self.embed_t(timesteps)

        # [r, s, a] per timestep -> sekwencja długości 3L
        tokens = torch.stack([r_tok, s_tok, a_tok], dim=2).reshape(
            B, L * 3, self.d_model
        )
        tokens = tokens + t_tok.repeat_interleave(3, dim=1)

        z = self.transformer(tokens)
        # bierzemy tokeny w pozycjach "state" (jak w Twoim dt)
        pos = torch.arange(1, L * 3, 3, device=obs.device)
        z_state = z[:, pos, :]  # [B, L, d_model]

        pred_actions = self.head(z_state)  # [B, L, act_dim]
        return pred_actions

    @torch.no_grad()
    def act(
        self,
        obs,
        rtg_scalar: float,
        t: int,
        prev_action=None,
        device: str = "cpu",
    ):
        """
        Prosta wersja act(): używa tylko bieżącego stanu + poprzedniej akcji.
        To nie jest pełne okno historii jak w starym projekcie, ale wystarczy,
        żeby pipeline działał.
        """
        self.eval()

        if isinstance(obs, dict):
            raise ValueError(
                "PandaDecisionTransformer.act expects a flat observation vector, "
                "not a dict (flatten it before calling)."
            )

        o = np.asarray(obs, dtype=np.float32).ravel()
        o_t = torch.tensor(o, dtype=torch.float32, device=device).view(1, 1, -1)

        if prev_action is None:
            prev = torch.zeros(1, 1, self.act_dim, dtype=torch.float32, device=device)
        else:
            pa = np.asarray(prev_action, dtype=np.float32).ravel()
            prev = torch.tensor(pa, dtype=torch.float32, device=device).view(
                1, 1, -1
            )

        r = torch.tensor([[[rtg_scalar]]], dtype=torch.float32, device=device)
        ts = torch.tensor([[t]], dtype=torch.long, device=device)

        pred = self.forward(o_t, prev, r, ts)  # [1, 1, act_dim]
        return pred[0, 0].detach().cpu().numpy()
