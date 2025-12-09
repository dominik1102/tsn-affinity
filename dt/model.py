from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


# ============================================================
# 1. Auxiliary layers (LayerNorm, Attention, Block)
#    – this is your "nanoGPT-style" part
# ============================================================

class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, bias, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        # flash attention
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Consider installing Flash Attention for faster training"
            )

        # causal mask (T x T)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)).view(
                1, 1, block_size, block_size
            ),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.size()

        # q, k, v
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # attn_mask [B, T] -> [B,1,1,T] and combine with causal bias
        if attn_mask is not None:
            attn_mask = attn_mask.view(B, 1, 1, T)
            attn_mask = attn_mask & self.bias[:, :, :T, :T]
        else:
            attn_mask = self.bias[:, :, :T, :T]

        if self.flash:
            # Flash attention expects additive mask
            float_mask = torch.zeros(B, 1, 1, T, device=attn_mask.device)
            float_mask = float_mask.masked_fill(~attn_mask, -10000.0)
            dropout_p = self.dropout if self.training else 0.0
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=float_mask, dropout_p=dropout_p)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(~attn_mask, -10000.0)
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd, bias, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, bias, block_size):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, bias, block_size)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, bias, dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


# ============================================================
# 2. Configuration and "bare" DT backbone (without observation encoding)
# ============================================================

@dataclass
class DecisionTransformerConfig:
    n_layer: int = 3
    n_head: int = 1
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = False
    K: int = 20                  # context length in timesteps
    max_ep_len: int = 1000       # for time embedding
    state_dim: int = 17
    act_dim: int = 6
    act_discrete: bool = False
    act_vocab_size: int = 1
    act_tanh: bool = False
    tanh_embeddings: bool = False


class DTBackbone(nn.Module):
    """
    Your original Decision Transformer (Karpathy-style),
    just under a new name so it doesn't conflict with the RL wrapper.
    """

    def __init__(self, config: DecisionTransformerConfig):
        super().__init__()
        self.config = config
        print(f"DTBackbone config: {config}")

        block_size = config.K * 3  # each block is composed of 3 tokens: R, s, a
        self.transformer = nn.ModuleDict(
            dict(
                te=nn.Embedding(config.max_ep_len, config.n_embd),
                re=nn.Linear(1, config.n_embd),
                se=nn.Linear(config.state_dim, config.n_embd),
                ae=(
                    nn.Embedding(config.act_vocab_size, config.n_embd)
                    if config.act_discrete
                    else nn.Linear(config.act_dim, config.n_embd)
                ),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(
                    [
                        Block(
                            config.n_embd,
                            config.n_head,
                            config.dropout,
                            config.bias,
                            block_size,
                        )
                        for _ in range(config.n_layer)
                    ]
                ),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
                ln_e=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        if config.act_discrete:
            self.act_head = nn.Linear(config.n_embd, config.act_vocab_size)
        else:
            self.act_head = nn.Linear(config.n_embd, config.act_dim)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        print("DTBackbone parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.te.weight.numel()
            n_params -= self.transformer.se.weight.numel()
            n_params -= self.transformer.ae.weight.numel()
            n_params -= self.transformer.re.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        states: torch.Tensor,   # [B, T, state_dim]
        actions: torch.Tensor,  # [B, T] for discrete or [B, T, act_dim]
        rtgs: torch.Tensor,     # [B, T, 1]
        tsteps: torch.Tensor,   # [B, T]
        attn_mask: torch.Tensor | None = None,  # [B, T] bool
        targets: torch.Tensor | None = None,    # for loss (discrete: [B, T])
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = states.device
        b, t = states.shape[0], states.shape[1]
        assert t <= self.config.K, f"Cannot forward sequence of length {t}, K is only {self.config.K}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # [t] (unused but OK)

        # embeddings
        state_emb = self.transformer.se(states)  # [B, T, n_embd]
        if self.config.act_discrete:
            # actions: [B, T] of ints
            action_emb = self.transformer.ae(actions.long())
        else:
            action_emb = self.transformer.ae(actions)

        rtg_emb = self.transformer.re(rtgs)      # [B, T, n_embd]
        tstep_emb = self.transformer.te(tsteps)  # [B, T, n_embd]

        if self.config.tanh_embeddings:
            state_emb = torch.tanh(state_emb)
            action_emb = torch.tanh(action_emb)
            rtg_emb = torch.tanh(rtg_emb)

        state_emb = state_emb + tstep_emb
        action_emb = action_emb + tstep_emb
        rtg_emb = rtg_emb + tstep_emb

        # [R_1, s_1, a_1, R_2, s_2, a_2, ...]
        stacked_emb = (
            torch.stack((rtg_emb, state_emb, action_emb), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(b, 3 * t, self.config.n_embd)
        )
        stacked_emb = self.transformer.ln_e(stacked_emb)

        if attn_mask is None:
            attn_mask = torch.ones(b, t, dtype=torch.bool, device=device)
        stacked_attn_mask = (
            torch.stack((attn_mask, attn_mask, attn_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(b, 3 * t)
        )

        x = self.transformer.drop(stacked_emb)
        for block in self.transformer.h:
            x = block(x, stacked_attn_mask)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # training path: full logits for all timesteps
            logits = self.act_head(x)
            if self.config.act_tanh:
                logits = torch.tanh(logits)
            logits = logits[:, 1::3, :]  # state positions
            if self.config.act_discrete:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-1,
                )
            else:
                act_dim = logits.shape[2]
                logits_ = logits.reshape(-1, act_dim)[attn_mask.reshape(-1) > 0]
                targets_ = targets.reshape(-1, act_dim)[attn_mask.reshape(-1) > 0]
                loss = F.mse_loss(logits_, targets_)
        else:
            # inference path: only the last action
            logits = self.act_head(x[:, [-2], :])
            if self.config.act_tanh:
                logits = torch.tanh(logits)
            loss = None

        return logits, loss


# ============================================================
# 3. ObsEncoder
# ============================================================

class ObsEncoder(nn.Module):
    """
    Encodes observations into a d_model-sized embedding.

    - For vector observations (shape [D]), uses an MLP.
    - For image observations (shape [C, H, W]), uses a small CNN + linear head.
    """

    def __init__(self, obs_shape, d_model: int):
        super().__init__()
        if len(obs_shape) == 1:
            # Vector observations
            self.kind = "mlp"
            self.enc = nn.Sequential(
                nn.Linear(obs_shape[0], d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
        else:
            # Image observations
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
            self.proj = nn.Sequential(
                nn.Linear(flat, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "mlp":
            return self.enc(x)
        z = self.enc(x)
        z = z.view(z.size(0), -1)
        return self.proj(z)


# ============================================================
# 4. RL wrapper: DecisionTransformer
#     – this class has the same interface as before,
#       but internally it uses DTBackbone.
# ============================================================

class DecisionTransformer(nn.Module):
    """
    Decision Transformer for discrete-action environments (CartPole, Atari).

    This wrapper:
      - encodes observations (vector / image) into dimension d_model,
      - maps them to 'states' for DTBackbone,
      - uses your DT implementation to predict actions.
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
        max_ep_len: int = 2048,  # formerly embed_t.num_embeddings
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_actions = n_actions
        self.max_ep_len = max_ep_len

        # Observation encoder (vector or image)
        self.obs_enc = ObsEncoder(obs_shape, d_model)

        # DTBackbone configuration (discrete actions)
        config = DecisionTransformerConfig(
            n_layer=n_layers,
            n_head=n_heads,
            n_embd=d_model,
            dropout=p_drop,
            bias=False,
            K=seq_len,
            max_ep_len=max_ep_len,
            state_dim=d_model,          # state = observation embedding
            act_dim=1,                  # unused when act_discrete=True
            act_discrete=True,
            act_vocab_size=n_actions,
            act_tanh=False,
            tanh_embeddings=False,
        )
        self.dt = DTBackbone(config)

        # History buffers for act()
        self.reset_history()

    # ----------------- Forward (training) ---------------------
    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rtg: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            obs:       [B, L, D] or [B, L, C, H, W]
            actions:   [B, L] (with -1 as padding)
            rtg:       [B, L, 1]
            timesteps: [B, L]

        Returns:
            logits: [B, L, n_actions]
        """
        B, L = actions.shape
        device = obs.device

        # --- encode observations -> states [B, L, d_model] ---
        if obs.dim() == 5:
            # Images: [B, L, C, H, W] -> [B*L, C, H, W]
            B, L, C, H, W = obs.shape
            obs_flat = obs.view(B * L, C, H, W)
            s_tok = self.obs_enc(obs_flat).view(B, L, -1)
        else:
            # Vectors: [B, L, D]
            obs_flat = obs.view(B * L, -1)
            s_tok = self.obs_enc(obs_flat).view(B, L, -1)

        # --- actions: clamp -1 -> 0 for embeddings, but targets remain -1 ---
        actions = actions.to(device)
        actions_for_embed = torch.clamp(actions, min=0)

        # --- rtg and timesteps ---
        rtg = rtg.to(device)
        timesteps = timesteps.to(device).long()
        timesteps = torch.clamp(timesteps, max=self.max_ep_len - 1)

        # attention mask – currently full (no special padding mask)
        attn_mask = torch.ones(B, L, dtype=torch.bool, device=device)

        # call DTBackbone in "training" mode to get logits for all L
        logits, _ = self.dt(
            states=s_tok,
            actions=actions_for_embed,
            rtgs=rtg,
            tsteps=timesteps,
            attn_mask=attn_mask,
            targets=actions,   # CE inside will use ignore_index=-1, but we ignore the returned loss
        )
        # logits: [B, L, n_actions]
        return logits

    # ========================================================
    # 5. History management & act() – almost unchanged
    # ========================================================

    def reset_history(self):
        """Reset internal history buffers used by act()."""
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
        """
        self.eval()

        # 1) real device taken from model parameters
        param_device = next(self.parameters()).device
        device = torch.device(param_device)

        # 2) obs -> tensor on the proper device
        if isinstance(obs, (list, tuple)):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        elif isinstance(obs, torch.Tensor):
            obs_t = obs.to(device=device, dtype=torch.float32)
        else:
            obs_arr = np.asarray(obs, dtype=np.float32)
            obs_t = torch.from_numpy(obs_arr).to(device=device)

        # ---- update history ----
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
        t_seq = torch.clamp(t_seq, max=self.max_ep_len - 1)

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
