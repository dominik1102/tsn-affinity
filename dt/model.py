from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Transformer building blocks (nanoGPT-style)
# =============================================================================

class LayerNorm(nn.Module):
    """LayerNorm with optional bias (PyTorch LayerNorm always has bias)."""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with an optional token validity mask.

    token_mask: bool [B, T] where True means "token exists".
    We apply it as a KEY mask (padded tokens can't be attended),
    combined with a causal (lower-triangular) mask.
    """

    def __init__(self, n_embd: int, n_head: int, dropout: float, bias: bool, block_size: int):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = int(n_head)
        self.n_embd = int(n_embd)
        self.dropout = float(dropout)

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # causal mask [1,1,block,block]
        self.register_buffer(
            "causal",
            torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)).view(1, 1, block_size, block_size),
            persistent=False,
        )

        self.flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor, token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, T, C]
        token_mask: bool [B, T] (True = valid token)
        """
        B, T, C = x.shape
        head_dim = C // self.n_head

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)  # [B, nh, T, hs]
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        causal = self.causal[:, :, :T, :T]  # [1,1,T,T]

        if token_mask is None:
            keep = causal.expand(B, 1, T, T)
        else:
            token_mask = token_mask.to(dtype=torch.bool)
            key_keep = token_mask.view(B, 1, 1, T)  # [B,1,1,T]
            keep = (key_keep & causal).expand(B, 1, T, T)  # [B,1,T,T]

        if self.flash:
            # additive mask: 0 for keep, -inf for masked
            attn_bias = torch.zeros((B, 1, T, T), device=x.device, dtype=torch.float32)
            attn_bias = attn_bias.masked_fill(~keep, -1e4)
            dropout_p = self.dropout if self.training else 0.0
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=dropout_p)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))  # [B,nh,T,T]
            att = att.masked_fill(~keep, -1e4)
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd: int, bias: bool, dropout: float):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, bias: bool, block_size: int):
        super().__init__()
        self.ln1 = LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, bias, block_size)
        self.ln2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, bias, dropout)

    def forward(self, x: torch.Tensor, token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_mask=token_mask)
        x = x + self.mlp(self.ln2(x))
        return x


# =============================================================================
# DT backbone
# =============================================================================

@dataclass
class DecisionTransformerConfig:
    # transformer
    n_layer: int = 3
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = False

    # DT specifics
    K: int = 20
    max_ep_len: int = 10000
    state_dim: int = 128
    act_vocab_size: int = 6


class DTBackbone(nn.Module):
    """
    Discrete-action DT backbone over interleaved tokens:
        (R_1, s_1, a_1, R_2, s_2, a_2, ...)

    Predict actions from STATE token positions: logits at s_t predict a_t.
    Dzięki temu: state token s_t NIE widzi a_t (bo a_t jest "w przyszłości").
    """

    def __init__(self, cfg: DecisionTransformerConfig):
        super().__init__()
        self.cfg = cfg
        print(f"DTBackbone config: {cfg}")

        block_size = cfg.K * 3

        self.te = nn.Embedding(cfg.max_ep_len, cfg.n_embd)
        self.re = nn.Linear(1, cfg.n_embd, bias=cfg.bias)
        self.se = nn.Linear(cfg.state_dim, cfg.n_embd, bias=cfg.bias)
        self.ae = nn.Embedding(cfg.act_vocab_size, cfg.n_embd)

        self.drop = nn.Dropout(cfg.dropout)
        self.h = nn.ModuleList(
            [Block(cfg.n_embd, cfg.n_head, cfg.dropout, cfg.bias, block_size) for _ in range(cfg.n_layer)]
        )
        self.ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.ln_e = LayerNorm(cfg.n_embd, bias=cfg.bias)

        self.act_head = nn.Linear(cfg.n_embd, cfg.act_vocab_size, bias=True)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

        print("DTBackbone parameters: %.2fM" % (self.num_parameters() / 1e6))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        states: torch.Tensor,                           # [B, T, state_dim]
        actions: torch.Tensor,                          # [B, T] int64 (dummy ok for last)
        rtgs: torch.Tensor,                             # [B, T, 1]
        timesteps: torch.Tensor,                        # [B, T] int64
        attention_mask: Optional[torch.Tensor] = None,   # [B, T] bool
    ) -> torch.Tensor:
        B, T, _ = states.shape
        assert T <= self.cfg.K, f"T={T} exceeds context K={self.cfg.K}"

        timesteps = timesteps.clamp_(0, self.cfg.max_ep_len - 1)
        t_emb = self.te(timesteps)  # [B,T,C]

        s_emb = self.se(states) + t_emb
        a_emb = self.ae(actions) + t_emb
        r_emb = self.re(rtgs) + t_emb

        # tokens: [R_1, s_1, a_1, R_2, s_2, a_2, ...]
        x = torch.stack((r_emb, s_emb, a_emb), dim=1)                # [B,3,T,C]
        x = x.permute(0, 2, 1, 3).reshape(B, 3 * T, self.cfg.n_embd)  # [B,3T,C]
        x = self.ln_e(x)

        if attention_mask is None:
            attention_mask = torch.ones((B, T), dtype=torch.bool, device=states.device)
        else:
            attention_mask = attention_mask.to(dtype=torch.bool, device=states.device)

        token_mask = torch.stack((attention_mask, attention_mask, attention_mask), dim=1)  # [B,3,T]
        token_mask = token_mask.permute(0, 2, 1).reshape(B, 3 * T)                         # [B,3T]

        x = self.drop(x)
        for block in self.h:
            x = block(x, token_mask=token_mask)
        x = self.ln_f(x)

        logits_all = self.act_head(x)     # [B,3T,A]
        logits = logits_all[:, 1::3, :]   # state tokens -> [B,T,A]
        return logits


# =============================================================================
# Observation encoder
# =============================================================================

class ObsEncoder(nn.Module):
    """Encodes observations into d_model (vector MLP or Atari CNN)."""

    def __init__(self, obs_shape, d_model: int):
        super().__init__()
        self.obs_shape = tuple(obs_shape)
        self.d_model = int(d_model)

        if len(self.obs_shape) == 1:
            self.kind = "mlp"
            self.mlp = nn.Sequential(
                nn.Linear(self.obs_shape[0], d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
        elif len(self.obs_shape) == 3:
            self.kind = "cnn"
            c, h, w = self.obs_shape
            # DQN-ish CNN (ReLU działa stabilnie na Atari)
            self.cnn = nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, c, h, w)
                z = self.cnn(dummy)
                flat = int(z.view(1, -1).shape[1])
            self.proj = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat, d_model),
                nn.ReLU(),
                nn.LayerNorm(d_model),
            )
        else:
            raise ValueError(f"Unsupported obs_shape={self.obs_shape}. Expected [D] or [C,H,W].")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "mlp":
            return self.mlp(x)
        z = self.cnn(x)
        z = self.proj(z)
        return z


# =============================================================================
# Public wrapper
# =============================================================================

class DecisionTransformer(nn.Module):
    """
    Discrete-action Decision Transformer wrapper:

      obs -> ObsEncoder -> DTBackbone -> action logits

    forward(): offline training
    act(): online rollout with rolling context
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
        max_ep_len: int = 10000,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.n_actions = int(n_actions)
        self.max_ep_len = int(max_ep_len)

        self.obs_enc = ObsEncoder(obs_shape, d_model)

        cfg = DecisionTransformerConfig(
            n_layer=int(n_layers),
            n_head=int(n_heads),
            n_embd=int(d_model),
            dropout=float(p_drop),
            bias=False,
            K=int(seq_len),
            max_ep_len=int(max_ep_len),
            state_dim=int(d_model),
            act_vocab_size=int(n_actions),
        )
        self.dt = DTBackbone(cfg)

        self.reset_history()

    # ----------------------------- normalization -----------------------------

    @staticmethod
    def _normalize_obs_np(obs: np.ndarray) -> np.ndarray:
        if obs.dtype == np.uint8:
            return obs.astype(np.float32) / 255.0
        x = obs.astype(np.float32, copy=False)
        if x.size > 0 and float(np.max(x)) > 1.5:
            x = x / 255.0
        return x

    @staticmethod
    def _normalize_obs_tensor(obs: torch.Tensor) -> torch.Tensor:
        if obs.dtype == torch.uint8:
            return obs.float() / 255.0
        x = obs.float()
        if x.numel() > 0 and float(x.max().item()) > 1.5:
            x = x / 255.0
        return x

    # ----------------------------- offline forward -----------------------------

    def forward(
        self,
        obs: torch.Tensor,                            # [B,L,C,H,W] or [B,L,D]
        actions: torch.Tensor,                        # [B,L] (padded -1)
        rtg: torch.Tensor,                            # [B,L,1]
        timesteps: torch.Tensor,                      # [B,L]
        attention_mask: Optional[torch.Tensor] = None  # [B,L] bool
    ) -> torch.Tensor:
        B, L = actions.shape
        device = obs.device

        # normalize
        if obs.dim() == 5:
            obs = self._normalize_obs_tensor(obs)
        elif obs.dim() == 3:
            obs = obs.float()
        else:
            raise ValueError(f"Unexpected obs shape: {tuple(obs.shape)}")

        # encode obs -> states [B,L,d_model]
        if obs.dim() == 5:
            _, _, C, H, W = obs.shape
            obs_flat = obs.view(B * L, C, H, W)
            s = self.obs_enc(obs_flat).view(B, L, -1)
        else:
            obs_flat = obs.view(B * L, -1)
            s = self.obs_enc(obs_flat).view(B, L, -1)

        actions = actions.to(device=device, dtype=torch.long)
        actions_for_embed = torch.clamp(actions, min=0)

        timesteps = timesteps.to(device=device, dtype=torch.long).clamp_(0, self.max_ep_len - 1)
        rtg = rtg.to(device=device, dtype=torch.float32)

        if attention_mask is None:
            attention_mask = torch.ones((B, L), dtype=torch.bool, device=device)
        else:
            attention_mask = attention_mask.to(device=device, dtype=torch.bool)

        logits = self.dt(
            states=s,
            actions=actions_for_embed,
            rtgs=rtg,
            timesteps=timesteps,
            attention_mask=attention_mask,
        )
        return logits

    # ----------------------------- rollout API -----------------------------

    def reset_history(self) -> None:
        self._hist_obs: List[torch.Tensor] = []
        self._hist_actions: List[int] = []
        self._hist_rtgs: List[float] = []
        self._hist_t: List[int] = []

    def _trim_history(self) -> None:
        if len(self._hist_obs) <= self.seq_len:
            return
        overflow = len(self._hist_obs) - self.seq_len
        self._hist_obs = self._hist_obs[overflow:]
        self._hist_rtgs = self._hist_rtgs[overflow:]
        self._hist_t = self._hist_t[overflow:]
        if overflow > 0 and len(self._hist_actions) > 0:
            self._hist_actions = self._hist_actions[overflow:]
        if len(self._hist_actions) > self.seq_len - 1:
            self._hist_actions = self._hist_actions[-(self.seq_len - 1):]

    def prime_history(self, obs, action: int, rtg_scalar: float, t: int) -> None:
        """
        Dodaj (state, action) do historii bez odpalania policy.
        Przydaje się jeśli wymuszasz FIRE po reset.
        """
        device_t = next(self.parameters()).device

        if isinstance(obs, torch.Tensor):
            obs_t = self._normalize_obs_tensor(obs.to(device_t))
        else:
            obs_arr = self._normalize_obs_np(np.asarray(obs))
            obs_t = torch.from_numpy(obs_arr).to(device_t, dtype=torch.float32)

        if obs_t.dim() not in (1, 3):
            raise ValueError(f"prime_history: unsupported obs shape {tuple(obs_t.shape)}")

        self._hist_obs.append(obs_t)
        self._hist_rtgs.append(float(rtg_scalar))
        self._hist_t.append(int(t))
        self._trim_history()

        self._hist_actions.append(int(action))
        if len(self._hist_actions) > self.seq_len:
            self._hist_actions = self._hist_actions[-self.seq_len:]

    @torch.no_grad()
    def act(
        self,
        obs: Union[np.ndarray, torch.Tensor, List[float]],
        rtg_scalar: float,
        t: int,
        prev_action: int = 0,   # kompatybilność
        device: str = "cpu",    # kompatybilność
        n_actions: Optional[int] = None,
    ) -> int:
        self.eval()
        device_t = next(self.parameters()).device

        if isinstance(obs, torch.Tensor):
            obs_t = self._normalize_obs_tensor(obs.to(device_t))
        else:
            obs_arr = self._normalize_obs_np(np.asarray(obs))
            obs_t = torch.from_numpy(obs_arr).to(device_t, dtype=torch.float32)

        if obs_t.dim() not in (1, 3):
            raise ValueError(f"act(): unsupported obs shape {tuple(obs_t.shape)}")

        # append current state
        self._hist_obs.append(obs_t)
        self._hist_rtgs.append(float(rtg_scalar))
        self._hist_t.append(int(t))
        self._trim_history()

        L = len(self._hist_obs)

        # past actions (for states 1..L-1)
        past = self._hist_actions[-(L - 1):] if L > 1 else []
        actions_seq = past + [0]  # dummy for current step (state token doesn't see it)

        if len(actions_seq) != L:
            actions_seq = (actions_seq + [0] * L)[:L]

        actions_t = torch.tensor(actions_seq, device=device_t, dtype=torch.long).unsqueeze(0)  # [1,L]
        rtg_t = torch.tensor(self._hist_rtgs, device=device_t, dtype=torch.float32).view(1, L, 1)
        ts_t = torch.tensor(self._hist_t, device=device_t, dtype=torch.long).clamp_(0, self.max_ep_len - 1).view(1, L)
        mask_t = torch.ones((1, L), device=device_t, dtype=torch.bool)

        if obs_t.dim() == 1:
            obs_batch = torch.stack(self._hist_obs, dim=0).unsqueeze(0)  # [1,L,D]
        else:
            obs_batch = torch.stack(self._hist_obs, dim=0).unsqueeze(0)  # [1,L,C,H,W]

        logits = self.forward(obs_batch, actions_t, rtg_t, ts_t, attention_mask=mask_t)  # [1,L,A]
        logits_last = logits[:, -1, :]

        if n_actions is not None:
            logits_last = logits_last.clone()
            logits_last[..., int(n_actions):] = -1e9

        action = int(torch.argmax(logits_last, dim=-1).item())

        self._hist_actions.append(action)
        if len(self._hist_actions) > self.seq_len:
            self._hist_actions = self._hist_actions[-self.seq_len:]

        return action
