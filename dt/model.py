from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Transformer blocks (GPT-style, causal)
# ============================================================

class LayerNorm(nn.Module):
    """LayerNorm with optional bias (PyTorch LayerNorm always has bias)."""
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, bias: bool, block_size: int):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.dropout = dropout

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # for manual attention fallback
        self.register_buffer(
            "causal",
            torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)).view(1, 1, block_size, block_size),
            persistent=False,
        )

        self.has_sdp = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, T, C]
        attn_mask: bool [B, T] where True=keep token (key padding mask)
        """
        B, T, C = x.shape

        q, k, v = self.c_attn(x).split(C, dim=-1)  # each [B,T,C]

        # -> [B, nh, T, hs]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        dropout_p = self.dropout if self.training else 0.0

        if self.has_sdp:
            # scaled_dot_product_attention supports is_causal=True
            # We pass key padding mask as attn_mask broadcastable to [B, nh, T, T].
            float_mask = None
            if attn_mask is not None:
                if attn_mask.dim() != 2:
                    raise ValueError(f"attn_mask must be [B,T] bool, got {attn_mask.shape}")
                keep = attn_mask[:, None, None, :]  # [B,1,1,T]
                neg = torch.finfo(q.dtype).min  # skończone, nie -inf
                float_mask = torch.zeros((B, 1, 1, T), device=x.device, dtype=q.dtype)
                float_mask = float_mask.masked_fill(~keep, neg)

            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=float_mask,
                dropout_p=dropout_p,
                is_causal=True,
            )
        else:
            # manual attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # [B,nh,T,T]
            causal = self.causal[:, :, :T, :T]  # [1,1,T,T]

            if attn_mask is not None:
                keep = attn_mask[:, None, None, :]  # [B,1,1,T]
                keep = keep & causal                # -> [B,1,T,T]
            else:
                keep = causal

            att = att.masked_fill(~keep, -1e4)
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # [B,nh,T,hs]

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

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


# ============================================================
# Decision Transformer backbone
# ============================================================

class DTBackbone(nn.Module):
    """
    Tokens: (R_0, s_0, a_0, R_1, s_1, a_1, ...)
    Output is read at state-token positions -> shape:
      - discrete:   [B, T, act_vocab_size] logits
      - continuous: [B, T, act_dim]        predicted actions
    """

    def __init__(self, cfg: DTConfig):
        super().__init__()
        self.cfg = cfg
        print(f"DTBackbone config: {cfg}")

        block_size = cfg.K * 3

        self.te = nn.Embedding(cfg.max_ep_len, cfg.n_embd)
        self.re = nn.Linear(1, cfg.n_embd, bias=cfg.bias)
        self.se = nn.Linear(cfg.state_dim, cfg.n_embd, bias=cfg.bias)

        # action embedding + head depend on discrete/continuous
        if cfg.act_discrete:
            self.ae = nn.Embedding(cfg.act_vocab_size, cfg.n_embd)
            self.act_head = nn.Linear(cfg.n_embd, cfg.act_vocab_size, bias=True)
        else:
            self.ae = nn.Linear(cfg.act_dim, cfg.n_embd, bias=cfg.bias)
            self.act_head = nn.Linear(cfg.n_embd, cfg.act_dim, bias=True)

        self.drop = nn.Dropout(cfg.dropout)
        self.h = nn.ModuleList(
            [Block(cfg.n_embd, cfg.n_head, cfg.dropout, cfg.bias, block_size) for _ in range(cfg.n_layer)]
        )
        self.ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.ln_e = LayerNorm(cfg.n_embd, bias=cfg.bias)

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
        states: torch.Tensor,                    # [B,T,state_dim]
        actions: torch.Tensor,                   # discrete: [B,T] long   | continuous: [B,T,act_dim] float
        rtgs: torch.Tensor,                      # [B,T,1]
        timesteps: Optional[torch.Tensor] = None,# [B,T] long
        attention_mask: Optional[torch.Tensor] = None, # [B,T] bool (True=valid)
        *,
        # ---- aliases for Panda code ----
        tsteps: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        # ---- optional loss ----
        targets: Optional[torch.Tensor] = None,
    ):
        # accept Panda keyword aliases
        if timesteps is None:
            timesteps = tsteps
        if attention_mask is None:
            attention_mask = attn_mask
        if timesteps is None:
            raise ValueError("DTBackbone.forward: timesteps (or tsteps) must be provided")

        B, T, _ = states.shape
        if T > self.cfg.K:
            raise ValueError(f"T={T} exceeds K={self.cfg.K}")

        timesteps = timesteps.to(device=states.device, dtype=torch.long)
        timesteps = torch.clamp(timesteps, 0, self.cfg.max_ep_len - 1)
        t_emb = self.te(timesteps)  # [B,T,E]

        s_emb = self.se(states) + t_emb
        r_emb = self.re(rtgs) + t_emb

        if self.cfg.act_discrete:
            # actions: [B,T] long (already clamped >=0 outside or here)
            a = actions.to(device=states.device, dtype=torch.long)
            a = torch.clamp(a, min=0)
            a_emb = self.ae(a) + t_emb
        else:
            # actions: [B,T,act_dim] float
            a = actions.to(device=states.device, dtype=torch.float32)
            if self.cfg.tanh_embeddings:
                a = torch.tanh(a)
            a_emb = self.ae(a) + t_emb  # Linear -> [B,T,E]

        # interleave => [B, 3T, E]
        x = torch.stack((r_emb, s_emb, a_emb), dim=2)  # [B,T,3,E]
        x = x.reshape(B, 3 * T, self.cfg.n_embd)
        x = self.ln_e(x)

        if attention_mask is None:
            attention_mask = torch.ones((B, T), dtype=torch.bool, device=states.device)
        else:
            attention_mask = attention_mask.to(device=states.device, dtype=torch.bool)

        # token-level mask => [B,3T]
        tok_mask = attention_mask[:, :, None].expand(B, T, 3).reshape(B, 3 * T)

        x = self.drop(x)
        for block in self.h:
            x = block(x, attn_mask=tok_mask)
        x = self.ln_f(x)

        out_all = self.act_head(x)          # [B,3T,A] or [B,3T,act_dim]
        out = out_all[:, 1::3, ...]         # state tokens => [B,T,*]

        # continuous: optional tanh on output
        if (not self.cfg.act_discrete) and self.cfg.act_tanh:
            out = torch.tanh(out)

        if targets is None:
            return out

        # compute loss if targets provided (optional)
        if self.cfg.act_discrete:
            # targets expected: [B,T] long, may include -1 for padding
            targ = targets.to(device=states.device, dtype=torch.long)
            A = out.size(-1)
            loss = F.cross_entropy(
                out.reshape(-1, A),
                targ.reshape(-1),
                ignore_index=-1,
                reduction="mean",
            )
            return out, loss
        else:
            # targets expected: [B,T,act_dim] float
            targ = targets.to(device=states.device, dtype=torch.float32)
            mse = ((out - targ) ** 2).mean(dim=-1)  # [B,T]
            m = attention_mask.float()
            denom = m.sum().clamp(min=1.0)
            loss = (mse * m).sum() / denom
            return out, loss


# ============================================================
# Obs encoder
# ============================================================

class ObsEncoder(nn.Module):
    def __init__(self, obs_shape, d_model: int):
        super().__init__()
        self.obs_shape = tuple(obs_shape)
        self.d_model = int(d_model)

        if len(self.obs_shape) == 1:
            self.kind = "mlp"
            self.net = nn.Sequential(
                nn.Linear(self.obs_shape[0], d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
        elif len(self.obs_shape) == 3:
            self.kind = "cnn"
            c, h, w = self.obs_shape
            self.cnn = nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=8, stride=4),
                nn.GELU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.GELU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.GELU(),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, c, h, w)
                z = self.cnn(dummy)
                flat = int(z.view(1, -1).shape[1])
            self.proj = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
        else:
            raise ValueError(f"Unsupported obs_shape={self.obs_shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "mlp":
            return self.net(x)
        z = self.cnn(x)
        return self.proj(z)

@dataclass
class DTConfig:
    n_layer: int = 3
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = False

    K: int = 20
    max_ep_len: int = 10000
    state_dim: int = 128

    # ---- actions ----
    # discrete Atari:
    act_discrete: bool = True
    act_vocab_size: int = 6

    # continuous Panda:
    act_dim: int = 1
    act_tanh: bool = False          # apply tanh on output actions
    tanh_embeddings: bool = False   # tanh() on action inputs before embedding



# ============================================================
# Public DT model
# ============================================================

class DecisionTransformer(nn.Module):
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
        rtg_scale: float = 1000.0,
    ):

        super().__init__()
        self.seq_len = int(seq_len)
        self.n_actions = int(n_actions)
        self.max_ep_len = int(max_ep_len)
        self.rtg_scale = float(rtg_scale)
        if not (self.rtg_scale > 0.0):
            raise ValueError(f"rtg_scale must be > 0, got {self.rtg_scale}")


        self.obs_enc = ObsEncoder(obs_shape, d_model)

        cfg = DTConfig(
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

    def forward(
            self,
            obs: torch.Tensor,  # [B,L,C,H,W] or [B,L,D]
            actions: torch.Tensor,  # [B,L] with -1 padding allowed
            rtg: torch.Tensor,  # [B,L,1]
            timesteps: torch.Tensor,  # [B,L]
            attention_mask: Optional[torch.Tensor] = None,  # [B,L] bool
    ) -> torch.Tensor:
        B, L = actions.shape
        device = obs.device

        # ----------------------------------------------------
        # Normalize observations (consistent between train and eval)
        # - if input is uint8 in 0..255 -> divide by 255
        # - if float but still in 0..255 -> divide by 255
        # ----------------------------------------------------
        orig_dtype = obs.dtype
        obs = obs.to(device=device, dtype=torch.float32)

        if orig_dtype == torch.uint8:
            obs = obs / 255.0
        else:
            # float/half/etc: heuristically detect 0..255 range
            if obs.numel() > 0 and float(obs.max().item()) > 1.5:
                obs = obs / 255.0

        # Encode observations => states [B,L,d_model]
        if obs.dim() == 5:
            B2, L2, C, H, W = obs.shape
            assert B2 == B and L2 == L
            obs_flat = obs.view(B * L, C, H, W)
            s = self.obs_enc(obs_flat).view(B, L, -1)
        elif obs.dim() == 3:
            obs_flat = obs.view(B * L, -1)
            s = self.obs_enc(obs_flat).view(B, L, -1)
        else:
            raise ValueError(f"Unexpected obs shape: {obs.shape}")

        actions = actions.to(device=device, dtype=torch.long)
        actions_for_embed = torch.clamp(actions, min=0)

        timesteps = timesteps.to(device=device, dtype=torch.long)
        timesteps = torch.clamp(timesteps, 0, self.max_ep_len - 1)

        # RTG scaling (inside the model so train and eval are always consistent)
        rtg = rtg.to(device=device, dtype=torch.float32)
        if self.rtg_scale != 1.0:
            rtg = rtg / self.rtg_scale

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

    # ------------------------
    # Inference history
    # ------------------------
    def reset_history(self) -> None:
        self._hist_obs: List[torch.Tensor] = []
        self._hist_actions: List[int] = []
        self._hist_rtgs: List[float] = []
        self._hist_t: List[int] = []

    @staticmethod
    def _to_tensor_obs(obs: Union[np.ndarray, torch.Tensor], device: torch.device) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            x = obs.to(device=device, dtype=torch.float32)
        else:
            x = torch.from_numpy(np.asarray(obs)).to(device=device, dtype=torch.float32)

        # If looks like 0..255, normalize
        if x.numel() > 0 and float(x.max().item()) > 1.5:
            x = x / 255.0
        return x

    @torch.no_grad()
    def act(
        self,
        obs: Union[np.ndarray, torch.Tensor],  # CHW float in [0,1] (or uint8)
        rtg_scalar: float,
        t: int,
        device: str = "cpu",
        n_actions: Optional[int] = None,
    ) -> int:
        self.eval()
        device_t = next(self.parameters()).device

        obs_t = self._to_tensor_obs(obs, device_t)
        if obs_t.dim() not in (1, 3):
            raise ValueError(f"act(): expected obs dim 1 or 3, got {tuple(obs_t.shape)}")

        self._hist_obs.append(obs_t)
        self._hist_rtgs.append(float(rtg_scalar))
        self._hist_t.append(int(t))

        # keep last K
        while len(self._hist_obs) > self.seq_len:
            self._hist_obs.pop(0)
            self._hist_rtgs.pop(0)
            self._hist_t.pop(0)
            if self._hist_actions:
                self._hist_actions.pop(0)

        L = len(self._hist_obs)

        # actions sequence = past actions + dummy for current step
        past = list(self._hist_actions)
        if len(past) > L - 1:
            past = past[-(L - 1):]
        actions_seq = past + [0]
        assert len(actions_seq) == L

        actions_t = torch.tensor(actions_seq, device=device_t, dtype=torch.long).unsqueeze(0)  # [1,L]
        rtg_t = torch.tensor(self._hist_rtgs, device=device_t, dtype=torch.float32).view(1, L, 1)
        ts_t = torch.tensor(self._hist_t, device=device_t, dtype=torch.long).view(1, L)
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
        while len(self._hist_actions) > self.seq_len:
            self._hist_actions.pop(0)

        return action
