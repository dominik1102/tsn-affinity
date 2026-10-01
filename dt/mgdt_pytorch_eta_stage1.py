
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MGDTConfig:
    """
    Stage-1 etaoxing-style MGDT patch.

    Main changes vs our previous PyTorch MGDT:
      - default in_channels=1 instead of 4
      - explicit learned image_pos_embed over patches
      - same global return/action tokenization API as current code
    """
    image_size: Tuple[int, int] = (84, 84)
    in_channels: int = 1
    patch_size: int = 14
    d_model: int = 512
    n_head: int = 8
    n_layer: int = 8
    dropout: float = 0.1
    max_steps: int = 4

    num_actions: int = 18
    reward_values: Tuple[int, ...] = (-1, 0, 1)
    return_range: Tuple[int, int] = (-20, 101)

    predict_reward: bool = True
    single_return_token: bool = False
    use_spatial_tokens: bool = True
    use_image_pos_embed: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        h, w = self.image_size
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

    @property
    def num_patches(self) -> int:
        h, w = self.image_size
        return (h // self.patch_size) * (w // self.patch_size)

    @property
    def num_rewards(self) -> int:
        return len(self.reward_values)

    @property
    def num_returns(self) -> int:
        low, high = self.return_range
        return int(high - low)

    @property
    def tokens_per_step(self) -> int:
        base = self.num_patches if self.use_spatial_tokens else 1
        return base + (3 if self.predict_reward else 2)

    @property
    def max_tokens(self) -> int:
        return self.max_steps * self.tokens_per_step


def encode_returns(rtg: torch.Tensor, return_range: Tuple[int, int]) -> torch.Tensor:
    low, high = return_range
    x = rtg.round().long()
    x = x.clamp(min=low, max=high - 1)
    return x - low


class RewardTokenizer:
    def __init__(self, reward_values: Tuple[int, ...] = (-1, 0, 1)):
        self.reward_values = tuple(int(v) for v in reward_values)
        self.index: Dict[int, int] = {v: i for i, v in enumerate(self.reward_values)}

    def encode(self, rewards: torch.Tensor) -> torch.Tensor:
        x = rewards.round().long()
        out = torch.empty_like(x)
        default_idx = self.index.get(0, 0)
        out.fill_(default_idx)
        for v, i in self.index.items():
            out[x == v] = i
        if len(self.reward_values) > 0:
            values = torch.tensor(self.reward_values, device=x.device, dtype=torch.long)
            unseen = torch.ones_like(x, dtype=torch.bool)
            for v in self.reward_values:
                unseen &= (x != v)
            if unseen.any():
                x_u = x[unseen].unsqueeze(-1)
                nearest = torch.argmin((x_u - values.view(1, -1)).abs(), dim=-1)
                out[unseen] = nearest
        return out


class PatchEmbed(nn.Module):
    """
    Input:  [B, T, C, H, W]
    Output: [B, T, P, D]
    """
    def __init__(self, cfg: MGDTConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Conv2d(
            in_channels=cfg.in_channels,
            out_channels=cfg.d_model,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
            bias=True,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = obs.shape
        x = obs.float()
        if x.numel() and x.max() > 1.5:
            x = x / 255.0
        x = x.view(b * t, c, h, w)
        x = self.proj(x)                   # [B*T, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)   # [B*T, P, D]
        x = x.view(b, t, x.shape[1], x.shape[2])
        return x


class MGDTBlock(nn.Module):
    def __init__(self, cfg: MGDTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_head,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.ln_2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        h = self.ln_1(x)
        h, _ = self.attn(
            h, h, h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop(h)
        x = x + self.mlp(self.ln_2(x))
        return x


class MultiGameDecisionTransformer(nn.Module):
    """
    Stage-1 faithful patch:
      - patchify grayscale C=1 observations
      - add image_pos_embed like etaoxing / official JAX code
      - keep current return/action/reward heads and loss API
    """
    def __init__(self, cfg: MGDTConfig):
        super().__init__()
        self.cfg = cfg
        self.reward_tokenizer = RewardTokenizer(cfg.reward_values)

        self.obs_embed = PatchEmbed(cfg)
        self.return_embed = nn.Embedding(cfg.num_returns, cfg.d_model)
        self.action_embed = nn.Embedding(cfg.num_actions, cfg.d_model)
        self.reward_embed = nn.Embedding(cfg.num_rewards, cfg.d_model) if cfg.predict_reward else None

        self.image_pos_embed = nn.Parameter(torch.zeros(1, 1, cfg.num_patches, cfg.d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.max_tokens, cfg.d_model))
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([MGDTBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        self.return_head = nn.Linear(cfg.d_model, cfg.num_returns, bias=False)
        self.action_head = nn.Linear(cfg.d_model, cfg.num_actions, bias=False)
        self.reward_head = nn.Linear(cfg.d_model, cfg.num_rewards, bias=False) if cfg.predict_reward else None

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=5 ** 0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        if hasattr(self, "image_pos_embed"):
            nn.init.normal_(self.image_pos_embed, mean=0.0, std=0.02)
        if hasattr(self, "pos_embed"):
            nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def _embed_inputs(
        self,
        observations: torch.Tensor,
        returns_to_go: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        obs_emb = self.obs_embed(observations)  # [B,T,P,D]
        b, t, p, d = obs_emb.shape

        if self.cfg.use_image_pos_embed:
            obs_emb = obs_emb + self.image_pos_embed[:, :, :p, :]

        num_obs_tokens = p if self.cfg.use_spatial_tokens else 1
        if not self.cfg.use_spatial_tokens:
            obs_emb = obs_emb.mean(dim=2, keepdim=True)
            p = 1

        ret_tok = encode_returns(returns_to_go, self.cfg.return_range)
        ret_emb = self.return_embed(ret_tok).unsqueeze(2)
        act_emb = self.action_embed(actions.long().clamp(min=0)).unsqueeze(2)

        chunks = [obs_emb, ret_emb, act_emb]
        if self.cfg.predict_reward:
            rew_tok = self.reward_tokenizer.encode(rewards)
            rew_emb = self.reward_embed(rew_tok).unsqueeze(2)
            chunks.append(rew_emb)

        token_emb = torch.cat(chunks, dim=2)   # [B,T,K,D]
        token_emb = token_emb.reshape(b, t * token_emb.shape[2], d)
        return token_emb, num_obs_tokens

    def _build_attn_mask(self, steps: int, num_obs_tokens: int, device: torch.device) -> torch.Tensor:
        k = num_obs_tokens + (3 if self.cfg.predict_reward else 2)
        total = steps * k
        mask = torch.ones(total, total, device=device, dtype=torch.bool)

        for tq in range(steps):
            for pq in range(k):
                q = tq * k + pq
                for tk in range(tq):
                    ks = tk * k
                    mask[q, ks:ks + k] = False
                if pq < num_obs_tokens:
                    ks = tq * k
                    mask[q, ks:ks + num_obs_tokens] = False
                else:
                    ks = tq * k
                    mask[q, ks:q + 1] = False
        return mask

    def _build_key_padding_mask(self, valid_steps: Optional[torch.Tensor], num_obs_tokens: int) -> Optional[torch.Tensor]:
        if valid_steps is None:
            return None
        b, t = valid_steps.shape
        k = num_obs_tokens + (3 if self.cfg.predict_reward else 2)
        step_mask = valid_steps.to(dtype=torch.bool).unsqueeze(-1).expand(b, t, k)
        if self.cfg.single_return_token and t > 1:
            ret_idx = num_obs_tokens
            step_mask[:, 1:, ret_idx] = False
        token_mask = step_mask.reshape(b, t * k)
        return ~token_mask

    def _target_positions(self, steps: int, num_obs_tokens: int, device: torch.device) -> Dict[str, torch.Tensor]:
        k = num_obs_tokens + (3 if self.cfg.predict_reward else 2)
        base = torch.arange(steps, device=device) * k
        pos = {
            "return_from": base + (num_obs_tokens - 1),
            "action_from": base + num_obs_tokens,
        }
        if self.cfg.predict_reward:
            pos["reward_from"] = base + num_obs_tokens + 1
        return pos

    def forward(
        self,
        observations: torch.Tensor,
        returns_to_go: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        valid_steps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        token_emb, num_obs_tokens = self._embed_inputs(observations, returns_to_go, actions, rewards)
        b, total, d = token_emb.shape
        if total > self.cfg.max_tokens:
            raise ValueError(f"Sequence too long: got {total} tokens, max_tokens={self.cfg.max_tokens}")

        x = token_emb + self.pos_embed[:, :total]
        x = self.drop(x)

        attn_mask = self._build_attn_mask(observations.shape[1], num_obs_tokens, observations.device)
        key_padding_mask = self._build_key_padding_mask(valid_steps, num_obs_tokens)

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.ln_f(x)

        pos = self._target_positions(observations.shape[1], num_obs_tokens, observations.device)
        ret_h = x[:, pos["return_from"]]
        act_h = x[:, pos["action_from"]]

        out: Dict[str, torch.Tensor] = {
            "return_logits": self.return_head(ret_h),
            "action_logits": self.action_head(act_h),
            "hidden": x,
        }
        if self.cfg.predict_reward:
            rew_h = x[:, pos["reward_from"]]
            out["reward_logits"] = self.reward_head(rew_h)
        return out

    def loss(
        self,
        observations: torch.Tensor,
        returns_to_go: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        valid_steps: Optional[torch.Tensor] = None,
        loss_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Dict[str, torch.Tensor]:
        out = self.forward(observations, returns_to_go, actions, rewards, valid_steps=valid_steps)
        ret_targets = encode_returns(returns_to_go, self.cfg.return_range)
        rew_targets = self.reward_tokenizer.encode(rewards)

        if valid_steps is None:
            valid_steps = torch.ones_like(actions, dtype=torch.bool)

        def masked_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            flat_logits = logits.reshape(-1, logits.shape[-1])
            flat_targets = targets.reshape(-1)
            flat_mask = mask.reshape(-1)
            if flat_mask.any():
                return F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])
            return flat_logits.sum() * 0.0

        ret_mask = valid_steps.clone()
        if self.cfg.single_return_token and ret_mask.shape[1] > 1:
            ret_mask[:, 1:] = False

        ret_loss = masked_ce(out["return_logits"], ret_targets, ret_mask)
        act_loss = masked_ce(out["action_logits"], actions.long(), valid_steps)
        total = loss_weights[0] * ret_loss + loss_weights[1] * act_loss

        rew_loss = None
        if self.cfg.predict_reward:
            rew_loss = masked_ce(out["reward_logits"], rew_targets, valid_steps)
            total = total + loss_weights[2] * rew_loss

        result: Dict[str, torch.Tensor] = {
            "loss": total,
            "return_loss": ret_loss,
            "action_loss": act_loss,
        }
        if rew_loss is not None:
            result["reward_loss"] = rew_loss
        return result


@torch.no_grad()
def _mask_to_top_percentile_logits(
    logits: torch.Tensor,
    top_percentile: Optional[float] = None,
) -> torch.Tensor:
    if top_percentile is None:
        return logits
    p = float(top_percentile)
    if p <= 0.0:
        return logits
    if p > 100.0:
        raise ValueError(f"top_percentile must be in [0,100], got {top_percentile}")

    squeeze = False
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        squeeze = True
    elif logits.dim() != 2:
        raise ValueError(f"Expected logits [B,V] or [V], got {tuple(logits.shape)}")

    filtered = logits.clone()
    work = filtered.to(torch.float32)
    sentinel = -1e8

    for b in range(work.shape[0]):
        row = work[b]
        valid = row > sentinel
        vals = row[valid]
        if vals.numel() == 0:
            continue
        thresh = torch.quantile(vals, q=p / 100.0)
        drop = valid & (row < thresh)
        filtered[b, drop] = -1e9

    return filtered.squeeze(0) if squeeze else filtered


@torch.no_grad()
def sample_expert_return(
    return_logits: torch.Tensor,
    return_range: Tuple[int, int],
    kappa: float = 10.0,
    temperature: float = 1.0,
    top_percentile: Optional[float] = None,
) -> torch.Tensor:
    low, high = return_range
    values = torch.arange(low, high, device=return_logits.device, dtype=torch.float32)
    scaled = return_logits / max(temperature, 1e-6)
    guided = scaled + kappa * (values - float(low)) / max(1.0, float(high - low))
    guided = _mask_to_top_percentile_logits(guided, top_percentile=top_percentile)
    probs = F.softmax(guided, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def sample_action_from_logits(
    action_logits: torch.Tensor,
    greedy: bool = False,
    temperature: float = 1.0,
    top_percentile: Optional[float] = None,
) -> torch.Tensor:
    if greedy:
        return action_logits.argmax(dim=-1)
    scaled = action_logits / max(temperature, 1e-6)
    scaled = _mask_to_top_percentile_logits(scaled, top_percentile=top_percentile)
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


__all__ = [
    "MGDTConfig",
    "MultiGameDecisionTransformer",
    "encode_returns",
    "RewardTokenizer",
    "sample_expert_return",
    "sample_action_from_logits",
]
