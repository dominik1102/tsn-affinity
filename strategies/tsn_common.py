from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


def _rebuild_optimizer_like(old_opt: torch.optim.Optimizer, params) -> torch.optim.Optimizer:
    """
    Re-create an optimizer of the same family with the same hyperparameters.
    """
    if isinstance(old_opt, torch.optim.AdamW):
        pg0 = old_opt.param_groups[0]
        return torch.optim.AdamW(
            params,
            lr=float(pg0.get("lr", 3e-4)),
            weight_decay=float(pg0.get("weight_decay", 0.0)),
            betas=tuple(pg0.get("betas", (0.9, 0.999))),
            eps=float(pg0.get("eps", 1e-8)),
        )
    if isinstance(old_opt, torch.optim.Adam):
        pg0 = old_opt.param_groups[0]
        return torch.optim.Adam(
            params,
            lr=float(pg0.get("lr", 3e-4)),
            weight_decay=float(pg0.get("weight_decay", 0.0)),
            betas=tuple(pg0.get("betas", (0.9, 0.999))),
            eps=float(pg0.get("eps", 1e-8)),
        )
    if isinstance(old_opt, torch.optim.SGD):
        pg0 = old_opt.param_groups[0]
        return torch.optim.SGD(
            params,
            lr=float(pg0.get("lr", 1e-3)),
            weight_decay=float(pg0.get("weight_decay", 0.0)),
            momentum=float(pg0.get("momentum", 0.0)),
            nesterov=bool(pg0.get("nesterov", False)),
        )
    return torch.optim.AdamW(params, lr=3e-4)


class _TopKMaskSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores: torch.Tensor, keep_ratio: float, free_mask: Optional[torch.Tensor]):
        flat_scores = scores.reshape(-1)
        n_all = int(flat_scores.numel())
        if n_all == 0:
            return torch.zeros_like(scores)

        keep_ratio = float(max(0.0, min(1.0, keep_ratio)))

        if free_mask is not None:
            flat_free = free_mask.reshape(-1).to(device=flat_scores.device, dtype=torch.bool)
            n_free = int(flat_free.sum().item())
            if n_free == 0:
                return torch.zeros_like(scores)

            k_keep = max(1, int(round(keep_ratio * n_free)))
            if k_keep >= n_free:
                return flat_free.view_as(scores).to(dtype=scores.dtype)

            work = flat_scores.clone()
            work[~flat_free] = -torch.inf
            topk_vals = torch.topk(work, k=k_keep, largest=True, sorted=True).values
            thr = topk_vals[-1]
            mask = (work >= thr) & flat_free
            return mask.view_as(scores).to(dtype=scores.dtype)

        k_keep = max(1, int(round(keep_ratio * n_all)))
        if k_keep >= n_all:
            return torch.ones_like(scores)
        topk_vals = torch.topk(flat_scores, k=k_keep, largest=True, sorted=True).values
        thr = topk_vals[-1]
        return (scores >= thr).to(dtype=scores.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class _TSNMaskMixin:
    keep_ratio: float
    allow_weight_reuse: bool
    occupied_weight_mask: Optional[torch.Tensor]
    occupied_bias_mask: Optional[torch.Tensor]
    active_weight_mask: Optional[torch.Tensor]
    active_bias_mask: Optional[torch.Tensor]
    weight_mask: Optional[torch.Tensor]
    bias_mask: Optional[torch.Tensor]

    def clear_active_masks(self) -> None:
        self.active_weight_mask = None
        self.active_bias_mask = None

    def clear_occupied_masks(self) -> None:
        self.occupied_weight_mask = None
        self.occupied_bias_mask = None

    def _get_free_weight_mask(self) -> Optional[torch.Tensor]:
        if self.allow_weight_reuse or self.occupied_weight_mask is None:
            return None
        return ~self.occupied_weight_mask.to(device=self.weight.device, dtype=torch.bool)

    def _get_free_bias_mask(self) -> Optional[torch.Tensor]:
        if getattr(self, "bias", None) is None:
            return None
        if self.allow_weight_reuse or self.occupied_bias_mask is None:
            return None
        return ~self.occupied_bias_mask.to(device=self.bias.device, dtype=torch.bool)

    def _current_weight_mask(self) -> torch.Tensor:
        if self.active_weight_mask is not None:
            mask = self.active_weight_mask.to(device=self.weight.device, dtype=self.weight.dtype)
        else:
            mask = _TopKMaskSTE.apply(self.score.abs(), float(self.keep_ratio), self._get_free_weight_mask())
        self.weight_mask = mask.detach().to(dtype=torch.uint8)
        return mask

    def _current_bias_mask(self) -> Optional[torch.Tensor]:
        if getattr(self, "bias", None) is None:
            self.bias_mask = None
            return None
        if self.active_bias_mask is not None:
            mask = self.active_bias_mask.to(device=self.bias.device, dtype=self.bias.dtype)
        else:
            mask = _TopKMaskSTE.apply(self.bias_score.abs(), float(self.keep_ratio), self._get_free_bias_mask())
        self.bias_mask = mask.detach().to(dtype=torch.uint8)
        return mask


class TSNLinear(nn.Linear, _TSNMaskMixin):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        keep_ratio: float = 0.5,
        allow_weight_reuse: bool = False,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.keep_ratio = float(keep_ratio)
        self.allow_weight_reuse = bool(allow_weight_reuse)

        self.score = nn.Parameter(torch.empty_like(self.weight))
        if bias:
            self.bias_score = nn.Parameter(torch.empty_like(self.bias))
        else:
            self.register_parameter("bias_score", None)

        self.weight_mask = None
        self.bias_mask = None
        self.active_weight_mask = None
        self.active_bias_mask = None
        self.occupied_weight_mask = None
        self.occupied_bias_mask = None

        self.reset_scores()

    def reset_scores(self) -> None:
        nn.init.kaiming_uniform_(self.score, a=np.sqrt(5))
        if self.bias_score is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(max(1, fan_in))
            nn.init.uniform_(self.bias_score, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wm = self._current_weight_mask()
        w = self.weight * wm
        if self.bias is None:
            b = None
        else:
            bm = self._current_bias_mask()
            b = self.bias * bm if bm is not None else self.bias
        return F.linear(x, w, b)


class TSNConv2d(nn.Conv2d, _TSNMaskMixin):
    def __init__(self, *args, keep_ratio: float = 0.5, allow_weight_reuse: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_ratio = float(keep_ratio)
        self.allow_weight_reuse = bool(allow_weight_reuse)

        self.score = nn.Parameter(torch.empty_like(self.weight))
        if self.bias is not None:
            self.bias_score = nn.Parameter(torch.empty_like(self.bias))
        else:
            self.register_parameter("bias_score", None)

        self.weight_mask = None
        self.bias_mask = None
        self.active_weight_mask = None
        self.active_bias_mask = None
        self.occupied_weight_mask = None
        self.occupied_bias_mask = None

        self.reset_scores()

    def reset_scores(self) -> None:
        nn.init.kaiming_uniform_(self.score, a=np.sqrt(5))
        if self.bias_score is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(max(1, fan_in))
            nn.init.uniform_(self.bias_score, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wm = self._current_weight_mask()
        w = self.weight * wm
        if self.bias is None:
            b = None
        else:
            bm = self._current_bias_mask()
            b = self.bias * bm if bm is not None else self.bias
        return F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)


class TSNEmbedding(nn.Embedding, _TSNMaskMixin):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *args,
        keep_ratio: float = 0.5,
        allow_weight_reuse: bool = False,
        **kwargs,
    ):
        super().__init__(num_embeddings, embedding_dim, *args, **kwargs)
        self.keep_ratio = float(keep_ratio)
        self.allow_weight_reuse = bool(allow_weight_reuse)

        self.score = nn.Parameter(torch.empty_like(self.weight))
        self.register_parameter("bias_score", None)

        self.weight_mask = None
        self.bias_mask = None
        self.active_weight_mask = None
        self.active_bias_mask = None
        self.occupied_weight_mask = None
        self.occupied_bias_mask = None

        self.reset_scores()

    def reset_scores(self) -> None:
        nn.init.normal_(self.score, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wm = self._current_weight_mask()
        w = self.weight * wm
        return F.embedding(
            x,
            w,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )


@dataclass
class TSNConversionConfig:
    keep_ratio: float = 0.5
    include_embeddings: bool = True
    allow_weight_reuse: bool = False
    skip_module_names: Tuple[str, ...] = ("dt.te",)


def _replace_child(parent: nn.Module, child_name: str, new_child: nn.Module) -> None:
    setattr(parent, child_name, new_child)


def _iter_tsn_modules(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, (TSNLinear, TSNConv2d, TSNEmbedding)):
            yield name, module


def _convert_module_inplace(module: nn.Module, cfg: TSNConversionConfig, prefix: str = "") -> None:
    for child_name, child in list(module.named_children()):
        fq_name = f"{prefix}.{child_name}" if prefix else child_name
        if fq_name in cfg.skip_module_names:
            continue

        new_child: Optional[nn.Module] = None

        if isinstance(child, nn.Linear) and not isinstance(child, TSNLinear):
            new_child = TSNLinear(
                child.in_features,
                child.out_features,
                bias=(child.bias is not None),
                keep_ratio=cfg.keep_ratio,
                allow_weight_reuse=cfg.allow_weight_reuse,
            )
            with torch.no_grad():
                new_child.weight.copy_(child.weight)
                if child.bias is not None and new_child.bias is not None:
                    new_child.bias.copy_(child.bias)

        elif isinstance(child, nn.Conv2d) and not isinstance(child, TSNConv2d):
            new_child = TSNConv2d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=(child.bias is not None),
                padding_mode=child.padding_mode,
                keep_ratio=cfg.keep_ratio,
                allow_weight_reuse=cfg.allow_weight_reuse,
            )
            with torch.no_grad():
                new_child.weight.copy_(child.weight)
                if child.bias is not None and new_child.bias is not None:
                    new_child.bias.copy_(child.bias)

        elif cfg.include_embeddings and isinstance(child, nn.Embedding) and not isinstance(child, TSNEmbedding):
            new_child = TSNEmbedding(
                child.num_embeddings,
                child.embedding_dim,
                padding_idx=child.padding_idx,
                max_norm=child.max_norm,
                norm_type=child.norm_type,
                scale_grad_by_freq=child.scale_grad_by_freq,
                sparse=child.sparse,
                keep_ratio=cfg.keep_ratio,
                allow_weight_reuse=cfg.allow_weight_reuse,
            )
            with torch.no_grad():
                new_child.weight.copy_(child.weight)

        if new_child is not None:
            _replace_child(module, child_name, new_child)
            child = new_child

        _convert_module_inplace(child, cfg, fq_name)


def _kmeans_quantize_selected_(tensor: torch.Tensor, select_mask: torch.Tensor, n_clusters: int) -> Optional[np.ndarray]:
    select_mask = select_mask.to(device=tensor.device, dtype=torch.bool)
    flat_t = tensor.detach().view(-1)
    flat_m = select_mask.view(-1)
    if int(flat_m.sum().item()) < 2:
        return None

    selected = flat_t[flat_m].detach().cpu().numpy().reshape(-1, 1)
    uniq = np.unique(selected)
    clusters = int(min(max(1, n_clusters), len(selected), len(uniq)))
    if clusters < 2:
        return None

    km = KMeans(n_clusters=clusters, n_init=4, random_state=0)
    labels = km.fit_predict(selected)
    centers = km.cluster_centers_.reshape(-1)

    quantized = flat_t.detach().cpu().numpy().copy()
    quantized[np.where(flat_m.detach().cpu().numpy())[0]] = centers[labels]

    with torch.no_grad():
        tensor.copy_(torch.from_numpy(quantized).view_as(tensor).to(device=tensor.device, dtype=tensor.dtype))

    return centers
