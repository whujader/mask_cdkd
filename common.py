# common.py
# -----------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "MLPBlock",
    "ASPP",
    "SEBlock",
    "MultiScaleAdapter",
    "LayerNorm2d",
]


def _is_perfect_square(n: int) -> bool:
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n


def _infer_hw(num_tokens: int) -> Tuple[int, int]:
    """Infer (H,W) from tokens count. Require perfect square."""
    h = int(math.isqrt(num_tokens))
    if h * h != num_tokens:
        raise ValueError(f"num_tokens={num_tokens} is not a perfect square.")
    return h, h


def _split_cls_tokens(x: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Support optional CLS token in (B,N,C) sequences."""
    if x.dim() != 3:
        raise ValueError(f"Expected (B,N,C), got {tuple(x.shape)}")

    _, n, _ = x.shape
    if _is_perfect_square(n):
        return None, x
    if _is_perfect_square(n - 1):
        return x[:, :1], x[:, 1:]
    raise ValueError(f"N={n} neither square nor (N-1) square; cannot map to 2D grid.")


class MLPBlock(nn.Module):
    """Simple MLP block used in ViT FFN (works for (...,C) input)."""

    def __init__(self, embedding_dim: int, mlp_dim: int, act: Type[nn.Module] = nn.GELU) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class ASPP(nn.Module):
    """ASPP with depthwise atrous conv branches + GAP branch (paper-consistent)."""

    def __init__(self, in_channels: int, out_channels: int, dilations: Sequence[int]) -> None:
        super().__init__()
        branches = []

        for d in dilations:
            branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        padding=d,
                        dilation=d,
                        groups=in_channels,
                        bias=False,
                    ),
                    nn.GELU(),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.GELU(),
                )
            )

        # GAP branch
        branches.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GELU(),
            )
        )

        self.branches = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.Conv2d(len(branches) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        feats = []
        for branch in self.branches:
            y = branch(x)
            if y.shape[-2:] != size:
                y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
            feats.append(y)
        return self.project(torch.cat(feats, dim=1))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (channel attention)."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, channels, bias=False)
        self.act = nn.GELU()
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.act(self.fc1(y))
        y = self.gate(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class MultiScaleAdapter(nn.Module):
    """Paper-style multi-scale adapter: MLP down -> ASPP -> SE -> DWConv -> MLP up.

    IMPORTANT:
        - If skip_connect=False: returns delta features (recommended for MMoA fusion per paper).
        - If skip_connect=True: returns x + delta (classic adapter).
    """

    def __init__(
        self,
        d_features: int,
        mlp_ratio: float = 0.25,
        act_layer: Type[nn.Module] = nn.GELU,
        skip_connect: bool = False,
        se_reduction: int = 16,
        aspp_dilations: Sequence[int] = (1, 3, 5),
    ) -> None:
        super().__init__()
        d_hidden = int(d_features * mlp_ratio)

        self.skip_connect = bool(skip_connect)
        self.fc1 = nn.Linear(d_features, d_hidden)
        self.act = act_layer()

        self.aspp = ASPP(in_channels=d_hidden, out_channels=d_hidden, dilations=aspp_dilations)
        self.se = SEBlock(d_hidden, reduction=se_reduction)

        self.dwconv = nn.Sequential(
            nn.Conv2d(d_hidden, d_hidden, kernel_size=3, padding=1, groups=d_hidden, bias=False),
            nn.GELU(),
        )

        self.fc2 = nn.Linear(d_hidden, d_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,N,C) with optional CLS
        y = self.act(self.fc1(x))  # (B,N,Chid)

        cls, tokens = _split_cls_tokens(y)
        h, w = _infer_hw(tokens.size(1))

        feat = tokens.transpose(1, 2).contiguous().view(tokens.size(0), tokens.size(2), h, w)
        feat = self.aspp(feat)
        feat = self.se(feat)
        feat = self.dwconv(feat)

        tokens = feat.flatten(2).transpose(1, 2).contiguous()
        y = tokens if cls is None else torch.cat([cls, tokens], dim=1)

        y = self.fc2(y)  # (B,N,C)
        return (x + y) if self.skip_connect else y


class LayerNorm2d(nn.Module):
    """2D LayerNorm for BCHW feature maps (kept for completeness)."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(dim=1, keepdim=True)
        s = (x - u).pow(2).mean(dim=1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]
