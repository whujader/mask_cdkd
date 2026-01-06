# common_ms.py
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

import luojianet_ms as ms
import luojianet_ms.nn as nn
import luojianet_ms.ops as ops
from luojianet_ms import Parameter, Tensor

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


def _split_cls_tokens(x: Tensor) -> Tuple[Optional[Tensor], Tensor]:
    """Support optional CLS token in (B,N,C) sequences."""
    if len(x.shape) != 3:
        raise ValueError(f"Expected (B,N,C), got {tuple(x.shape)}")

    n = int(x.shape[1])
    if _is_perfect_square(n):
        return None, x
    if _is_perfect_square(n - 1):
        return x[:, :1, :], x[:, 1:, :]
    raise ValueError(f"N={n} neither square nor (N-1) square; cannot map to 2D grid.")


def _resize_bilinear(x: Tensor, size_hw: Tuple[int, int]) -> Tensor:
    """Bilinear resize for NCHW tensors to (H,W), align_corners=False (torch-consistent here)."""
    # ops.ResizeBilinearV2 expects `size` as a Tensor([new_h, new_w], int32)
    size = Tensor([int(size_hw[0]), int(size_hw[1])], ms.int32)
    resize = ops.ResizeBilinearV2(align_corners=False, half_pixel_centers=False)
    return resize(x, size)


class MLPBlock(nn.Module):
    """Simple MLP block used in ViT FFN (works for (...,C) input)."""

    def __init__(self, embedding_dim: int, mlp_dim: int, act: Type[nn.Module] = nn.GELU) -> None:
        super().__init__()
        self.lin1 = nn.Dense(embedding_dim, mlp_dim, has_bias=True)
        self.lin2 = nn.Dense(mlp_dim, embedding_dim, has_bias=True)
        self.act = act()

    def forward(self, x: Tensor) -> Tensor:
        return self.lin2(self.act(self.lin1(x)))


class ASPP(nn.Module):
    """ASPP with depthwise atrous conv branches + GAP branch (paper-consistent)."""

    def __init__(self, in_channels: int, out_channels: int, dilations: Sequence[int]) -> None:
        super().__init__()
        branches = []

        for d in dilations:
            branches.append(
                nn.SequentialModule(
                    [
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=3,
                            pad_mode="pad",
                            padding=int(d),
                            dilation=int(d),
                            group=in_channels,
                            has_bias=False,
                        ),
                        nn.GELU(),
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=1,
                            pad_mode="pad",
                            padding=0,
                            has_bias=False,
                        ),
                        nn.GELU(),
                    ]
                )
            )

        # GAP branch
        branches.append(
            nn.SequentialModule(
                [
                    nn.AdaptiveAvgPool2d(output_size=1),
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        pad_mode="pad",
                        padding=0,
                        has_bias=False,
                    ),
                    nn.GELU(),
                ]
            )
        )

        self.branches = nn.ModuleList(branches)
        self.project = nn.SequentialModule(
            [
                nn.Conv2d(
                    in_channels=len(branches) * out_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    pad_mode="pad",
                    padding=0,
                    has_bias=False,
                ),
                nn.GELU(),
            ]
        )

        self._concat_c = ops.Concat(axis=1)

    def forward(self, x: Tensor) -> Tensor:
        size = (int(x.shape[-2]), int(x.shape[-1]))
        feats = []
        for branch in self.branches:
            y = branch(x)
            if (int(y.shape[-2]), int(y.shape[-1])) != size:
                y = _resize_bilinear(y, size_hw=size)
            feats.append(y)
        return self.project(self._concat_c(tuple(feats)))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (channel attention)."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // int(reduction))
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Dense(channels, hidden, has_bias=False)
        self.fc2 = nn.Dense(hidden, channels, has_bias=False)
        self.act = nn.GELU()
        self.gate = nn.Sigmoid()

        self._reshape = ops.Reshape()

    def forward(self, x: Tensor) -> Tensor:
        b, c = int(x.shape[0]), int(x.shape[1])
        y = self.pool(x)                       # (B,C,1,1)
        y = self._reshape(y, (b, c))           # (B,C)
        y = self.act(self.fc1(y))              # (B,hidden)
        y = self.gate(self.fc2(y))             # (B,C)
        y = self._reshape(y, (b, c, 1, 1))     # (B,C,1,1)
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
        d_hidden = int(d_features * float(mlp_ratio))

        self.skip_connect = bool(skip_connect)
        self.fc1 = nn.Dense(d_features, d_hidden, has_bias=True)
        self.act = act_layer()

        self.aspp = ASPP(in_channels=d_hidden, out_channels=d_hidden, dilations=aspp_dilations)
        self.se = SEBlock(d_hidden, reduction=se_reduction)

        self.dwconv = nn.SequentialModule(
            [
                nn.Conv2d(
                    in_channels=d_hidden,
                    out_channels=d_hidden,
                    kernel_size=3,
                    pad_mode="pad",
                    padding=1,
                    group=d_hidden,
                    has_bias=False,
                ),
                nn.GELU(),
            ]
        )

        self.fc2 = nn.Dense(d_hidden, d_features, has_bias=True)

        self._transpose = ops.Transpose()
        self._reshape = ops.Reshape()
        self._concat_seq = ops.Concat(axis=1)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B,N,C) with optional CLS
        y = self.act(self.fc1(x))  # (B,N,Chid)

        cls, tokens = _split_cls_tokens(y)
        h, w = _infer_hw(int(tokens.shape[1]))

        b = int(tokens.shape[0])
        ch = int(tokens.shape[2])

        # (B,N,Ch) -> (B,Ch,N) -> (B,Ch,H,W)
        feat = self._transpose(tokens, (0, 2, 1))
        feat = self._reshape(feat, (b, ch, h, w))

        feat = self.aspp(feat)
        feat = self.se(feat)
        feat = self.dwconv(feat)

        # (B,Ch,H,W) -> (B,Ch,H*W) -> (B,H*W,Ch)
        tokens2 = self._reshape(feat, (b, ch, h * w))
        tokens2 = self._transpose(tokens2, (0, 2, 1))

        y2 = tokens2 if cls is None else self._concat_seq((cls, tokens2))  # (B,N,Chid)
        y2 = self.fc2(y2)  # (B,N,C)

        return (x + y2) if self.skip_connect else y2


class LayerNorm2d(nn.Module):
    """2D LayerNorm for BCHW feature maps (kept for completeness)."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = Parameter(ops.ones((num_channels,), ms.float32), name="weight")
        self.bias = Parameter(ops.zeros((num_channels,), ms.float32), name="bias")
        self.eps = float(eps)

        self._mean = ops.ReduceMean(keep_dims=True)
        self._sqrt = ops.Sqrt()
        self._reshape = ops.Reshape()

    def forward(self, x: Tensor) -> Tensor:
        # mean/var over channel dim (dim=1)
        u = self._mean(x, 1)
        s = self._mean((x - u) ** 2, 1)
        x_norm = (x - u) / self._sqrt(s + self.eps)

        w = self._reshape(self.weight, (1, -1, 1, 1))
        b = self._reshape(self.bias, (1, -1, 1, 1))
        return w * x_norm + b
