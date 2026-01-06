# sparse.py
# -----------------------------------------------------------------------------
# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
from __future__ import annotations

from typing import Optional, Tuple

import luojianet_ms as ms
import luojianet_ms.nn as nn
import luojianet_ms.ops as ops
from luojianet_ms import Tensor

__all__ = [
    "set_active_mask",
    "clear_active_mask",
    "SparseConv2d",
    "SparseMaxPool2d",
    "SparseAvgPool2d",
    "SparseEncoder",
]

_ACTIVE_MASK: Optional[Tensor] = None
"""
Global active mask for sparse ops.

Expected shape:
- (B, 1, H, W) or (B, H, W)
Values: 0/1 or bool.
If None: sparsity disabled (all ones).
"""


def set_active_mask(mask: Optional[Tensor]) -> None:
    """Set global active mask used by sparse layers."""
    global _ACTIVE_MASK
    _ACTIVE_MASK = mask


def clear_active_mask() -> None:
    """Disable sparsity."""
    set_active_mask(None)


def _resize_nearest(x: Tensor, size_hw: Tuple[int, int]) -> Tensor:
    """Nearest-neighbor resize for NCHW tensors to (H,W)."""
    size = Tensor([int(size_hw[0]), int(size_hw[1])], ms.int32)
    resize = ops.ResizeNearestNeighbor(size)
    return resize(x)


def _expand_active_mask(
    *,
    batch_size: int,
    height: int,
    width: int,
    dtype: ms.dtype,
) -> Tensor:
    """Return a (B, 1, H, W) mask on requested dtype."""
    global _ACTIVE_MASK

    if _ACTIVE_MASK is None:
        return ops.ones((int(batch_size), 1, int(height), int(width)), dtype)

    m = _ACTIVE_MASK
    if len(m.shape) == 3:
        m = ops.ExpandDims()(m, 1)
    if len(m.shape) != 4 or int(m.shape[1]) != 1:
        raise ValueError(
            f"Active mask must be (B,1,H,W) or (B,H,W), got {tuple(_ACTIVE_MASK.shape)}."
        )
    if int(m.shape[0]) != int(batch_size):
        raise ValueError(f"Mask batch mismatch: mask B={int(m.shape[0])} vs input B={int(batch_size)}")

    # MindSpore Tensor does not need explicit .to(device=...) here;
    # we only enforce dtype to match behavior.
    if m.dtype != dtype:
        m = ops.Cast()(m, dtype)

    if (int(m.shape[-2]), int(m.shape[-1])) == (int(height), int(width)):
        return m

    # nearest keeps binary mask stable
    return _resize_nearest(m, size_hw=(int(height), int(width)))


class SparseConv2d(nn.Module):
    """Dense Conv2d followed by spatial masking of output."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        pad_mode: str = "pad",
        padding: int = 0,
        dilation=1,
        group: int = 1,
        has_bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            kernel_size=kernel_size,
            stride=stride,
            pad_mode=pad_mode,
            padding=padding,
            dilation=dilation,
            group=int(group),
            has_bias=bool(has_bias),
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv(x)
        mask = _expand_active_mask(
            batch_size=int(y.shape[0]),
            height=int(y.shape[2]),
            width=int(y.shape[3]),
            dtype=y.dtype,
        )
        return y * mask


class SparseMaxPool2d(nn.Module):
    """Dense MaxPool2d followed by spatial masking of output."""

    def __init__(
        self,
        kernel_size,
        stride=None,
        pad_mode: str = "VALID",
        padding: int = 0,
        dilation: int = 1,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        # MindSpore MaxPool2d: pad_mode in {"VALID","SAME"}
        # To strictly follow behavior, use pad_mode="VALID" by default.
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, pad_mode=pad_mode)

    def forward(self, x: Tensor) -> Tensor:
        y = self.pool(x)
        mask = _expand_active_mask(
            batch_size=int(y.shape[0]),
            height=int(y.shape[2]),
            width=int(y.shape[3]),
            dtype=y.dtype,
        )
        return y * mask


class SparseAvgPool2d(nn.Module):
    """Dense AvgPool2d followed by spatial masking of output."""

    def __init__(
        self,
        kernel_size,
        stride=None,
        pad_mode: str = "VALID",
        padding: int = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> None:
        super().__init__()
        # MindSpore AvgPool2d: pad_mode in {"VALID","SAME"}
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, pad_mode=pad_mode)

    def forward(self, x: Tensor) -> Tensor:
        y = self.pool(x)
        mask = _expand_active_mask(
            batch_size=int(y.shape[0]),
            height=int(y.shape[2]),
            width=int(y.shape[3]),
            dtype=y.dtype,
        )
        return y * mask


class SparseEncoder(nn.Module):
    """Compatibility wrapper."""

    def __init__(self, module: nn.Module, sbn: bool = False) -> None:
        super().__init__()
        self.module = module
        self.sbn = bool(sbn)

    def forward(self, x: Tensor) -> Tensor:
        return self.module(x)
