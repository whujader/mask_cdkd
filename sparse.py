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

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "set_active_mask",
    "clear_active_mask",
    "SparseConv2d",
    "SparseMaxPool2d",
    "SparseAvgPool2d",
    "SparseEncoder",
]

_ACTIVE_MASK: Optional[torch.Tensor] = None
"""
Global active mask for sparse ops.

Expected shape:
- (B, 1, H, W) or (B, H, W)
Values: 0/1 or bool.
If None: sparsity disabled (all ones).
"""


def set_active_mask(mask: Optional[torch.Tensor]) -> None:
    """Set global active mask used by sparse layers."""
    global _ACTIVE_MASK
    _ACTIVE_MASK = mask


def clear_active_mask() -> None:
    """Disable sparsity."""
    set_active_mask(None)


def _expand_active_mask(
    *,
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a (B, 1, H, W) mask on requested device/dtype."""
    if _ACTIVE_MASK is None:
        return torch.ones(batch_size, 1, height, width, device=device, dtype=dtype)

    m = _ACTIVE_MASK
    if m.dim() == 3:
        m = m.unsqueeze(1)
    if m.dim() != 4 or m.size(1) != 1:
        raise ValueError(f"Active mask must be (B,1,H,W) or (B,H,W), got {tuple(_ACTIVE_MASK.shape)}.")
    if m.size(0) != batch_size:
        raise ValueError(f"Mask batch mismatch: mask B={m.size(0)} vs input B={batch_size}")

    m = m.to(device=device, dtype=dtype)
    if m.shape[-2:] == (height, width):
        return m

    # nearest keeps binary mask stable
    return F.interpolate(m, size=(height, width), mode="nearest")


class SparseConv2d(nn.Conv2d):
    """Dense Conv2d followed by spatial masking of output."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        mask = _expand_active_mask(
            batch_size=y.size(0),
            height=y.size(2),
            width=y.size(3),
            device=y.device,
            dtype=y.dtype,
        )
        return y * mask


class SparseMaxPool2d(nn.MaxPool2d):
    """Dense MaxPool2d followed by spatial masking of output."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        mask = _expand_active_mask(
            batch_size=y.size(0),
            height=y.size(2),
            width=y.size(3),
            device=y.device,
            dtype=y.dtype,
        )
        return y * mask


class SparseAvgPool2d(nn.AvgPool2d):
    """Dense AvgPool2d followed by spatial masking of output."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        mask = _expand_active_mask(
            batch_size=y.size(0),
            height=y.size(2),
            width=y.size(3),
            device=y.device,
            dtype=y.dtype,
        )
        return y * mask


class SparseEncoder(nn.Module):
    """Compatibility wrapper.

    In your original code SparseEncoder(...) was used to wrap adapters.
    For paper-consistent MMoA we no longer *require* this wrapper, but we keep it so
    your repo's historical imports won't break.
    """

    def __init__(self, module: nn.Module, sbn: bool = False) -> None:
        super().__init__()
        self.module = module
        self.sbn = bool(sbn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)
