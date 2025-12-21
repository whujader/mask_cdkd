# teacher_sam_vit_mmoa.py
# -----------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
from __future__ import annotations

import math
from typing import Optional, Tuple, Type, Sequence, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import MLPBlock, MultiScaleAdapter

__all__ = ["ImageEncoderViTMMoA", "build_sam_vit_l_teacher_mmoa"]


# ---------------------------
# Small utilities
# ---------------------------
def trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    # light wrapper (avoid extra deps)
    return nn.init.trunc_normal_(tensor, std=std)


# ---------------------------
# Window utils (SAM-style)
# ---------------------------
def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """[B,H,W,C] -> windows [B*nW, ws, ws, C], with padding."""
    b, h, w, c = x.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

    hp, wp = h + pad_h, w + pad_w
    x = x.view(b, hp // window_size, window_size, wp // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows, (hp, wp)


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: Tuple[int, int],
    hw: Tuple[int, int],
) -> torch.Tensor:
    """windows -> [B,H,W,C], remove padding."""
    hp, wp = pad_hw
    h, w = hw
    b = windows.shape[0] // (hp * wp // window_size // window_size)

    x = windows.view(b, hp // window_size, wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, hp, wp, -1)
    return x[:, :h, :w, :].contiguous()


# ---------------------------
# Relative position (SAM-style)
# ---------------------------
def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    q_h, q_w = q_size
    k_h, k_w = k_size
    rh = get_rel_pos(q_h, k_h, rel_pos_h)
    rw = get_rel_pos(q_w, k_w, rel_pos_w)

    b, _, dim = q.shape
    r_q = q.reshape(b, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, rw)

    attn = attn.view(b, q_h, q_w, k_h, k_w)
    attn = attn + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    return attn.view(b, q_h * q_w, k_h * k_w)


# ---------------------------
# Patch embedding (BHWC)
# ---------------------------
class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B,C,H,W)
        return x.permute(0, 2, 3, 1).contiguous()  # (B,H,W,C)


# ---------------------------
# SAM-style attention (BHWC)
# ---------------------------
class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        use_rel_pos: bool = True,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} not divisible by num_heads={num_heads}")

        self.num_heads = int(num_heads)
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = bool(use_rel_pos)
        if self.use_rel_pos:
            if input_size is None:
                raise ValueError("input_size required when use_rel_pos=True")
            h, w = input_size
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * h - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * w - 1, head_dim))
            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        qkv = self.qkv(x).reshape(b, h * w, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3,B,Hd,N,dh)
        q, k, v = qkv.unbind(0)

        q = q.reshape(b * self.num_heads, h * w, -1)
        k = k.reshape(b * self.num_heads, h * w, -1)
        v = v.reshape(b * self.num_heads, h * w, -1)

        attn = (q * self.scale) @ k.transpose(-2, -1)  # (B*heads, N, N)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                attn=attn,
                q=q,
                rel_pos_h=self.rel_pos_h,
                rel_pos_w=self.rel_pos_w,
                q_size=(h, w),
                k_size=(h, w),
            )

        attn = attn.softmax(dim=-1)
        out = (attn @ v).view(b, self.num_heads, h, w, -1)
        out = out.permute(0, 2, 3, 1, 4).reshape(b, h, w, c)
        return self.proj(out)


# ---------------------------
# MMoA Gate (expert-attention router)
# ---------------------------
class MoAGate(nn.Module):
    """Attention-style gating over expert dimension (cheap: E x E, E=3)."""

    def __init__(self, dim: int, num_experts: int = 3) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.query = nn.Linear(dim, self.num_experts, bias=False)
        self.key = nn.Linear(dim, self.num_experts, bias=False)
        self.value = nn.Linear(dim, self.num_experts, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B,N,C) -> weights: (B,N,E)
        q = self.query(tokens)  # (B,N,E)
        k = self.key(tokens)    # (B,N,E)
        v = self.value(tokens)  # (B,N,E)

        # expert-wise attention: (B,N,E,E)
        attn = (q.unsqueeze(-1) @ k.unsqueeze(-2))
        attn = attn.softmax(dim=-1)

        # (B,N,E)
        logits = (attn @ v.unsqueeze(-1)).squeeze(-1)
        weights = logits.softmax(dim=-1)
        return weights


# ---------------------------
# Transformer block with MMoA (paper-consistent)
# ---------------------------
class MMoABlock(nn.Module):
    """SAM block + MMoA in FFN stage.

    Paper:
        X_n -> {FFN, Adapter_fine, Adapter_coarse}
        gate(X_n) -> weights over 3 streams
        X_out = X + sum_j w_j * F_j
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        norm_layer: Type[nn.Module],
        act_layer: Type[nn.Module],
        use_rel_pos: bool,
        rel_pos_zero_init: bool,
        window_size: int,
        input_size: Tuple[int, int],
        # adapter configs
        adapter_mlp_ratio: float = 0.25,
        fine_dilations: Sequence[int] = (1, 3, 5),
        coarse_dilations: Sequence[int] = (7, 9, 11),
    ) -> None:
        super().__init__()
        self.window_size = int(window_size)

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if self.window_size == 0 else (self.window_size, self.window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        # Two adapters (paper: fine + coarse), output delta (skip_connect=False)
        self.adapter_fine = MultiScaleAdapter(
            d_features=dim,
            mlp_ratio=adapter_mlp_ratio,
            aspp_dilations=fine_dilations,
            se_reduction=16,
            act_layer=nn.GELU,
            skip_connect=False,
        )
        self.adapter_coarse = MultiScaleAdapter(
            d_features=dim,
            mlp_ratio=adapter_mlp_ratio,
            aspp_dilations=coarse_dilations,
            se_reduction=16,
            act_layer=nn.GELU,
            skip_connect=False,
        )

        # 3-expert gate (FFN + 2 adapters)
        self.gate = MoAGate(dim=dim, num_experts=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,H,W,C)
        shortcut = x
        x = self.norm1(x)

        if self.window_size > 0:
            h, w = x.shape[1], x.shape[2]
            x_win, pad_hw = window_partition(x, self.window_size)
            x_win = self.attn(x_win)
            x = window_unpartition(x_win, self.window_size, pad_hw, (h, w))
        else:
            x = self.attn(x)

        x = shortcut + x

        # MMoA FFN stage
        xn = self.norm2(x)  # (B,H,W,C)
        b, h, w, c = xn.shape
        tokens = xn.view(b, h * w, c)  # (B,N,C)

        mlp_delta = self.mlp(xn).view(b, h * w, c)  # (B,N,C)
        fine_delta = self.adapter_fine(tokens)      # (B,N,C)
        coarse_delta = self.adapter_coarse(tokens)  # (B,N,C)

        weights = self.gate(tokens)                 # (B,N,3)
        outputs = torch.stack([mlp_delta, fine_delta, coarse_delta], dim=2)  # (B,N,3,C)
        mixed = torch.sum(outputs * weights.unsqueeze(-1), dim=2)            # (B,N,C)

        x = x + mixed.view(b, h, w, c)
        return x


# ---------------------------
# Token decoder for MAE (depth=4 per paper)
# ---------------------------
class DecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, norm_layer: Type[nn.Module]) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        y = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = x + self.proj(y)

        x = x + self.mlp(self.norm2(x))
        return x


class MAEDecoder(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, mlp_ratio: float, out_dim: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([DecoderBlock(dim, num_heads, mlp_ratio, nn.LayerNorm) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_dim)

    def forward(self, x_full: torch.Tensor, return_token_num: int) -> torch.Tensor:
        for blk in self.blocks:
            x_full = blk(x_full)
        x_full = self.norm(x_full)
        x_mask = x_full[:, -return_token_num:]   # masked tokens
        return self.head(x_mask)                 # (B, N_mask, patch_dim)


# ---------------------------
# Teacher: SAM-ViT-L + MMoA + MAE branch
# ---------------------------
class ImageEncoderViTMMoA(nn.Module):
    """Teacher encoder used in Mask-CDKD.

    - Backbone: SAM-style ViT (BHWC) with rel-pos & window/global attention.
    - Trainable: MMoA modules + MAE decoder branch (backbone params can be frozen externally).
    - Forward returns:
        latent: tuple of per-block token features (B, N_vis, C)
        pred: masked patch predictions (B, N_mask, patch_dim)
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,            # SAM ViT-L
        depth: int = 24,                  # SAM ViT-L
        num_heads: int = 16,              # SAM ViT-L
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = True,
        rel_pos_zero_init: bool = True,
        window_size: int = 14,
        global_attn_indexes: Tuple[int, ...] = (5, 11, 17, 23),
        # MAE decoder
        decoder_embed_dim: int = 1024,
        decoder_depth: int = 4,           # paper: 4-layer decoder
        decoder_num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        grid = self.img_size // self.patch_size
        self.num_patches = grid * grid

        self.patch_embed = PatchEmbed(patch_size=self.patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, grid, grid, embed_dim))
            trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList()
        for i in range(depth):
            is_global = i in set(global_attn_indexes)
            blk = MMoABlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=0 if is_global else window_size,
                input_size=(grid, grid),
                adapter_mlp_ratio=0.25,
                fine_dilations=(1, 3, 5),
                coarse_dilations=(7, 9, 11),
            )
            self.blocks.append(blk)

        # MAE decoder branch
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pos_embed_decoder = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim))

        patch_dim = self.patch_size * self.patch_size * in_chans
        self.decoder = MAEDecoder(
            dim=decoder_embed_dim,
            depth=int(decoder_depth),
            num_heads=int(decoder_num_heads),
            mlp_ratio=float(mlp_ratio),
            out_dim=int(patch_dim),
        )

        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.pos_embed_decoder, std=0.02)

    def forward_encoder(self, x: torch.Tensor, bool_masked_pos: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        b = x.size(0)
        mask = bool_masked_pos.view(b, -1)  # (B, L)

        x = self.patch_embed(x)  # (B, grid, grid, C)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        x_tokens = x.view(b, -1, x.shape[-1])               # (B, L, C)
        x_vis = x_tokens[~mask].reshape(b, -1, x.shape[-1]) # (B, N_vis, C)

        # To run BHWC blocks, we require N_vis to be a perfect square.
        n_vis = x_vis.size(1)
        side = int(math.isqrt(n_vis))
        if side * side != n_vis:
            raise ValueError(
                f"N_vis={n_vis} not square. "
                "For SAM-style BHWC blocks, please ensure mask ratio yields square visible tokens."
            )

        x_vis = x_vis.view(b, side, side, x.shape[-1])  # (B, side, side, C)

        outs: List[torch.Tensor] = []
        for blk in self.blocks:
            x_vis = blk(x_vis)
            outs.append(x_vis.view(b, -1, x_vis.shape[-1]))
        return tuple(outs)

    def forward_decoder(self, x_vis_tokens: torch.Tensor, bool_masked_pos: torch.Tensor) -> torch.Tensor:
        b = x_vis_tokens.size(0)
        mask = bool_masked_pos.view(b, -1)  # (B, L)

        x_vis = self.decoder_embed(x_vis_tokens)  # (B, N_vis, D_dec)

        pos = self.pos_embed_decoder.expand(b, -1, -1)  # (B, L, D_dec)
        pos_vis = pos[~mask].reshape(b, -1, x_vis.size(-1))
        pos_mask = pos[mask].reshape(b, -1, x_vis.size(-1))

        x_full = torch.cat(
            [x_vis + pos_vis, self.mask_token.expand(b, pos_mask.size(1), -1) + pos_mask],
            dim=1,
        )
        return self.decoder(x_full, return_token_num=pos_mask.size(1))  # (B, N_mask, patch_dim)

    def forward(self, x: torch.Tensor, bool_masked_pos: torch.Tensor) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        latent = self.forward_encoder(x, bool_masked_pos)
        pred = self.forward_decoder(latent[-1], bool_masked_pos)
        return latent, pred


def build_sam_vit_l_teacher_mmoa() -> ImageEncoderViTMMoA:
    """Paper-default teacher: SAM ViT-L (1024 dim, 24 blocks, 16 heads) + MMoA + 4-layer MAE decoder."""
    return ImageEncoderViTMMoA(
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        use_abs_pos=True,
        use_rel_pos=True,
        rel_pos_zero_init=True,
        window_size=14,
        global_attn_indexes=(5, 11, 17, 23),
        decoder_embed_dim=1024,
        decoder_depth=4,
        decoder_num_heads=8,
    )
