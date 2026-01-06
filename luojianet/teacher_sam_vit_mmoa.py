# teacher_sam_vit_mmoa_ms.py
# -----------------------------------------------------------------------------
# MindSpore version of teacher_sam_vit_mmoa.py (with SparseConv/Pool conversion
# for MultiScaleAdapter + active mask setting around adapter calls)
# -----------------------------------------------------------------------------
from __future__ import annotations

import math
from typing import Optional, Tuple, Type, Sequence, List

import luojianet_ms as ms
import luojianet_ms.nn as nn
import luojianet_ms.ops as ops
from luojianet_ms import Tensor, Parameter
from luojianet_ms.common.initializer import initializer, TruncatedNormal

from common import MLPBlock, MultiScaleAdapter
from sparse import (
    set_active_mask,
    clear_active_mask,
    SparseConv2d,
    SparseMaxPool2d,
    SparseAvgPool2d,
)

__all__ = ["ImageEncoderViTMMoA", "build_sam_vit_l_teacher_mmoa"]


# ---------------------------
# Small utilities
# ---------------------------
def trunc_normal_(tensor: Parameter, std: float = 0.02) -> Parameter:
    tensor.set_data(initializer(TruncatedNormal(sigma=float(std)), tensor.shape, tensor.dtype))
    return tensor


def _build_norm(norm_layer: Type[nn.Module], dim: int) -> nn.Module:
    # MindSpore LayerNorm expects normalized_shape as tuple/list.
    if norm_layer is nn.LayerNorm:
        return nn.LayerNorm((int(dim),), epsilon=1e-6)
    return norm_layer(int(dim))


def _convert_module_conv_pool_to_sparse(m: nn.Module) -> None:
    """
    Recursively replace Conv/Pool layers with Sparse* versions inside `m`.
    Only touches:
      - nn.Conv2d -> SparseConv2d (weights/bias copied)
      - nn.MaxPool2d -> SparseMaxPool2d
      - nn.AvgPool2d -> SparseAvgPool2d
    """
    # MindSpore: immediate children are in name_cells()
    for name, child in list(m.name_cells().items()):
        # 1) Conv2d -> SparseConv2d
        if isinstance(child, nn.Conv2d) and (not isinstance(child, SparseConv2d)):
            new = SparseConv2d(
                in_channels=int(getattr(child, "in_channels", 0)),
                out_channels=int(getattr(child, "out_channels", 0)),
                kernel_size=getattr(child, "kernel_size", 1),
                stride=getattr(child, "stride", 1),
                pad_mode=getattr(child, "pad_mode", "pad"),
                padding=int(getattr(child, "padding", 0)),
                dilation=getattr(child, "dilation", 1),
                group=int(getattr(child, "group", 1)),
                has_bias=bool(getattr(child, "has_bias", False)),
            )

            # try align dtype (best-effort)
            try:
                new.to_float(child.weight.dtype)
            except Exception:
                pass

            # copy weights / bias into wrapped conv
            new.conv.weight.set_data(child.weight.data)
            if bool(getattr(child, "has_bias", False)) and getattr(child, "bias", None) is not None:
                new.conv.bias.set_data(child.bias.data)

            setattr(m, name, new)
            continue

        # 2) MaxPool2d -> SparseMaxPool2d
        if isinstance(child, nn.MaxPool2d) and (not isinstance(child, SparseMaxPool2d)):
            new = SparseMaxPool2d(
                kernel_size=getattr(child, "kernel_size", 1),
                stride=getattr(child, "stride", None),
                pad_mode=getattr(child, "pad_mode", "VALID"),
                padding=int(getattr(child, "padding", 0)),
                dilation=int(getattr(child, "dilation", 1)),
                ceil_mode=bool(getattr(child, "ceil_mode", False)),
            )
            setattr(m, name, new)
            continue

        # 3) AvgPool2d -> SparseAvgPool2d
        if isinstance(child, nn.AvgPool2d) and (not isinstance(child, SparseAvgPool2d)):
            new = SparseAvgPool2d(
                kernel_size=getattr(child, "kernel_size", 1),
                stride=getattr(child, "stride", None),
                pad_mode=getattr(child, "pad_mode", "VALID"),
                padding=int(getattr(child, "padding", 0)),
                ceil_mode=bool(getattr(child, "ceil_mode", False)),
                count_include_pad=bool(getattr(child, "count_include_pad", True)),
            )
            setattr(m, name, new)
            continue

        # 4) recurse
        _convert_module_conv_pool_to_sparse(child)


# ---------------------------
# Window utils (SAM-style)
# ---------------------------
def window_partition(x: Tensor, window_size: int) -> Tuple[Tensor, Tuple[int, int]]:
    """[B,H,W,C] -> windows [B*nW, ws, ws, C], with padding."""
    b, h, w, c = x.shape
    ws = int(window_size)

    pad_h = (ws - int(h) % ws) % ws
    pad_w = (ws - int(w) % ws) % ws
    if pad_h > 0 or pad_w > 0:
        x = ops.Pad(((0, 0), (0, pad_h), (0, pad_w), (0, 0)))(x)

    hp, wp = int(h) + pad_h, int(w) + pad_w
    x = ops.Reshape()(x, (int(b), hp // ws, ws, wp // ws, ws, int(c)))
    windows = ops.Transpose()(x, (0, 1, 3, 2, 4, 5))
    windows = ops.Reshape()(windows, (-1, ws, ws, int(c)))
    return windows, (hp, wp)


def window_unpartition(windows: Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]) -> Tensor:
    """windows -> [B,H,W,C], remove padding."""
    hp, wp = int(pad_hw[0]), int(pad_hw[1])
    h, w = int(hw[0]), int(hw[1])
    ws = int(window_size)

    num_win_per_img = (hp * wp) // (ws * ws)
    b = int(windows.shape[0]) // int(num_win_per_img)

    x = ops.Reshape()(windows, (b, hp // ws, wp // ws, ws, ws, -1))
    x = ops.Transpose()(x, (0, 1, 3, 2, 4, 5))
    x = ops.Reshape()(x, (b, hp, wp, -1))
    return x[:, :h, :w, :]


# ---------------------------
# Relative position (SAM-style)
# ---------------------------
def _resize_rel_pos_1d(rel_pos: Tensor, target_len: int) -> Tensor:
    """
    PyTorch: F.interpolate(..., mode="linear") on (1, C, L).
    MindSpore: emulate via ResizeBilinearV2 on (1, C, L, 1).
    Output: (target_len, C)
    """
    l, c = int(rel_pos.shape[0]), int(rel_pos.shape[1])
    if l == int(target_len):
        return rel_pos

    x = ops.Reshape()(rel_pos, (1, l, c))            # (1, L, C)
    x = ops.Transpose()(x, (0, 2, 1))                # (1, C, L)
    x = ops.Reshape()(x, (1, c, l, 1))               # (1, C, L, 1)

    size = Tensor([int(target_len), 1], ms.int32)
    resize = ops.ResizeBilinearV2(align_corners=False, half_pixel_centers=False)
    x = resize(x, size)                              # (1, C, target_len, 1)

    x = ops.Reshape()(x, (1, c, int(target_len)))     # (1, C, target_len)
    x = ops.Transpose()(x, (0, 2, 1))                 # (1, target_len, C)
    x = ops.Reshape()(x, (int(target_len), c))        # (target_len, C)
    return x


def _arange_0_n(n: int, dtype: ms.dtype = ms.float32) -> Tensor:
    # stable across versions: use Range op
    r = ops.Range()
    return r(Tensor(0, dtype), Tensor(int(n), dtype), Tensor(1, dtype))


def get_rel_pos(q_size: int, k_size: int, rel_pos: Tensor) -> Tensor:
    max_rel_dist = int(2 * max(int(q_size), int(k_size)) - 1)
    rel_pos_resized = _resize_rel_pos_1d(rel_pos, max_rel_dist)

    q_scale = max(float(k_size) / float(q_size), 1.0)
    k_scale = max(float(q_size) / float(k_size), 1.0)

    q_coords = _arange_0_n(int(q_size), ms.float32)
    k_coords = _arange_0_n(int(k_size), ms.float32)

    q_coords = ops.Reshape()(q_coords, (int(q_size), 1)) * float(q_scale)
    k_coords = ops.Reshape()(k_coords, (1, int(k_size))) * float(k_scale)

    relative_coords = (q_coords - k_coords) + float(k_size - 1) * float(k_scale)  # (q, k)
    relative_coords = ops.Cast()(relative_coords, ms.int32)

    flat = ops.Reshape()(relative_coords, (-1,))
    gathered = ops.Gather()(rel_pos_resized, flat, 0)  # (q*k, C)
    return ops.Reshape()(gathered, (int(q_size), int(k_size), -1))  # (q, k, C)


def add_decomposed_rel_pos(
    attn: Tensor,
    q: Tensor,
    rel_pos_h: Tensor,
    rel_pos_w: Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> Tensor:
    q_h, q_w = int(q_size[0]), int(q_size[1])
    k_h, k_w = int(k_size[0]), int(k_size[1])

    rh = get_rel_pos(q_h, k_h, rel_pos_h)  # (q_h, k_h, dim)
    rw = get_rel_pos(q_w, k_w, rel_pos_w)  # (q_w, k_w, dim)

    b, _, dim = q.shape
    r_q = ops.Reshape()(q, (int(b), q_h, q_w, int(dim)))  # (b,qh,qw,dim)

    # rel_h: (b,qh,qw,kh)
    r_q_h = ops.ExpandDims()(r_q, 3)                         # (b,qh,qw,1,dim)
    rh_e = ops.Reshape()(rh, (1, q_h, 1, k_h, int(dim)))     # (1,qh,1,kh,dim)
    rel_h = ops.ReduceSum(keep_dims=False)(r_q_h * rh_e, -1)  # (b,qh,qw,kh)

    # rel_w: (b,qh,qw,kw)
    r_q_w = ops.ExpandDims()(r_q, 3)                         # (b,qh,qw,1,dim)
    rw_e = ops.Reshape()(rw, (1, 1, q_w, k_w, int(dim)))     # (1,1,qw,kw,dim)
    rel_w = ops.ReduceSum(keep_dims=False)(r_q_w * rw_e, -1)  # (b,qh,qw,kw)

    attn5 = ops.Reshape()(attn, (int(b), q_h, q_w, k_h, k_w))
    attn5 = attn5 + ops.ExpandDims()(rel_h, -1) + ops.ExpandDims()(rel_w, -2)
    return ops.Reshape()(attn5, (int(b), q_h * q_w, k_h * k_w))


# ---------------------------
# Patch embedding (BHWC)
# ---------------------------
class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 1024) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=int(in_chans),
            out_channels=int(embed_dim),
            kernel_size=int(patch_size),
            stride=int(patch_size),
            pad_mode="valid",
            has_bias=True,
        )
        self.transpose = ops.Transpose()

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)                 # (B,C,H,W)
        return self.transpose(x, (0, 2, 3, 1))  # (B,H,W,C)


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
        dim = int(dim)
        num_heads = int(num_heads)
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} not divisible by num_heads={num_heads}")

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = float(head_dim) ** -0.5

        self.qkv = nn.Dense(dim, dim * 3, has_bias=bool(qkv_bias))
        self.proj = nn.Dense(dim, dim, has_bias=True)

        self.use_rel_pos = bool(use_rel_pos)
        if self.use_rel_pos:
            if input_size is None:
                raise ValueError("input_size required when use_rel_pos=True")
            h, w = int(input_size[0]), int(input_size[1])
            self.rel_pos_h = Parameter(ops.zeros((2 * h - 1, head_dim), ms.float32), name="rel_pos_h")
            self.rel_pos_w = Parameter(ops.zeros((2 * w - 1, head_dim), ms.float32), name="rel_pos_w")
            if not bool(rel_pos_zero_init):
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)

        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.unstack0 = ops.Unstack(axis=0)
        self.softmax = ops.Softmax(axis=-1)
        self.bmm_t = ops.BatchMatMul(transpose_b=True)
        self.bmm = ops.BatchMatMul()

    def forward(self, x: Tensor) -> Tensor:
        b, h, w, c = x.shape
        n = int(h) * int(w)
        heads = self.num_heads
        dh = int(c) // heads

        qkv = self.qkv(x)  # (B,H,W,3C)
        qkv = self.reshape(qkv, (int(b), n, 3, heads, dh))
        qkv = self.transpose(qkv, (2, 0, 3, 1, 4))  # (3,B,H,N,dh)
        q, k, v = self.unstack0(qkv)

        q2 = self.reshape(q, (int(b) * heads, n, dh))
        k2 = self.reshape(k, (int(b) * heads, n, dh))
        v2 = self.reshape(v, (int(b) * heads, n, dh))

        attn = self.bmm_t(q2 * self.scale, k2)  # (B*heads, N, N)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                attn=attn,
                q=q2,
                rel_pos_h=self.rel_pos_h,
                rel_pos_w=self.rel_pos_w,
                q_size=(int(h), int(w)),
                k_size=(int(h), int(w)),
            )

        attn = self.softmax(attn)
        out = self.bmm(attn, v2)  # (B*heads, N, dh)

        out = self.reshape(out, (int(b), heads, int(h), int(w), dh))
        out = self.transpose(out, (0, 2, 3, 1, 4))
        out = self.reshape(out, (int(b), int(h), int(w), int(c)))
        return self.proj(out)


# ---------------------------
# MMoA Gate (expert-attention router)
# ---------------------------
class MoAGate(nn.Module):
    """Attention-style gating over expert dimension (cheap: E x E, E=3)."""

    def __init__(self, dim: int, num_experts: int = 3) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.query = nn.Dense(int(dim), self.num_experts, has_bias=False)
        self.key = nn.Dense(int(dim), self.num_experts, has_bias=False)
        self.value = nn.Dense(int(dim), self.num_experts, has_bias=False)

        self.matmul = ops.MatMul()
        self.softmax = ops.Softmax(axis=-1)
        self.expand = ops.ExpandDims()
        self.squeeze = ops.Squeeze(axis=-1)

    def forward(self, tokens: Tensor) -> Tensor:
        # tokens: (B,N,C) -> weights: (B,N,E)
        q = self.query(tokens)  # (B,N,E)
        k = self.key(tokens)    # (B,N,E)
        v = self.value(tokens)  # (B,N,E)

        attn = self.matmul(self.expand(q, -1), self.expand(k, -2))  # (B,N,E,E)
        attn = self.softmax(attn)

        logits = self.squeeze(self.matmul(attn, self.expand(v, -1)))  # (B,N,E)
        weights = self.softmax(logits)
        return weights


# ---------------------------
# Transformer block with MMoA (paper-consistent)
# ---------------------------
class MMoABlock(nn.Module):
    """SAM block + MMoA in FFN stage."""

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
        adapter_mlp_ratio: float = 0.25,
        fine_dilations: Sequence[int] = (1, 3, 5),
        coarse_dilations: Sequence[int] = (7, 9, 11),
    ) -> None:
        super().__init__()
        self.window_size = int(window_size)

        self.norm1 = _build_norm(norm_layer, int(dim))
        self.attn = Attention(
            dim=int(dim),
            num_heads=int(num_heads),
            qkv_bias=bool(qkv_bias),
            use_rel_pos=bool(use_rel_pos),
            rel_pos_zero_init=bool(rel_pos_zero_init),
            input_size=input_size if self.window_size == 0 else (self.window_size, self.window_size),
        )

        self.norm2 = _build_norm(norm_layer, int(dim))
        self.mlp = MLPBlock(embedding_dim=int(dim), mlp_dim=int(int(dim) * float(mlp_ratio)), act=act_layer)

        self.adapter_fine = MultiScaleAdapter(
            d_features=int(dim),
            mlp_ratio=float(adapter_mlp_ratio),
            aspp_dilations=tuple(int(d) for d in fine_dilations),
            se_reduction=16,
            act_layer=nn.GELU,
            skip_connect=False,
        )
        self.adapter_coarse = MultiScaleAdapter(
            d_features=int(dim),
            mlp_ratio=float(adapter_mlp_ratio),
            aspp_dilations=tuple(int(d) for d in coarse_dilations),
            se_reduction=16,
            act_layer=nn.GELU,
            skip_connect=False,
        )

        # IMPORTANT: convert adapter conv/pool to sparse versions
        _convert_module_conv_pool_to_sparse(self.adapter_fine)
        _convert_module_conv_pool_to_sparse(self.adapter_coarse)

        self.gate = MoAGate(dim=int(dim), num_experts=3)

        self.reshape = ops.Reshape()
        self.stack = ops.Stack(axis=2)
        self.expand = ops.ExpandDims()
        self.reduce_sum = ops.ReduceSum(keep_dims=False)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B,H,W,C)
        shortcut = x
        x = self.norm1(x)

        if self.window_size > 0:
            h, w = int(x.shape[1]), int(x.shape[2])
            x_win, pad_hw = window_partition(x, self.window_size)
            x_win = self.attn(x_win)
            x = window_unpartition(x_win, self.window_size, pad_hw, (h, w))
        else:
            x = self.attn(x)

        x = shortcut + x

        # MMoA FFN stage
        xn = self.norm2(x)  # (B,H,W,C)
        b, h, w, c = xn.shape
        tokens = self.reshape(xn, (int(b), int(h) * int(w), int(c)))  # (B,N,C)

        mlp_delta = self.reshape(self.mlp(xn), (int(b), int(h) * int(w), int(c)))  # (B,N,C)

        # Active mask must be (B,1,H,W) or (B,H,W), True=active, False=inactive.
        active_mask = ops.ones((int(b), 1, int(h), int(w)), ms.bool_)

        set_active_mask(active_mask)
        try:
            fine_delta = self.adapter_fine(tokens)      # (B,N,C)
            coarse_delta = self.adapter_coarse(tokens)  # (B,N,C)
        finally:
            clear_active_mask()

        weights = self.gate(tokens)  # (B,N,3)
        outputs = self.stack((mlp_delta, fine_delta, coarse_delta))  # (B,N,3,C)
        mixed = self.reduce_sum(outputs * self.expand(weights, -1), 2)  # (B,N,C)

        x = x + self.reshape(mixed, (int(b), int(h), int(w), int(c)))
        return x


# ---------------------------
# Token decoder for MAE (depth=4 per paper)
# ---------------------------
class DecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, norm_layer: Type[nn.Module]) -> None:
        super().__init__()
        dim = int(dim)
        self.num_heads = int(num_heads)
        self.scale = float(dim // self.num_heads) ** -0.5

        self.norm1 = _build_norm(norm_layer, dim)
        self.qkv = nn.Dense(dim, dim * 3, has_bias=True)
        self.proj = nn.Dense(dim, dim, has_bias=True)

        self.norm2 = _build_norm(norm_layer, dim)
        hidden = int(dim * float(mlp_ratio))
        self.mlp = nn.SequentialModule(
            [
                nn.Dense(dim, hidden, has_bias=True),
                nn.GELU(),
                nn.Dense(hidden, dim, has_bias=True),
            ]
        )

        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.unstack0 = ops.Unstack(axis=0)
        self.softmax = ops.Softmax(axis=-1)
        self.bmm_t = ops.BatchMatMul(transpose_b=True)
        self.bmm = ops.BatchMatMul()

    def forward(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        heads = self.num_heads
        dh = int(c) // heads

        qkv = self.qkv(self.norm1(x))
        qkv = self.reshape(qkv, (int(b), int(n), 3, heads, dh))
        qkv = self.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = self.unstack0(qkv)

        q2 = self.reshape(q, (int(b) * heads, int(n), dh))
        k2 = self.reshape(k, (int(b) * heads, int(n), dh))
        v2 = self.reshape(v, (int(b) * heads, int(n), dh))

        attn = self.bmm_t(q2, k2) * self.scale
        attn = self.softmax(attn)

        y = self.bmm(attn, v2)
        y = self.reshape(y, (int(b), heads, int(n), dh))
        y = self.transpose(y, (0, 2, 1, 3))
        y = self.reshape(y, (int(b), int(n), int(c)))

        x = x + self.proj(y)
        x = x + self.mlp(self.norm2(x))
        return x


class MAEDecoder(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, mlp_ratio: float, out_dim: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [DecoderBlock(int(dim), int(num_heads), float(mlp_ratio), nn.LayerNorm) for _ in range(int(depth))]
        )
        self.norm = nn.LayerNorm((int(dim),), epsilon=1e-6)
        self.head = nn.Dense(int(dim), int(out_dim), has_bias=True)

    def forward(self, x_full: Tensor, return_token_num: int) -> Tensor:
        for blk in self.blocks:
            x_full = blk(x_full)
        x_full = self.norm(x_full)
        x_mask = x_full[:, -int(return_token_num) :, :]
        return self.head(x_mask)


# ---------------------------
# Teacher: SAM-ViT-L + MMoA + MAE branch
# ---------------------------
class ImageEncoderViTMMoA(nn.Module):
    """Teacher encoder used in Mask-CDKD (MindSpore)."""

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = True,
        rel_pos_zero_init: bool = True,
        window_size: int = 14,
        global_attn_indexes: Tuple[int, ...] = (5, 11, 17, 23),
        decoder_embed_dim: int = 1024,
        decoder_depth: int = 4,
        decoder_num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        grid = self.img_size // self.patch_size
        self.num_patches = int(grid * grid)

        self.patch_embed = PatchEmbed(patch_size=self.patch_size, in_chans=int(in_chans), embed_dim=int(embed_dim))

        self.pos_embed: Optional[Parameter] = None
        if bool(use_abs_pos):
            self.pos_embed = Parameter(ops.zeros((1, grid, grid, int(embed_dim)), ms.float32), name="pos_embed")
            trunc_normal_(self.pos_embed, std=0.02)

        ga_set = set(int(i) for i in global_attn_indexes)

        self.blocks = nn.ModuleList()
        for i in range(int(depth)):
            is_global = int(i) in ga_set
            blk = MMoABlock(
                dim=int(embed_dim),
                num_heads=int(num_heads),
                mlp_ratio=float(mlp_ratio),
                qkv_bias=bool(qkv_bias),
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=bool(use_rel_pos),
                rel_pos_zero_init=bool(rel_pos_zero_init),
                window_size=0 if is_global else int(window_size),
                input_size=(int(grid), int(grid)),
                adapter_mlp_ratio=0.25,
                fine_dilations=(1, 3, 5),
                coarse_dilations=(7, 9, 11),
            )
            self.blocks.append(blk)

        self.decoder_embed = nn.Dense(int(embed_dim), int(decoder_embed_dim), has_bias=True)
        self.mask_token = Parameter(ops.zeros((1, 1, int(decoder_embed_dim)), ms.float32), name="mask_token")
        self.pos_embed_decoder = Parameter(
            ops.zeros((1, self.num_patches, int(decoder_embed_dim)), ms.float32), name="pos_embed_decoder"
        )

        patch_dim = self.patch_size * self.patch_size * int(in_chans)
        self.decoder = MAEDecoder(
            dim=int(decoder_embed_dim),
            depth=int(decoder_depth),
            num_heads=int(decoder_num_heads),
            mlp_ratio=float(mlp_ratio),
            out_dim=int(patch_dim),
        )

        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.pos_embed_decoder, std=0.02)

        self.reshape = ops.Reshape()
        self.logical_not = ops.LogicalNot()
        self.concat1 = ops.Concat(axis=1)

    def forward_encoder(self, x: Tensor, bool_masked_pos: Tensor) -> Tuple[Tensor, ...]:
        b = int(x.shape[0])
        mask = self.reshape(bool_masked_pos, (b, -1))  # (B, L)

        x = self.patch_embed(x)  # (B, grid, grid, C)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        x_tokens = self.reshape(x, (b, -1, int(x.shape[-1])))  # (B, L, C)
        inv_mask = self.logical_not(mask)

        x_vis = x_tokens[inv_mask]
        x_vis = self.reshape(x_vis, (b, -1, int(x_tokens.shape[-1])))  # (B, N_vis, C)

        n_vis = int(x_vis.shape[1])
        side = int(math.isqrt(n_vis))
        if side * side != n_vis:
            raise ValueError(
                f"N_vis={n_vis} not square. "
                "For SAM-style BHWC blocks, please ensure mask ratio yields square visible tokens."
            )

        x_vis = self.reshape(x_vis, (b, side, side, int(x_tokens.shape[-1])))  # (B, side, side, C)

        outs: List[Tensor] = []
        for blk in self.blocks:
            x_vis = blk(x_vis)
            outs.append(self.reshape(x_vis, (b, -1, int(x_vis.shape[-1]))))
        return tuple(outs)

    def forward_decoder(self, x_vis_tokens: Tensor, bool_masked_pos: Tensor) -> Tensor:
        b = int(x_vis_tokens.shape[0])
        mask = self.reshape(bool_masked_pos, (b, -1))  # (B, L)
        inv_mask = self.logical_not(mask)

        x_vis = self.decoder_embed(x_vis_tokens)  # (B, N_vis, D_dec)
        d_dec = int(x_vis.shape[-1])

        pos = ops.tile(self.pos_embed_decoder, (b, 1, 1))  # (B, L, D_dec)

        pos_vis = pos[inv_mask]
        pos_vis = self.reshape(pos_vis, (b, -1, d_dec))

        pos_mask = pos[mask]
        pos_mask = self.reshape(pos_mask, (b, -1, d_dec))

        n_mask = int(pos_mask.shape[1])
        mask_tok = ops.tile(self.mask_token, (b, n_mask, 1))

        x_full = self.concat1((x_vis + pos_vis, mask_tok + pos_mask))
        return self.decoder(x_full, return_token_num=n_mask)

    def forward(self, x: Tensor, bool_masked_pos: Tensor) -> Tuple[Tuple[Tensor, ...], Tensor]:
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
