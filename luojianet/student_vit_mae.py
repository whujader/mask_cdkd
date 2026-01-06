# student_vit_mae.py

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union, Type, List

import luojianet_ms as ms
import luojianet_ms.nn as nn
import luojianet_ms.ops as ops
from luojianet_ms import Tensor, Parameter
from luojianet_ms.common.initializer import initializer, XavierUniform, TruncatedNormal, Constant


# ---------------------------
# Utils
# ---------------------------
def to_2tuple(x: Union[int, Sequence[int]]) -> Tuple[int, int]:
    if isinstance(x, (tuple, list)):
        if len(x) != 2:
            raise ValueError(f"Expected a 2-tuple/list, got len={len(x)}.")
        return int(x[0]), int(x[1])
    return int(x), int(x)


def trunc_normal_(
    tensor: Union[Tensor, Parameter],
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> Union[Tensor, Parameter]:
    """
    Truncated normal init (MindSpore).
    Notes:
      - Uses MindSpore's TruncatedNormal(sigma=std) (commonly truncated at ~2 sigma).
      - Then shifts by mean (if mean != 0).
      - a/b are kept for signature compatibility with the PyTorch code.
    """
    data = initializer(TruncatedNormal(sigma=float(std)), tensor.shape, tensor.dtype)
    if float(mean) != 0.0:
        data = data + float(mean)

    if isinstance(tensor, Parameter):
        tensor.set_data(data)
        return tensor
    return data


# ---------------------------
# Core Blocks
# ---------------------------
class Identity(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class DropPath(nn.Module):
    """Stochastic Depth per sample (residual branch)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)
        keep_prob = 1.0 - self.drop_prob
        # Dropout in MindSpore is scaled by 1/keep_prob; applying it to ones
        # gives either 0 or 1/keep_prob, matching stochastic-depth scaling.
        self._drop = nn.Dropout(keep_prob=keep_prob) if self.drop_prob > 0.0 else None
        self._ones = ops.Ones()

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or (not self.training):
            return x
        shape = (x.shape[0],) + (1,) * (len(x.shape) - 1)  # (B, 1, 1, ...)
        mask = self._drop(self._ones(shape, x.dtype))
        return x * mask


class Mlp(nn.Module):
    """Transformer MLP: Dense -> GELU -> Dropout -> Dense -> Dropout."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = int(hidden_features) if hidden_features is not None else int(in_features)
        out = int(out_features) if out_features is not None else int(in_features)

        self.fc1 = nn.Dense(int(in_features), hidden, has_bias=True)
        self.act = act_layer()
        self.fc2 = nn.Dense(hidden, out, has_bias=True)
        self.drop = nn.Dropout(keep_prob=1.0 - float(drop))

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class LayerNorm(nn.Module):
    """PyTorch-style LayerNorm over the last dimension."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.gamma = Parameter(ops.ones((int(dim),), ms.float32), name="gamma")
        self.beta = Parameter(ops.zeros((int(dim),), ms.float32), name="beta")
        self.reduce_mean = ops.ReduceMean(keep_dims=True)
        self.sqrt = ops.Sqrt()

    def forward(self, x: Tensor) -> Tensor:
        mean = self.reduce_mean(x, -1)
        var = self.reduce_mean((x - mean) ** 2, -1)
        x = (x - mean) / self.sqrt(var + self.eps)
        return x * self.gamma + self.beta


class Attention(nn.Module):
    """Multi-head self-attention for (B, N, C)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        dim = int(dim)
        num_heads = int(num_heads)
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = float(qk_scale) if qk_scale is not None else float(head_dim) ** -0.5

        self.qkv = nn.Dense(dim, dim * 3, has_bias=bool(qkv_bias))
        self.attn_drop = nn.Dropout(keep_prob=1.0 - float(attn_drop))

        self.proj = nn.Dense(dim, dim, has_bias=True)
        self.proj_drop = nn.Dropout(keep_prob=1.0 - float(proj_drop))

        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.unstack0 = ops.Unstack(axis=0)
        self.softmax = ops.Softmax(axis=-1)
        self.bmm_t = ops.BatchMatMul(transpose_b=True)
        self.bmm = ops.BatchMatMul()

    def forward(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        h = self.num_heads
        dh = c // h

        qkv = self.qkv(x)                                # (B, N, 3C)
        qkv = self.reshape(qkv, (b, n, 3, h, dh))        # (B, N, 3, H, Dh)
        qkv = self.transpose(qkv, (2, 0, 3, 1, 4))       # (3, B, H, N, Dh)
        q, k, v = self.unstack0(qkv)                     # each (B, H, N, Dh)

        # Flatten (B,H) for BatchMatMul (expects 3D)
        q2 = self.reshape(q, (b * h, n, dh))
        k2 = self.reshape(k, (b * h, n, dh))
        v2 = self.reshape(v, (b * h, n, dh))

        attn = self.bmm_t(q2, k2) * self.scale           # (B*H, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        out = self.bmm(attn, v2)                         # (B*H, N, Dh)
        out = self.reshape(out, (b, h, n, dh))            # (B, H, N, Dh)
        out = self.transpose(out, (0, 2, 1, 3))           # (B, N, H, Dh)
        out = self.reshape(out, (b, n, c))                # (B, N, C)

        out = self.proj_drop(self.proj(out))
        return out


class Block(nn.Module):
    """Transformer block: LN -> Attn -> residual; LN -> MLP -> residual."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(int(dim))
        self.attn = Attention(
            dim=int(dim),
            num_heads=int(num_heads),
            qkv_bias=bool(qkv_bias),
            qk_scale=qk_scale,
            attn_drop=float(attn_drop),
            proj_drop=float(drop),
        )
        self.drop_path = DropPath(float(drop_path)) if float(drop_path) > 0.0 else Identity()

        self.norm2 = norm_layer(int(dim))
        mlp_hidden = int(int(dim) * float(mlp_ratio))
        self.mlp = Mlp(int(dim), hidden_features=mlp_hidden, act_layer=act_layer, drop=float(drop))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding (Conv2d stem)."""

    def __init__(
        self,
        img_size: Union[int, Sequence[int]],
        patch_size: Union[int, Sequence[int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size

        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = int(self.grid_size[0] * self.grid_size[1])

        self.proj = nn.Conv2d(
            in_channels=int(in_chans),
            out_channels=int(embed_dim),
            kernel_size=patch_size,
            stride=patch_size,
            pad_mode="valid",
            has_bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        if (int(h), int(w)) != self.img_size:
            raise ValueError(f"Input {(int(h), int(w))} must match img_size={self.img_size}.")
        return self.proj(x)  # (B, D, H', W')


# ---------------------------
# MAE Decoder (Transformer)
# ---------------------------
class PretrainVisionTransformerDecoder(nn.Module):
    """Lightweight ViT decoder for masked token prediction (MAE-style)."""

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Type[nn.Module] = LayerNorm,
        out_dim: int = 768,  # patch_size*patch_size*in_chans by default
    ) -> None:
        super().__init__()
        depth = int(depth)

        # drop-path schedule
        if depth <= 1:
            dpr = [float(drop_path_rate)]
        else:
            dpr = [float(drop_path_rate) * i / (depth - 1) for i in range(depth)]

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=int(embed_dim),
                    num_heads=int(num_heads),
                    mlp_ratio=float(mlp_ratio),
                    qkv_bias=bool(qkv_bias),
                    qk_scale=qk_scale,
                    drop=float(drop_rate),
                    attn_drop=float(attn_drop_rate),
                    drop_path=float(dpr[i]),
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(int(embed_dim))
        self.head = nn.Dense(int(embed_dim), int(out_dim), has_bias=True)

        self._init_weights_all()

    def _init_weights_all(self) -> None:
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(initializer(XavierUniform(), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(initializer(Constant(0.0), cell.bias.shape, cell.bias.dtype))
            # For our custom LayerNorm, gamma/beta already default to (1,0);
            # if user passes MindSpore's nn.LayerNorm, keep its default.

    def forward(self, x: Tensor, return_token_num: int) -> Tensor:
        # x: (B, N_vis + N_mask, D)
        for blk in self.blocks:
            x = blk(x)
        x = x[:, -int(return_token_num):, :]      # only masked tokens
        x = self.norm(x)
        return self.head(x)                       # (B, N_mask, out_dim)


# ---------------------------
# Student ViT + MAE Branch
# ---------------------------
class VisionTransformer(nn.Module):
    """Mask-CDKD student backbone: ViT encoder + MAE reconstruction head.

    Forward:
        latent, pred = model(img, bool_masked_pos)
        - latent: tuple(features after each encoder block), each is (B, N_vis, D)
        - pred: reconstruction prediction for masked tokens, (B, N_mask, patch_dim)
    """

    def __init__(
        self,
        img_size: Union[int, Sequence[int]] = (1024, 1024),
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,          # ViT-S default
        depth: int = 12,               # ViT-S default
        num_heads: int = 6,            # ViT-S default
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Type[nn.Module] = LayerNorm,
        # MAE branch (paper: mask ratio 75% is external to model; decoder is 4-layer)
        decoder_embed_dim: int = 384,
        decoder_depth: int = 4,        # <-- align with paper
        decoder_num_heads: int = 8,
    ) -> None:
        super().__init__()

        self.img_size = to_2tuple(img_size)
        self.patch_size = int(patch_size)
        self.in_chans = int(in_chans)
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
        )
        self.num_patches = int(self.patch_embed.num_patches)
        self.grid_size = self.patch_embed.grid_size  # (H', W')

        # NOTE: no cls token (consistent with MAE-style encoder)
        self.pos_embed = Parameter(ops.zeros((1, self.num_patches, self.embed_dim), ms.float32), name="pos_embed")

        # Encoder blocks
        if self.depth <= 1:
            dpr = [float(drop_path_rate)]
        else:
            dpr = [float(drop_path_rate) * i / (self.depth - 1) for i in range(self.depth)]

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=self.embed_dim,
                    num_heads=int(num_heads),
                    mlp_ratio=float(mlp_ratio),
                    qkv_bias=bool(qkv_bias),
                    qk_scale=qk_scale,
                    drop=float(drop_rate),
                    attn_drop=float(attn_drop_rate),
                    drop_path=float(dpr[i]),
                    norm_layer=norm_layer,
                )
                for i in range(self.depth)
            ]
        )

        # MAE decoder branch
        self.decoder_embed_dim = int(decoder_embed_dim)
        self.decoder_embed = nn.Dense(self.embed_dim, self.decoder_embed_dim, has_bias=True)

        self.mask_token = Parameter(ops.zeros((1, 1, self.decoder_embed_dim), ms.float32), name="mask_token")

        # IMPORTANT FIX: do NOT hardcode batch dimension here.
        # Standard learnable positional embedding for decoder: (1, N, D_dec)
        self.pos_embed_decoder = Parameter(
            ops.zeros((1, self.num_patches, self.decoder_embed_dim), ms.float32),
            name="pos_embed_decoder",
        )

        patch_dim = self.patch_size * self.patch_size * self.in_chans  # e.g., 16*16*3=768
        self.decoder = PretrainVisionTransformerDecoder(
            embed_dim=self.decoder_embed_dim,
            depth=int(decoder_depth),               # <-- align with paper
            num_heads=int(decoder_num_heads),
            mlp_ratio=float(mlp_ratio),
            qkv_bias=bool(qkv_bias),
            qk_scale=qk_scale,
            drop_rate=float(drop_rate),
            attn_drop_rate=float(attn_drop_rate),
            drop_path_rate=0.0,                     # usually 0 for decoder
            norm_layer=norm_layer,
            out_dim=int(patch_dim),
        )

        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.concat1 = ops.Concat(axis=1)
        self.logical_not = ops.LogicalNot()

        self._init_parameters()

    def _init_parameters(self) -> None:
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.pos_embed_decoder, std=0.02)
        trunc_normal_(self.mask_token, std=0.02)

    @property
    def no_weight_decay(self) -> set:
        return {"pos_embed", "pos_embed_decoder", "mask_token"}

    # -------- Encoder / Decoder --------
    def forward_encoder(
        self,
        x: Tensor,
        bool_masked_pos: Tensor,
        return_all_blocks: bool = True,
    ) -> Tuple[Tensor, ...]:
        """Encode visible tokens. Return per-block outputs (B, N_vis, D)."""
        b = int(x.shape[0])
        mask = self.reshape(bool_masked_pos, (b, -1))  # True = masked, False = visible

        x = self.patch_embed(x)                         # (B, D, H', W')
        # (B, D, H', W') -> (B, D, N) -> (B, N, D)
        x = self.reshape(x, (b, self.embed_dim, -1))
        x = self.transpose(x, (0, 2, 1))
        x = x + self.pos_embed                          # (B, N, D)

        # Keep only visible tokens (match PyTorch: x[~mask].reshape(b, -1, D))
        inv_mask = self.logical_not(mask)
        x = x[inv_mask]
        x = self.reshape(x, (b, -1, int(x.shape[-1])))  # (B, N_vis, D)

        outs: List[Tensor] = []
        for blk in self.blocks:
            x = blk(x)
            if return_all_blocks:
                outs.append(x)

        return tuple(outs) if return_all_blocks else (x,)

    def forward_decoder(self, x_vis: Tensor, bool_masked_pos: Tensor) -> Tensor:
        """MAE decoder: predict pixels for masked patches only."""
        b = int(x_vis.shape[0])
        mask = self.reshape(bool_masked_pos, (b, -1))  # (B, N)

        # Project encoder tokens to decoder dim
        x_vis = self.decoder_embed(x_vis)              # (B, N_vis, D_dec)

        # Expand decoder pos emb to batch and split pos for vis/masked
        pos = ops.tile(self.pos_embed_decoder, (b, 1, 1))                 # (B, N, D_dec)
        inv_mask = self.logical_not(mask)

        pos_vis = pos[inv_mask]
        pos_vis = self.reshape(pos_vis, (b, -1, self.decoder_embed_dim))  # (B, N_vis, D_dec)

        pos_mask = pos[mask]
        pos_mask = self.reshape(pos_mask, (b, -1, self.decoder_embed_dim))  # (B, N_mask, D_dec)

        # Append mask tokens (masked tokens go to the end)
        x_full = self.concat1((x_vis + pos_vis, self.mask_token + pos_mask))  # (B, N_vis+N_mask, D_dec)

        # Only return masked token predictions
        return self.decoder(x_full, return_token_num=int(pos_mask.shape[1]))  # (B, N_mask, patch_dim)

    def forward(self, x: Tensor, bool_masked_pos: Tensor) -> Tuple[Tuple[Tensor, ...], Tensor]:
        """Return (latent_features, masked_patch_predictions)."""
        latent = self.forward_encoder(x, bool_masked_pos, return_all_blocks=True)
        pred = self.forward_decoder(latent[-1], bool_masked_pos)
        return latent, pred


# ---------------------------
# Convenience builder (paper student: ViT-S /16 @ 1024)
# ---------------------------
def build_mask_cdkd_student_vit_small_1024(
    in_chans: int = 3,
    patch_size: int = 16,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
) -> VisionTransformer:
    """Paper-default student: ViT-Small, 1024x1024 input."""
    return VisionTransformer(
        img_size=(1024, 1024),
        patch_size=int(patch_size),
        in_chans=int(in_chans),
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=float(drop_rate),
        attn_drop_rate=0.0,
        drop_path_rate=float(drop_path_rate),
        decoder_embed_dim=384,
        decoder_depth=4,          # paper: 4-layer decoder
        decoder_num_heads=8,
    )
