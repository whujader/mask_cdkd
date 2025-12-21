# student_vit_mae.py
# -----------------------------------------------------------------------------
# Mask-CDKD Student Model (ViT-S + MAE branch)
# - Keep core architecture unchanged (ViT encoder + MAE-style decoder).
# - Adjusted to match paper description:
#   (1) MAE decoder depth = 4 (paper: "a four-layer decoder").
#   (2) pos_embed_decoder uses shape (1, N, D) (no batch-size hardcoding).
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union, Type, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Utils
# ---------------------------
def to_2tuple(x: Union[int, Sequence[int]]) -> Tuple[int, int]:
    if isinstance(x, (tuple, list)):
        if len(x) != 2:
            raise ValueError(f"Expected a 2-tuple/list, got len={len(x)}.")
        return int(x[0]), int(x[1])
    return int(x), int(x)


def _no_grad_trunc_normal_(
    tensor: torch.Tensor,
    mean: float,
    std: float,
    a: float,
    b: float,
) -> torch.Tensor:
    """Truncated normal init (PyTorch-compatible)."""
    def norm_cdf(x: float) -> float:
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# ---------------------------
# Core Blocks
# ---------------------------
class DropPath(nn.Module):
    """Stochastic Depth per sample (residual branch)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor.floor()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    """Transformer MLP: Linear -> GELU -> Dropout -> Linear -> Dropout."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = hidden_features or in_features
        out = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden, out)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


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
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.num_heads = int(num_heads)
        head_dim = dim // num_heads
        self.scale = float(qk_scale) if qk_scale is not None else head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, Dh)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj_drop(self.proj(x))
        return x


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
        norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, hidden_features=mlp_hidden, act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if (h, w) != self.img_size:
            raise ValueError(f"Input {(h, w)} must match img_size={self.img_size}.")
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
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        out_dim: int = 768,  # patch_size*patch_size*in_chans by default
    ) -> None:
        super().__init__()
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=float(dpr[i]),
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, out_dim)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, return_token_num: int) -> torch.Tensor:
        # x: (B, N_vis + N_mask, D)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:, -return_token_num:])  # only masked tokens
        return self.head(x)  # (B, N_mask, out_dim)


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
        norm_layer: Type[nn.Module] = nn.LayerNorm,
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
        self.num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size  # (H', W')

        # NOTE: no cls token (consistent with MAE-style encoder)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))

        # Encoder blocks
        dpr = torch.linspace(0, drop_path_rate, self.depth).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=self.embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=float(dpr[i]),
                    norm_layer=norm_layer,
                )
                for i in range(self.depth)
            ]
        )

        # MAE decoder branch
        self.decoder_embed_dim = int(decoder_embed_dim)
        self.decoder_embed = nn.Linear(self.embed_dim, self.decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))

        # IMPORTANT FIX: do NOT hardcode batch dimension here.
        # Standard learnable positional embedding for decoder: (1, N, D_dec)
        self.pos_embed_decoder = nn.Parameter(torch.zeros(1, self.num_patches, self.decoder_embed_dim))

        patch_dim = self.patch_size * self.patch_size * self.in_chans  # e.g., 16*16*3=768
        self.decoder = PretrainVisionTransformerDecoder(
            embed_dim=self.decoder_embed_dim,
            depth=int(decoder_depth),               # <-- align with paper
            num_heads=int(decoder_num_heads),
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=0.0,                     # usually 0 for decoder
            norm_layer=norm_layer,
            out_dim=int(patch_dim),
        )

        self._init_parameters()

    def _init_parameters(self) -> None:
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.pos_embed_decoder, std=0.02)
        trunc_normal_(self.mask_token, std=0.02)

    @property
    def no_weight_decay(self) -> set[str]:
        return {"pos_embed", "pos_embed_decoder", "mask_token"}

    # -------- Encoder / Decoder --------
    def forward_encoder(
        self,
        x: torch.Tensor,
        bool_masked_pos: torch.Tensor,
        return_all_blocks: bool = True,
    ) -> Tuple[torch.Tensor, ...]:
        """Encode visible tokens. Return per-block outputs (B, N_vis, D)."""
        b = x.size(0)
        mask = bool_masked_pos.view(b, -1)  # True = masked, False = visible

        x = self.patch_embed(x)                     # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)            # (B, N, D)
        x = x + self.pos_embed                      # (B, N, D)

        # Keep only visible tokens
        x = x[~mask].reshape(b, -1, x.shape[-1])    # (B, N_vis, D)

        outs: List[torch.Tensor] = []
        for blk in self.blocks:
            x = blk(x)
            if return_all_blocks:
                outs.append(x)

        return tuple(outs) if return_all_blocks else (x,)

    def forward_decoder(self, x_vis: torch.Tensor, bool_masked_pos: torch.Tensor) -> torch.Tensor:
        """MAE decoder: predict pixels for masked patches only."""
        b = x_vis.size(0)
        mask = bool_masked_pos.view(b, -1)  # (B, N)

        # Project encoder tokens to decoder dim
        x_vis = self.decoder_embed(x_vis)   # (B, N_vis, D_dec)

        # Expand decoder pos emb to batch and split pos for vis/masked
        pos = self.pos_embed_decoder.expand(b, -1, -1)  # (B, N, D_dec)
        pos_vis = pos[~mask].reshape(b, -1, self.decoder_embed_dim)   # (B, N_vis, D_dec)
        pos_mask = pos[mask].reshape(b, -1, self.decoder_embed_dim)   # (B, N_mask, D_dec)

        # Append mask tokens (masked tokens go to the end)
        x_full = torch.cat([x_vis + pos_vis, self.mask_token + pos_mask], dim=1)  # (B, N_vis+N_mask, D_dec)

        # Only return masked token predictions
        return self.decoder(x_full, return_token_num=pos_mask.size(1))  # (B, N_mask, patch_dim)

    def forward(
        self,
        x: torch.Tensor,
        bool_masked_pos: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
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
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=drop_rate,
        attn_drop_rate=0.0,
        drop_path_rate=drop_path_rate,
        decoder_embed_dim=384,
        decoder_depth=4,          # paper: 4-layer decoder
        decoder_num_heads=8,
    )
