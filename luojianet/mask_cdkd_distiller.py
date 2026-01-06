# mask_cdkd_distiller.py
# -----------------------------------------------------------------------------
# Mask-CDKD Distillation Wrapper
# Teacher: SAM-ViT-L + MMoA, Student: ViT-S + MAE
# Paper-consistent:
#   - Distill layers: Teacher {6,12,18} <-> Student {3,6,9} (1-based in paper).
#   - Teacher backbone frozen; trainable parts: MMoA + teacher MAE branch.
# -----------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import luojianet_ms as ms
import luojianet_ms.nn as nn
import luojianet_ms.ops as ops
from luojianet_ms import Tensor

from teacher_sam_vit_mmoa import ImageEncoderViTMMoA, build_sam_vit_l_teacher_mmoa
from student_vit_mae import VisionTransformer, build_mask_cdkd_student_vit_small_1024

__all__ = [
    "TeacherConfig",
    "StudentConfig",
    "DistillConfig",
    "MaskCDKDDistiller",
]


# -----------------------------------------------------------------------------
# Configs
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class TeacherConfig:
    """Paper-default teacher: SAM ViT-L (1024 dim, 24 blocks) + MMoA + 4-layer MAE decoder."""
    img_size: int = 1024
    patch_size: int = 16
    in_chans: int = 3

    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0

    window_size: int = 14
    global_attn_indexes: Tuple[int, ...] = (5, 11, 17, 23)

    # MAE branch
    decoder_embed_dim: int = 1024
    decoder_depth: int = 4
    decoder_num_heads: int = 8

    teacher_pretrained: Optional[Union[str, Dict[str, Tensor]]] = "/data02/ZhangZhan/sdy_cross_domain/sam_vit_l_0b3195.pth"  # optional ckpt


@dataclass(frozen=True)
class StudentConfig:
    """Paper-default student: ViT-S (384 dim, 12 blocks) + 4-layer MAE decoder, input 1024."""
    img_size: Tuple[int, int] = (1024, 1024)
    patch_size: int = 16
    in_chans: int = 3

    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0

    # MAE branch
    decoder_embed_dim: int = 384
    decoder_depth: int = 4
    decoder_num_heads: int = 8


@dataclass(frozen=True)
class DistillConfig:
    """Paper distillation layer mapping (1-based indices)."""
    teacher_layers: Tuple[int, int, int] = (6, 12, 18)   # paper: teacher blocks {6,12,18} in 24-layer ViT-L
    student_layers: Tuple[int, int, int] = (3, 6, 9)     # paper: student blocks {3,6,9} in 12-layer ViT-S


# -----------------------------------------------------------------------------
# Distiller
# -----------------------------------------------------------------------------
class MaskCDKDDistiller(nn.Module):
    """Mask-CDKD teacher-student wrapper (MindSpore).

    construct returns:
        - student_feats: list of selected student features (B, N_vis, D_s)
        - teacher_feats: list of selected teacher features (B, N_vis, D_t)
        - student_mae_pred: (B, N_mask, patch_dim)
        - teacher_mae_pred: (B, N_mask, patch_dim)

    Notes:
        - Teacher is NOT wrapped in stop_gradient because paper uses bidirectional distillation
          (teacher MMoA + MAE branch receive gradients).
    """

    # Teacher trainable keywords (paper-consistent): MMoA + MAE branch
    _TEACHER_TRAINABLE_KEYWORDS = (
        "adapter_fine",
        "adapter_coarse",
        "gate",
        "decoder",            # decoder blocks + head
        "decoder_embed",
        "mask_token",
        "pos_embed_decoder",
    )

    def __init__(
        self,
        teacher: Optional[ImageEncoderViTMMoA] = None,
        student: Optional[VisionTransformer] = None,
        teacher_cfg: TeacherConfig = TeacherConfig(),
        student_cfg: StudentConfig = StudentConfig(),
        distill_cfg: DistillConfig = DistillConfig(),
        freeze_teacher_backbone: bool = True,
    ) -> None:
        super().__init__()

        # Build teacher / student if not provided
        self.teacher = teacher if teacher is not None else self._build_teacher(teacher_cfg)
        self.student = student if student is not None else self._build_student(student_cfg)

        # Load teacher ckpt if provided (NOTE: MindSpore expects .ckpt; PyTorch .pth
        # needs offline conversion. Function provided for completeness.)
        if teacher_cfg.teacher_pretrained is not None:
            self.load_teacher_checkpoint(teacher_cfg.teacher_pretrained)

        # Store distillation mapping (convert to 0-based indices)
        self.teacher_layer_idx0 = tuple([int(i) - 1 for i in distill_cfg.teacher_layers])
        self.student_layer_idx0 = tuple([int(i) - 1 for i in distill_cfg.student_layers])

        self._validate_layer_mapping()

        if bool(freeze_teacher_backbone):
            self.freeze_teacher_backbone()

    @staticmethod
    def _build_teacher(cfg: TeacherConfig) -> ImageEncoderViTMMoA:
        # Use provided builder (paper-default)
        # If you want cfg-driven construction, replace with ImageEncoderViTMMoA(...)
        return build_sam_vit_l_teacher_mmoa()

    @staticmethod
    def _build_student(cfg: StudentConfig) -> VisionTransformer:
        # Use provided builder (paper-default)
        return build_mask_cdkd_student_vit_small_1024(in_chans=cfg.in_chans, patch_size=cfg.patch_size)

    def _validate_layer_mapping(self) -> None:
        if len(self.teacher_layer_idx0) != len(self.student_layer_idx0):
            raise ValueError("teacher_layers and student_layers must have same length.")

        t_blocks = getattr(self.teacher, "blocks", None)
        s_blocks = getattr(self.student, "blocks", None)
        if t_blocks is None or s_blocks is None:
            return

        t_depth = len(self.teacher.blocks)
        s_depth = len(self.student.blocks)

        for i in self.teacher_layer_idx0:
            if int(i) < 0 or int(i) >= int(t_depth):
                raise ValueError(f"Teacher layer index out of range: {int(i)} (depth={t_depth}).")
        for i in self.student_layer_idx0:
            if int(i) < 0 or int(i) >= int(s_depth):
                raise ValueError(f"Student layer index out of range: {int(i)} (depth={s_depth}).")

    # -------------------------
    # Checkpoint loading
    # -------------------------
    def load_teacher_checkpoint(self, ckpt: Union[str, Dict[str, Tensor]]) -> None:
        """
        Load teacher weights.

        IMPORTANT:
          - MindSpore loads *.ckpt via ms.load_checkpoint.
          - If ckpt is a PyTorch *.pth, you must convert it to MindSpore format externally.
        """
        if isinstance(ckpt, str):
            if ckpt.endswith(".ckpt"):
                param_dict = ms.load_checkpoint(ckpt)
            else:
                raise ValueError(
                    f"Given checkpoint path looks like PyTorch ({ckpt}). "
                    "Please convert .pth -> .ckpt first, then load with MindSpore."
                )
        elif isinstance(ckpt, dict):
            # assume MindSpore param_dict already
            param_dict = ckpt
        else:
            raise TypeError("ckpt must be a path str or a MindSpore parameter dict.")

        # Tolerant key cleaning (similar to PyTorch version)
        cleaned: Dict[str, Tensor] = {}
        for k, v in param_dict.items():
            key = k
            if key.startswith("module."):
                key = key.replace("module.", "", 1)
            if key.startswith("image_encoder."):
                key = key.replace("image_encoder.", "", 1)
            cleaned[key] = v

        # Load non-strictly
        ms.load_param_into_net(self.teacher, cleaned, strict_load=False)

    # -------------------------
    # Teacher freezing
    # -------------------------
    def freeze_teacher_backbone(self) -> None:
        """Freeze teacher backbone; keep MMoA + MAE branch trainable (paper-consistent)."""
        for p in self.teacher.get_parameters():
            name = p.name
            trainable = any(k in name for k in self._TEACHER_TRAINABLE_KEYWORDS)
            p.requires_grad = bool(trainable)

    # -------------------------
    # Feature selection
    # -------------------------
    @staticmethod
    def _select_layers(latent: Sequence[Tensor], idx0: Sequence[int]) -> List[Tensor]:
        if len(latent) == 0:
            raise ValueError("latent features is empty.")
        return [latent[int(i)] for i in idx0]

    # -------------------------
    # Forward
    # -------------------------
    def forward(
        self,
        images: Tensor,
        bool_masked_pos_student: Tensor,
        bool_masked_pos_teacher: Optional[Tensor] = None,
    ) -> Tuple[List[Tensor], List[Tensor], Tensor, Tensor]:
        """
        Args:
            images: (B, 3, H, W)
            bool_masked_pos_student: (B, N) bool, True=masked
            bool_masked_pos_teacher: (B, N) bool, True=masked; if None, use student mask
        """
        if bool_masked_pos_teacher is None:
            bool_masked_pos_teacher = bool_masked_pos_student

        # student forward
        latent_s, mae_pred_s = self.student(images, bool_masked_pos_student)
        feats_s = self._select_layers(latent_s, self.student_layer_idx0)

        # teacher forward (no stop_gradient; bidirectional distillation)
        latent_t, mae_pred_t = self.teacher(images, bool_masked_pos_teacher)
        feats_t = self._select_layers(latent_t, self.teacher_layer_idx0)

        return feats_s, feats_t, mae_pred_s, mae_pred_t
