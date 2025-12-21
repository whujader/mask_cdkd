# mask_cdkd_losses.py
# -----------------------------------------------------------------------------
# Mask-CDKD Losses (paper-aligned, open-source ready)
#
# Paper alignment:
#   - Feature KD (MSE): L_KD = ||T_l - S_l||_2^2 (implemented as mean squared error)
#   - MAE reconstruction on masked region Omega:
#       L_MAE = (1/|Omega|) * sum_{i in Omega} || I_i - I_hat_i ||_2^2
#     Here, both teacher/student decoders output ONLY masked token predictions:
#       pred: (B, N_mask, patch_dim)
#     so we patchify the image into (B, N, patch_dim) and select masked GT patches.
#   - Total loss with adaptation-state curriculum:
#       L_total = λ1 L_KD + λ2 L_T_MAE + λ3 L_S_MAE , λ1+λ2+λ3=1
#     r = L_T_MAE / L_S_MAE drives a 3-stage schedule (paper text):
#       Early : r > 0.85 -> (0.20, 0.40, 0.40)
#       Middle: r < 0.85 -> (0.60, 0.20, 0.20)
#       Late  : r > 0.95 -> (0.70, 0.15, 0.15)
#
# Notes:
#   - KD requires matching feature dimensions; this file provides lightweight
#     linear projectors to map student dims -> teacher dims (standard practice).
#   - For MAE target "pixel values", by default we assume SAM-style normalization
#     was applied to input images and we unnormalize back to [0,255] space before
#     patchifying. You can switch recon_target="normalized" if your training uses
#     normalized targets.
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Literal

import torch
import torch.nn as nn


Tensor = torch.Tensor


# --------------------------- patchify & masked GT -----------------------------

def patchify_rgb(images: Tensor, patch_size: int) -> Tensor:
    """
    images: (B, 3, H, W)
    returns: (B, N, P) where N=(H/p)*(W/p), P=3*p*p
    """
    if images.dim() != 4:
        raise ValueError(f"patchify_rgb expects (B,3,H,W), got {tuple(images.shape)}")
    b, c, h, w = images.shape
    if c != 3:
        raise ValueError(f"patchify_rgb expects 3 channels, got C={c}")
    p = int(patch_size)
    if h % p != 0 or w % p != 0:
        raise ValueError(f"H,W must be divisible by patch_size={p}, got {(h,w)}")

    gh, gw = h // p, w // p
    x = images.reshape(b, c, gh, p, gw, p)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.reshape(b, gh * gw, c * p * p)


def _ensure_2d_bool_mask(mask: Tensor, batch_size: int) -> Tensor:
    """
    mask: (N,) or (B,N), True=masked
    returns: (B,N) bool
    """
    if mask.dtype != torch.bool:
        mask = mask.to(torch.bool)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0).expand(batch_size, -1)
    if mask.dim() != 2 or mask.size(0) != batch_size:
        raise ValueError(f"mask must be (N,) or (B,N), got {tuple(mask.shape)} with B={batch_size}")
    return mask


def masked_patch_targets(images: Tensor, bool_masked_pos: Tensor, patch_size: int) -> Tensor:
    """
    images: (B,3,H,W)
    bool_masked_pos: (B,N) or (N,), True=masked
    returns: masked GT patches (B, N_mask, P)
    """
    b = images.size(0)
    patches = patchify_rgb(images, patch_size)  # (B,N,P)
    mask = _ensure_2d_bool_mask(bool_masked_pos, b)  # (B,N)

    n_mask = mask.sum(dim=1)
    if not torch.all(n_mask == n_mask[0]):
        raise ValueError(
            "All samples in a batch must have the same number of masked tokens. "
            "Use a fixed-ratio mask generator."
        )

    p_dim = patches.size(-1)
    return patches[mask].reshape(b, int(n_mask[0].item()), p_dim)


# -------------------------- KD with projector bank ----------------------------

class KDProjectorBank(nn.Module):
    """
    Lightweight per-layer projector: student_dim -> teacher_dim
    (required because ViT-S=384, ViT-L=1024).
    """

    def __init__(self, student_dims: Sequence[int], teacher_dims: Sequence[int]) -> None:
        super().__init__()
        if len(student_dims) != len(teacher_dims):
            raise ValueError("student_dims and teacher_dims must have the same length.")

        proj: List[nn.Module] = []
        for ds, dt in zip(student_dims, teacher_dims):
            ds, dt = int(ds), int(dt)
            if ds == dt:
                proj.append(nn.Identity())
            else:
                proj.append(nn.Linear(ds, dt, bias=False))
        self.proj = nn.ModuleList(proj)

    def forward(self, feats_s: Sequence[Tensor], feats_t: Sequence[Tensor]) -> Tensor:
        if len(feats_s) != len(feats_t) or len(feats_s) != len(self.proj):
            raise ValueError("Feature list length mismatch with projector bank.")

        losses: List[Tensor] = []
        for i, (s, t) in enumerate(zip(feats_s, feats_t)):
            if s.dim() != 3 or t.dim() != 3:
                raise ValueError(f"KD expects (B,N,D), got {tuple(s.shape)} and {tuple(t.shape)}")
            if s.size(0) != t.size(0) or s.size(1) != t.size(1):
                raise ValueError(f"KD requires matching (B,N): {tuple(s.shape)} vs {tuple(t.shape)}")

            s_aligned = self.proj[i](s)  # (B,N,Dt)
            if s_aligned.size(-1) != t.size(-1):
                raise ValueError("Projector output dim mismatch with teacher dim.")
            losses.append((s_aligned - t).pow(2).mean())

        return torch.stack(losses).mean()


# -------------------- adaptation-state curriculum (paper) ---------------------

@dataclass(frozen=True)
class AdaptationStateSchedule:
    early_threshold: float = 0.85
    late_threshold: float = 0.95

    # (λ1, λ2, λ3) = (KD, T_MAE, S_MAE)
    w_early: Tuple[float, float, float] = (0.20, 0.40, 0.40)
    w_middle: Tuple[float, float, float] = (0.60, 0.20, 0.20)
    w_late: Tuple[float, float, float] = (0.70, 0.15, 0.15)

    def init_state(self) -> str:
        return "early"

    def step(self, state: str, r_value: float) -> str:
        # state machine matches paper narrative: early -> middle -> late
        if state == "early":
            return "middle" if r_value < self.early_threshold else "early"
        if state == "middle":
            return "late" if r_value > self.late_threshold else "middle"
        if state == "late":
            return "late"
        raise ValueError(f"Unknown schedule state: {state}")

    def weights(self, state: str) -> Tuple[float, float, float]:
        if state == "early":
            return self.w_early
        if state == "middle":
            return self.w_middle
        if state == "late":
            return self.w_late
        raise ValueError(f"Unknown schedule state: {state}")


# ------------------------------ full loss module ------------------------------

class MaskCDKDLoss(nn.Module):
    """
    Full Mask-CDKD loss:
      L_total = λ1 L_KD + λ2 L_T_MAE + λ3 L_S_MAE
    """

    def __init__(
        self,
        *,
        student_dims: Sequence[int],
        teacher_dims: Sequence[int],
        patch_size: int = 16,
        schedule: Optional[AdaptationStateSchedule] = None,
        eps: float = 1e-6,
        recon_target: Literal["raw", "normalized"] = "raw",
        images_are_sam_normalized: bool = True,
        sam_pixel_mean: Tuple[float, float, float] = (123.675, 116.28, 103.53),
        sam_pixel_std: Tuple[float, float, float] = (58.395, 57.12, 57.375),
    ) -> None:
        super().__init__()
        self.kd = KDProjectorBank(student_dims=student_dims, teacher_dims=teacher_dims)
        self.patch_size = int(patch_size)
        self.schedule = schedule or AdaptationStateSchedule()
        self.eps = float(eps)

        self._state = self.schedule.init_state()

        self.recon_target = recon_target
        self.images_are_sam_normalized = bool(images_are_sam_normalized)
        self.register_buffer("pixel_mean", torch.tensor(sam_pixel_mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(sam_pixel_std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

    @property
    def state(self) -> str:
        return self._state

    def reset_state(self) -> None:
        self._state = self.schedule.init_state()

    @torch.no_grad()
    def _update_state(self, r_value: float) -> None:
        self._state = self.schedule.step(self._state, r_value)

    def _prepare_recon_target_image(self, images: Tensor) -> Tensor:
        # Paper defines MAE loss on pixel values in Omega.
        # Default: unnormalize SAM-normalized input back to raw [0,255] space.
        if self.recon_target == "normalized":
            return images
        if self.images_are_sam_normalized:
            return images * self.pixel_std + self.pixel_mean
        return images

    def forward(
        self,
        *,
        feats_s: Sequence[Tensor],
        feats_t: Sequence[Tensor],
        pred_s: Tensor,  # (B, N_mask, patch_dim)
        pred_t: Tensor,  # (B, N_mask, patch_dim)
        images: Tensor,  # (B, 3, H, W)
        mask_s: Tensor,  # (B, N) or (N,), True=masked
        mask_t: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if mask_t is None:
            mask_t = mask_s

        # 1) KD
        loss_kd = self.kd(feats_s, feats_t)

        # 2) MAE targets (masked patches only)
        target_img = self._prepare_recon_target_image(images)
        gt_s = masked_patch_targets(target_img, mask_s, self.patch_size)  # (B,N_mask,P)
        gt_t = masked_patch_targets(target_img, mask_t, self.patch_size)

        if pred_s.shape != gt_s.shape:
            raise ValueError(f"Student pred {tuple(pred_s.shape)} != GT {tuple(gt_s.shape)}")
        if pred_t.shape != gt_t.shape:
            raise ValueError(f"Teacher pred {tuple(pred_t.shape)} != GT {tuple(gt_t.shape)}")

        loss_s_mae = (pred_s - gt_s).pow(2).mean()
        loss_t_mae = (pred_t - gt_t).pow(2).mean()

        # 3) r-driven schedule
        r = (loss_t_mae.detach() / (loss_s_mae.detach() + self.eps)).clamp(min=0.0)
        self._update_state(float(r.item()))
        lam1, lam2, lam3 = self.schedule.weights(self._state)

        loss_total = lam1 * loss_kd + lam2 * loss_t_mae + lam3 * loss_s_mae

        return {
            "loss_total": loss_total,
            "loss_kd": loss_kd.detach(),
            "loss_t_mae": loss_t_mae.detach(),
            "loss_s_mae": loss_s_mae.detach(),
            "r_t_over_s": r.detach(),
            "lambda_kd": torch.tensor(lam1, device=loss_total.device),
            "lambda_t_mae": torch.tensor(lam2, device=loss_total.device),
            "lambda_s_mae": torch.tensor(lam3, device=loss_total.device),
        }
