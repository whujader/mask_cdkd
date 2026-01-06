# mask_cdkd_losses.py
# -----------------------------------------------------------------------------
# Mask-CDKD Losses
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Literal, Union

import luojianet_ms as ms
import luojianet_ms.nn as nn
import luojianet_ms.ops as ops
from luojianet_ms import Tensor, Parameter

TensorMS = Tensor


# --------------------------- patchify & masked GT -----------------------------

def patchify_rgb(images: TensorMS, patch_size: int) -> TensorMS:
    """
    images: (B, 3, H, W)
    returns: (B, N, P) where N=(H/p)*(W/p), P=3*p*p
    """
    if len(images.shape) != 4:
        raise ValueError(f"patchify_rgb expects (B,3,H,W), got {tuple(images.shape)}")
    b, c, h, w = images.shape
    if int(c) != 3:
        raise ValueError(f"patchify_rgb expects 3 channels, got C={int(c)}")
    p = int(patch_size)
    if int(h) % p != 0 or int(w) % p != 0:
        raise ValueError(f"H,W must be divisible by patch_size={p}, got {(int(h), int(w))}")

    gh, gw = int(h) // p, int(w) // p
    reshape = ops.Reshape()
    transpose = ops.Transpose()

    x = reshape(images, (int(b), int(c), gh, p, gw, p))
    x = transpose(x, (0, 2, 4, 1, 3, 5))
    x = reshape(x, (int(b), gh * gw, int(c) * p * p))
    return x


def _ensure_2d_bool_mask(mask: TensorMS, batch_size: int) -> TensorMS:
    """
    mask: (N,) or (B,N), True=masked
    returns: (B,N) bool
    """
    cast = ops.Cast()
    if mask.dtype != ms.bool_:
        mask = cast(mask, ms.bool_)
    if len(mask.shape) == 1:
        # (N,) -> (1,N) -> (B,N)
        mask = ops.ExpandDims()(mask, 0)
        mask = ops.BroadcastTo((int(batch_size), int(mask.shape[1])))(mask)
    if len(mask.shape) != 2 or int(mask.shape[0]) != int(batch_size):
        raise ValueError(f"mask must be (N,) or (B,N), got {tuple(mask.shape)} with B={batch_size}")
    return mask


def masked_patch_targets(images: TensorMS, bool_masked_pos: TensorMS, patch_size: int) -> TensorMS:
    """
    images: (B,3,H,W)
    bool_masked_pos: (B,N) or (N,), True=masked
    returns: masked GT patches (B, N_mask, P)
    """
    b = int(images.shape[0])
    patches = patchify_rgb(images, patch_size)  # (B,N,P)
    mask = _ensure_2d_bool_mask(bool_masked_pos, b)  # (B,N)

    # n_mask per sample
    n_mask = ops.ReduceSum(keep_dims=False)(ops.Cast()(mask, ms.int32), 1)  # (B,)

    # require same number of masked tokens in batch
    # (MindSpore graph-friendly check still raises in eager/PyNative; kept to match PyTorch semantics)
    n0 = int(n_mask[0].asnumpy().item())
    if not bool((n_mask == n_mask[0]).all().asnumpy().item()):
        raise ValueError(
            "All samples in a batch must have the same number of masked tokens. "
            "Use a fixed-ratio mask generator."
        )

    p_dim = int(patches.shape[-1])

    # PyTorch behavior: patches[mask] flattens batch then reshapes back to (B, N_mask, P).
    flat = patches[mask]  # (B*N_mask, P)
    return ops.Reshape()(flat, (b, n0, p_dim))


# -------------------------- KD with projector bank ----------------------------

class Identity(nn.Module):
    def forward(self, x: TensorMS) -> TensorMS:
        return x


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
                proj.append(Identity())
            else:
                proj.append(nn.Dense(ds, dt, has_bias=False))
        self.proj = nn.ModuleList(proj)

        self.reduce_mean = ops.ReduceMean(keep_dims=False)
        self.stack = ops.Stack(axis=0)

    def forward(self, feats_s: Sequence[TensorMS], feats_t: Sequence[TensorMS]) -> TensorMS:
        if len(feats_s) != len(feats_t) or len(feats_s) != len(self.proj):
            raise ValueError("Feature list length mismatch with projector bank.")

        losses: List[TensorMS] = []
        for i, (s, t) in enumerate(zip(feats_s, feats_t)):
            if len(s.shape) != 3 or len(t.shape) != 3:
                raise ValueError(f"KD expects (B,N,D), got {tuple(s.shape)} and {tuple(t.shape)}")
            if int(s.shape[0]) != int(t.shape[0]) or int(s.shape[1]) != int(t.shape[1]):
                raise ValueError(f"KD requires matching (B,N): {tuple(s.shape)} vs {tuple(t.shape)}")

            s_aligned = self.proj[i](s)  # (B,N,Dt)
            if int(s_aligned.shape[-1]) != int(t.shape[-1]):
                raise ValueError("Projector output dim mismatch with teacher dim.")

            diff = s_aligned - t
            losses.append(self.reduce_mean(diff * diff))

        return self.reduce_mean(self.stack(tuple(losses)))


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

        self.pixel_mean = Tensor(sam_pixel_mean, ms.float32).view(1, 3, 1, 1)
        self.pixel_std = Tensor(sam_pixel_std, ms.float32).view(1, 3, 1, 1)

        self.reduce_mean = ops.ReduceMean(keep_dims=False)
        self.clamp_min = ops.Maximum()
        self.cast = ops.Cast()

    @property
    def state(self) -> str:
        return self._state

    def reset_state(self) -> None:
        self._state = self.schedule.init_state()

    def _update_state(self, r_value: float) -> None:
        self._state = self.schedule.step(self._state, float(r_value))

    def _prepare_recon_target_image(self, images: TensorMS) -> TensorMS:
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
        feats_s: Sequence[TensorMS],
        feats_t: Sequence[TensorMS],
        pred_s: TensorMS,  # (B, N_mask, patch_dim)
        pred_t: TensorMS,  # (B, N_mask, patch_dim)
        images: TensorMS,  # (B, 3, H, W)
        mask_s: TensorMS,  # (B, N) or (N,), True=masked
        mask_t: Optional[TensorMS] = None,
    ) -> Dict[str, TensorMS]:
        if mask_t is None:
            mask_t = mask_s

        # 1) KD
        loss_kd = self.kd(feats_s, feats_t)

        # 2) MAE targets (masked patches only)
        target_img = self._prepare_recon_target_image(images)
        gt_s = masked_patch_targets(target_img, mask_s, self.patch_size)  # (B,N_mask,P)
        gt_t = masked_patch_targets(target_img, mask_t, self.patch_size)

        if tuple(pred_s.shape) != tuple(gt_s.shape):
            raise ValueError(f"Student pred {tuple(pred_s.shape)} != GT {tuple(gt_s.shape)}")
        if tuple(pred_t.shape) != tuple(gt_t.shape):
            raise ValueError(f"Teacher pred {tuple(pred_t.shape)} != GT {tuple(gt_t.shape)}")

        diff_s = pred_s - gt_s
        diff_t = pred_t - gt_t
        loss_s_mae = self.reduce_mean(diff_s * diff_s)
        loss_t_mae = self.reduce_mean(diff_t * diff_t)

        # 3) r-driven schedule
        # r = clamp_min(loss_t / (loss_s + eps), 0.0)
        r = loss_t_mae / (loss_s_mae + Tensor(self.eps, ms.float32))
        r = self.clamp_min(r, Tensor(0.0, ms.float32))

        # update schedule state in python-side
        self._update_state(float(r.asnumpy().item()))
        lam1, lam2, lam3 = self.schedule.weights(self._state)

        loss_total = Tensor(lam1, ms.float32) * loss_kd + Tensor(lam2, ms.float32) * loss_t_mae + Tensor(lam3, ms.float32) * loss_s_mae

        # MindSpore doesn't have .detach(); return raw tensors like PyTorch "detached" scalars
        return {
            "loss_total": loss_total,
            "loss_kd": loss_kd,
            "loss_t_mae": loss_t_mae,
            "loss_s_mae": loss_s_mae,
            "r_t_over_s": r,
            "lambda_kd": Tensor(lam1, ms.float32),
            "lambda_t_mae": Tensor(lam2, ms.float32),
            "lambda_s_mae": Tensor(lam3, ms.float32),
        }
