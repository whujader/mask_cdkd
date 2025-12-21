# test_mask_cdkd_forward.py
import torch

from teacher_sam_vit_mmoa import build_sam_vit_l_teacher_mmoa
from student_vit_mae import build_mask_cdkd_student_vit_small_1024
from mask_cdkd_distiller import MaskCDKDDistiller


def make_strided_mask(batch_size: int, grid: int, stride: int = 2, device: str = "cpu") -> torch.Tensor:
    """
    Paper-friendly structured mask:
      - True  = masked
      - False = visible
    For grid=64 and stride=2:
      visible = 32*32 = 1024  (25%)
      masked  = 4096-1024=3072 (75%)
    """
    mask_2d = torch.ones(grid, grid, dtype=torch.bool, device=device)
    mask_2d[::stride, ::stride] = False  # visible positions
    mask_1d = mask_2d.reshape(-1)        # (N,)
    return mask_1d.unsqueeze(0).repeat(batch_size, 1)  # (B,N)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = 1
    IMG_SIZE = 1024
    PATCH = 16
    GRID = IMG_SIZE // PATCH
    N = GRID * GRID

    # random input
    images = torch.randn(B, 3, IMG_SIZE, IMG_SIZE, device=device)

    # structured 75% mask
    bool_mask = make_strided_mask(B, GRID, stride=2, device=device)
    assert bool_mask.shape == (B, N)
    num_mask = bool_mask[0].sum().item()
    num_vis = (~bool_mask[0]).sum().item()
    print(f"[Mask] N={N}, masked={num_mask}, visible={num_vis} (visible should be a perfect square)")

    # build models
    teacher = build_sam_vit_l_teacher_mmoa().to(device)
    student = build_mask_cdkd_student_vit_small_1024(in_chans=3, patch_size=PATCH).to(device)

    # distiller
    distiller = MaskCDKDDistiller(teacher=teacher, student=student, freeze_teacher_backbone=True).to(device)

    # forward
    feats_s, feats_t, mae_s, mae_t = distiller(images, bool_mask, bool_mask)

    print("\n[Student selected features]")
    for i, f in enumerate(feats_s):
        print(f"  S[{i}] = {tuple(f.shape)}")  # (B, N_vis, D_s)

    print("\n[Teacher selected features]")
    for i, f in enumerate(feats_t):
        print(f"  T[{i}] = {tuple(f.shape)}")  # (B, N_vis, D_t)

    print("\n[MAE predictions]")
    print("  mae_pred_s =", tuple(mae_s.shape))  # (B, N_mask, patch_dim)
    print("  mae_pred_t =", tuple(mae_t.shape))  # (B, N_mask, patch_dim)


if __name__ == "__main__":
    main()
