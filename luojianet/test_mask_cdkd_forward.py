# test_mask_cdkd_forward.py
import luojianet_ms as ms
import luojianet_ms.ops as ops
from luojianet_ms import Tensor

from teacher_sam_vit_mmoa import build_sam_vit_l_teacher_mmoa
from student_vit_mae import build_mask_cdkd_student_vit_small_1024
from mask_cdkd_distiller import MaskCDKDDistiller


def make_strided_mask(batch_size: int, grid: int, stride: int = 2) -> Tensor:
    """
    Paper-friendly structured mask:
      - True  = masked
      - False = visible
    For grid=64 and stride=2:
      visible = 32*32 = 1024  (25%)
      masked  = 4096-1024=3072 (75%)
    Returns: (B, N) bool
    """
    # mask_2d: True everywhere then set visible positions to False
    mask_2d = ops.ones((int(grid), int(grid)), ms.bool_)
    # MindSpore supports slicing assignment in PyNative mode; keep test simple.
    mask_2d[:: int(stride), :: int(stride)] = False
    mask_1d = ops.Reshape()(mask_2d, (-1,))  # (N,)
    mask_1d = ops.ExpandDims()(mask_1d, 0)   # (1,N)
    mask = ops.tile(mask_1d, (int(batch_size), 1))  # (B,N)
    return mask


def main():
    # Device setup
    # If you run on GPU: export DEVICE_TARGET=GPU before running
    # ms.set_context(mode=ms.PYNATIVE_MODE) is recommended for this quick test.
    ms.set_context(mode=ms.PYNATIVE_MODE)

    B = 1
    IMG_SIZE = 1024
    PATCH = 16
    GRID = IMG_SIZE // PATCH
    N = GRID * GRID

    # random input
    images = ops.standard_normal((B, 3, IMG_SIZE, IMG_SIZE)).astype(ms.float32)

    # structured 75% mask
    bool_mask = make_strided_mask(B, GRID, stride=2)
    assert tuple(bool_mask.shape) == (B, N)

    num_mask = int(ops.ReduceSum()(ops.Cast()(bool_mask[0], ms.int32)).asnumpy().item())
    num_vis = int(ops.ReduceSum()(ops.Cast()(ops.LogicalNot()(bool_mask[0]), ms.int32)).asnumpy().item())
    print(f"[Mask] N={N}, masked={num_mask}, visible={num_vis} (visible should be a perfect square)")

    # build models
    teacher = build_sam_vit_l_teacher_mmoa()
    student = build_mask_cdkd_student_vit_small_1024(in_chans=3, patch_size=PATCH)

    # distiller
    distiller = MaskCDKDDistiller(teacher=teacher, student=student, freeze_teacher_backbone=True)

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
