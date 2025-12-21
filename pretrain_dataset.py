# pretrain_dataset.py
# -----------------------------------------------------------------------------
# Mask-CDKD Pretraining Dataset (Source-Free & Label-Free)
# - Single file: listing + loading + augmentation + SAM normalization + MAE masks
# - Paper-aligned defaults:
#     * img_size = 1024, patch_size = 16
#     * mask_ratio = 0.75
#     * unlabeled target-domain imagery
# - Teacher-safe masking:
#     SAM teacher (BHWC blocks) requires N_vis to be a perfect square.
#     We use a strided-grid mask with random shift (still random, but teacher-safe).
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union, Literal, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# SAM preprocessing (official convention)
_SAM_PIXEL_MEAN = (123.675, 116.28, 103.53)
_SAM_PIXEL_STD = (58.395, 57.12, 57.375)


def _list_image_files(
    root: Union[str, Path],
    extensions: Sequence[str],
    recursive: bool = True,
) -> List[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    exts = {e.lower().lstrip(".") for e in extensions}
    if recursive:
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower().lstrip(".") in exts]
    else:
        files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower().lstrip(".") in exts]

    files = sorted(files)
    if len(files) == 0:
        raise RuntimeError(f"No images found under {root} with extensions={sorted(exts)}")
    return files


def _to_sam_normalized_tensor(img_rgb: Image.Image) -> torch.Tensor:
    """
    PIL RGB -> float32 tensor (C,H,W) normalized with SAM mean/std.
    """
    if img_rgb.mode != "RGB":
        img_rgb = img_rgb.convert("RGB")

    arr = np.asarray(img_rgb, dtype=np.float32)  # (H,W,3) in [0,255]
    x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3,H,W)

    mean = torch.tensor(_SAM_PIXEL_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(_SAM_PIXEL_STD, dtype=torch.float32).view(3, 1, 1)
    return (x - mean) / std


def _load_rgb_image(
    path: Path,
    *,
    tiff_backend: Literal["gdal", "pil"] = "gdal",
    swap_bgr_for_4band_tiff: bool = True,
    bad_range: Literal["raise", "clip", "minmax"] = "raise",
) -> Image.Image:
    """
    Load an image file as PIL RGB.

    For tif/tiff:
      - Prefer GDAL (more correct for multi-band remote sensing imagery)
      - Special 4-band handling (aligned with your code):
          if num_bands == 4:
              take first 3 bands, then reorder [2,1,0] (BGR -> RGB)
      - Range check:
          if values not in [0,255], handle by `bad_range`

    For non-tiff:
      - Use PIL and convert to RGB.

    Returns:
      PIL.Image in RGB, uint8.
    """
    suffix = path.suffix.lower()
    if suffix not in {".tif", ".tiff"} or tiff_backend == "pil":
        return Image.open(path).convert("RGB")

    # --- GDAL branch for tif/tiff ---
    try:
        from osgeo import gdal  # type: ignore
    except Exception as e:
        raise ImportError(
            "Reading .tif/.tiff requires GDAL. "
            "Install it (e.g., conda install -c conda-forge gdal) "
            "or convert tiff images to png/jpg before training. "
            f"File: {path}"
        ) from e

    ds = gdal.Open(str(path))
    if ds is None:
        raise RuntimeError(f"GDAL failed to open: {path}")

    arr = ds.ReadAsArray()  # typically (C,H,W) for multi-band; (H,W) for single-band
    if arr is None:
        raise RuntimeError(f"GDAL ReadAsArray returned None: {path}")

    if arr.ndim == 2:
        # single band -> replicate to RGB
        arr = np.stack([arr, arr, arr], axis=0)
    elif arr.ndim == 3:
        pass
    else:
        raise RuntimeError(f"Unexpected tif array shape {arr.shape} for {path}")

    num_bands = arr.shape[0]

    # Select / reorder bands (paper uses RGB input; match your reference logic)
    if num_bands == 4:
        arr = arr[:3, :, :]
        if swap_bgr_for_4band_tiff:
            arr = arr[[2, 1, 0], :, :]  # BGR -> RGB (exactly like your code)
    elif num_bands >= 3:
        arr = arr[:3, :, :]
    else:
        # 1-2 bands -> replicate first band
        arr = np.stack([arr[0], arr[0], arr[0]], axis=0)

    # (C,H,W) -> (H,W,C), float32
    img = np.transpose(arr, (1, 2, 0)).astype(np.float32)

    # sanitize NaN/Inf
    if not np.isfinite(img).all():
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    vmin = float(img.min())
    vmax = float(img.max())

    # Many RS tiffs are uint16/float; decide what to do when out of [0,255]
    if vmin < 0.0 or vmax > 255.0:
        if bad_range == "raise":
            raise ValueError(
                f"TIFF value range out of [0,255]: min={vmin:.3f}, max={vmax:.3f}. "
                f"File: {path}. "
                "If your dataset contains high-bit-depth TIFFs, set bad_range='minmax' "
                "to rescale each image to [0,255] before SAM normalization."
            )
        elif bad_range == "clip":
            img = np.clip(img, 0.0, 255.0)
        elif bad_range == "minmax":
            denom = (vmax - vmin) if (vmax - vmin) > 1e-6 else 1.0
            img = (img - vmin) / denom * 255.0
        else:
            raise ValueError(f"Unknown bad_range mode: {bad_range}")

    img_u8 = np.clip(img, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(img_u8, mode="RGB")


@dataclass
class StridedRandomMaskGenerator:
    """
    Teacher-safe random masking (paper-aligned mask_ratio=0.75 for 1024/16).

    Teacher (SAM-style BHWC blocks) requires N_vis to be a perfect square.
    For 1024×1024 with patch_size=16 => grid=64, N=4096
    mask_ratio=0.75 => keep=1024 => keep_side=32 => stride=2
    """
    grid_size: int
    mask_ratio: float = 0.75
    random_shift: bool = True

    def __post_init__(self) -> None:
        if not (0.0 < self.mask_ratio < 1.0):
            raise ValueError(f"mask_ratio must be in (0,1), got {self.mask_ratio}")

        num_patches = self.grid_size * self.grid_size
        keep = int(round(num_patches * (1.0 - self.mask_ratio)))

        keep_side = int(math.isqrt(keep))
        if keep_side * keep_side != keep:
            raise ValueError(
                f"Visible token count must be a perfect square for teacher. "
                f"grid={self.grid_size}, N={num_patches}, keep={keep} is not square. "
                "Use paper default: img_size=1024, patch=16, mask_ratio=0.75."
            )
        if self.grid_size % keep_side != 0:
            raise ValueError(
                f"grid_size {self.grid_size} must be divisible by keep_side {keep_side}."
            )

        self.keep_side = keep_side
        self.stride = self.grid_size // self.keep_side

    def __call__(self) -> torch.Tensor:
        """
        Returns:
          mask: (N,) bool tensor, True=masked, False=visible
        """
        if self.random_shift and self.stride > 1:
            dh = int(torch.randint(0, self.stride, (1,)).item())
            dw = int(torch.randint(0, self.stride, (1,)).item())
        else:
            dh, dw = 0, 0

        mask_2d = torch.ones(self.grid_size, self.grid_size, dtype=torch.bool)
        mask_2d[dh :: self.stride, dw :: self.stride] = False
        return mask_2d.flatten()


class MaskCDKDPretrainDataset(Dataset):
    """
    Mask-CDKD pretraining dataset (unlabeled).

    Returns dict (paper-aligned):
      - images: (3, H, W) float32, SAM normalized
      - bool_masked_pos_student: (N,) bool, True=masked
      - bool_masked_pos_teacher: (N,) bool, True=masked (default identical to student)
      - name: file stem (for logging/debug)
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        img_size: int = 1024,
        patch_size: int = 16,
        mask_ratio: float = 0.75,
        same_mask_for_teacher: bool = True,
        recursive: bool = True,
        extensions: Sequence[str] = ("jpg", "jpeg", "png", "bmp", "tif", "tiff"),
        # tif handling
        tiff_backend: Literal["gdal", "pil"] = "gdal",
        swap_bgr_for_4band_tiff: bool = True,
        tiff_bad_range: Literal["raise", "clip", "minmax"] = "raise",
        # augmentation (paper: flips + small rotation + color jitter)
        use_augmentation: bool = True,
        rotation_degrees: float = 10.0,
        jitter: Tuple[float, float, float, float] = (0.2, 0.2, 0.2, 0.05),
    ) -> None:
        super().__init__()

        if img_size <= 0 or patch_size <= 0:
            raise ValueError("img_size and patch_size must be positive.")
        if img_size % patch_size != 0:
            raise ValueError(f"img_size must be divisible by patch_size: {img_size} % {patch_size} != 0")

        self.root_dir = Path(root_dir)
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.grid_size = self.img_size // self.patch_size
        self.same_mask_for_teacher = bool(same_mask_for_teacher)

        self.tiff_backend = tiff_backend
        self.swap_bgr_for_4band_tiff = bool(swap_bgr_for_4band_tiff)
        self.tiff_bad_range = tiff_bad_range

        self.files = _list_image_files(self.root_dir, extensions=extensions, recursive=recursive)

        self.mask_gen = StridedRandomMaskGenerator(
            grid_size=self.grid_size,
            mask_ratio=float(mask_ratio),
            random_shift=True,
        )

        # transforms operate on PIL.Image
        ops: List[transforms.Transform] = [
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        ]
        if use_augmentation:
            b, c, s, h = jitter
            ops.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(degrees=rotation_degrees, fill=0),
                    transforms.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h),
                ]
            )
        self.transform = transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.files[idx]
        name = path.stem

        img = _load_rgb_image(
            path,
            tiff_backend=self.tiff_backend,
            swap_bgr_for_4band_tiff=self.swap_bgr_for_4band_tiff,
            bad_range=self.tiff_bad_range,
        )
        img = self.transform(img)

        images = _to_sam_normalized_tensor(img)  # (3,H,W), float32

        mask_s = self.mask_gen()  # (N,) bool
        mask_t = mask_s.clone() if self.same_mask_for_teacher else self.mask_gen()

        return {
            "name": name,
            "images": images,
            "bool_masked_pos_student": mask_s,
            "bool_masked_pos_teacher": mask_t,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"root_dir={str(self.root_dir)!r}, num_files={len(self.files)}, "
            f"img_size={self.img_size}, patch_size={self.patch_size}, grid_size={self.grid_size}, "
            f"tiff_backend={self.tiff_backend!r}, tiff_bad_range={self.tiff_bad_range!r})"
        )
