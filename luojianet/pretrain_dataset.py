# pretrain_dataset.py
# -----------------------------------------------------------------------------
# Mask-CDKD Pretraining Dataset (MindSpore, Source-Free & Label-Free)
# - listing + loading + augmentation + SAM normalization + MAE masks
# - Paper-aligned defaults:
#     * img_size = 1024, patch_size = 16
#     * mask_ratio = 0.75
# - Teacher-safe masking:
#     SAM teacher (BHWC blocks) requires N_vis to be a perfect square.
#     We use a strided-grid mask with random shift.
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union, Literal, Optional

import numpy as np
from PIL import Image, ImageEnhance

import luojianet_ms as ms
from luojianet_ms import Tensor


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


def _to_sam_normalized_tensor(img_rgb: Image.Image) -> Tensor:
    """
    PIL RGB -> float32 tensor (C,H,W) normalized with SAM mean/std.
    """
    if img_rgb.mode != "RGB":
        img_rgb = img_rgb.convert("RGB")

    arr = np.asarray(img_rgb, dtype=np.float32)  # (H,W,3) in [0,255]
    x = np.transpose(arr, (2, 0, 1)).copy()      # (3,H,W)

    mean = np.asarray(_SAM_PIXEL_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(_SAM_PIXEL_STD, dtype=np.float32).reshape(3, 1, 1)
    x = (x - mean) / std
    return Tensor(x, ms.float32)


def _load_rgb_image(
    path: Path,
    *,
    tiff_backend: Literal["gdal", "pil"] = "gdal",
    swap_bgr_for_4band_tiff: bool = True,
    bad_range: Literal["raise", "clip", "minmax"] = "raise",
) -> Image.Image:
    """
    Load an image file as PIL RGB (uint8).

    For tif/tiff:
      - Prefer GDAL
      - 4-band: take first 3 bands, then reorder [2,1,0] (BGR -> RGB) if enabled
      - Range check/handle by `bad_range`
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

    arr = ds.ReadAsArray()  # (C,H,W) for multi-band; (H,W) for single-band
    if arr is None:
        raise RuntimeError(f"GDAL ReadAsArray returned None: {path}")

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=0)
    elif arr.ndim == 3:
        pass
    else:
        raise RuntimeError(f"Unexpected tif array shape {arr.shape} for {path}")

    num_bands = int(arr.shape[0])

    if num_bands == 4:
        arr = arr[:3, :, :]
        if swap_bgr_for_4band_tiff:
            arr = arr[[2, 1, 0], :, :]  # BGR -> RGB
    elif num_bands >= 3:
        arr = arr[:3, :, :]
    else:
        arr = np.stack([arr[0], arr[0], arr[0]], axis=0)

    img = np.transpose(arr, (1, 2, 0)).astype(np.float32)  # (H,W,3)

    if not np.isfinite(img).all():
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    vmin = float(img.min())
    vmax = float(img.max())

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


# --------------------------- PIL augmentations (no torchvision) ---------------------------

def _pil_resize(img: Image.Image, size_hw: Tuple[int, int]) -> Image.Image:
    # PIL expects (W,H)
    h, w = int(size_hw[0]), int(size_hw[1])
    return img.resize((w, h), resample=Image.BILINEAR)


def _pil_random_hflip(img: Image.Image, p: float = 0.5) -> Image.Image:
    if float(np.random.rand()) < float(p):
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def _pil_random_vflip(img: Image.Image, p: float = 0.5) -> Image.Image:
    if float(np.random.rand()) < float(p):
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


def _pil_random_rotation(img: Image.Image, degrees: float, fill: int = 0) -> Image.Image:
    deg = float(degrees)
    if deg <= 0.0:
        return img
    angle = float(np.random.uniform(-deg, deg))
    # keep size, fill background with 0
    try:
        return img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=fill)
    except TypeError:
        # older PIL may not support fillcolor
        return img.rotate(angle, resample=Image.BILINEAR, expand=False)


def _adjust_hue(img: Image.Image, hue_factor: float) -> Image.Image:
    # hue_factor in [-0.5, 0.5], like torchvision. Here jitter uses small value (e.g., 0.05).
    if hue_factor == 0.0:
        return img
    if img.mode != "RGB":
        img = img.convert("RGB")

    hsv = img.convert("HSV")
    arr = np.array(hsv, dtype=np.uint8)
    # Hue channel [0,255], shift by hue_factor*255
    shift = int(round(float(hue_factor) * 255.0))
    arr[..., 0] = (arr[..., 0].astype(np.int16) + shift) % 256
    hsv2 = Image.fromarray(arr, mode="HSV")
    return hsv2.convert("RGB")


def _pil_color_jitter(
    img: Image.Image,
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
) -> Image.Image:
    # Factors sampled like torchvision: [max(0,1-x), 1+x]
    ops_list = []

    if brightness > 0:
        factor = float(np.random.uniform(max(0.0, 1.0 - brightness), 1.0 + brightness))
        ops_list.append(("brightness", factor))
    if contrast > 0:
        factor = float(np.random.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast))
        ops_list.append(("contrast", factor))
    if saturation > 0:
        factor = float(np.random.uniform(max(0.0, 1.0 - saturation), 1.0 + saturation))
        ops_list.append(("saturation", factor))
    if hue > 0:
        factor = float(np.random.uniform(-hue, hue))
        ops_list.append(("hue", factor))

    # torchvision randomizes the order
    if len(ops_list) > 1:
        np.random.shuffle(ops_list)

    out = img
    for name, factor in ops_list:
        if name == "brightness":
            out = ImageEnhance.Brightness(out).enhance(factor)
        elif name == "contrast":
            out = ImageEnhance.Contrast(out).enhance(factor)
        elif name == "saturation":
            out = ImageEnhance.Color(out).enhance(factor)
        elif name == "hue":
            out = _adjust_hue(out, factor)
        else:
            raise ValueError(f"Unknown jitter op: {name}")
    return out


# --------------------------- teacher-safe mask generator ---------------------------

@dataclass
class StridedRandomMaskGenerator:
    """
    Teacher-safe random masking (paper-aligned mask_ratio=0.75 for 1024/16).

    For 1024×1024 with patch_size=16 => grid=64, N=4096
    mask_ratio=0.75 => keep=1024 => keep_side=32 => stride=2
    """
    grid_size: int
    mask_ratio: float = 0.75
    random_shift: bool = True

    def __post_init__(self) -> None:
        if not (0.0 < float(self.mask_ratio) < 1.0):
            raise ValueError(f"mask_ratio must be in (0,1), got {self.mask_ratio}")

        num_patches = int(self.grid_size) * int(self.grid_size)
        keep = int(round(num_patches * (1.0 - float(self.mask_ratio))))

        keep_side = int(math.isqrt(keep))
        if keep_side * keep_side != keep:
            raise ValueError(
                f"Visible token count must be a perfect square for teacher. "
                f"grid={self.grid_size}, N={num_patches}, keep={keep} is not square. "
                "Use paper default: img_size=1024, patch=16, mask_ratio=0.75."
            )
        if int(self.grid_size) % keep_side != 0:
            raise ValueError(f"grid_size {self.grid_size} must be divisible by keep_side {keep_side}.")

        self.keep_side = keep_side
        self.stride = int(self.grid_size) // int(self.keep_side)

    def __call__(self) -> Tensor:
        """
        Returns:
          mask: (N,) bool tensor, True=masked, False=visible
        """
        if bool(self.random_shift) and int(self.stride) > 1:
            dh = int(np.random.randint(0, int(self.stride)))
            dw = int(np.random.randint(0, int(self.stride)))
        else:
            dh, dw = 0, 0

        mask_2d = np.ones((int(self.grid_size), int(self.grid_size)), dtype=np.bool_)
        mask_2d[dh :: int(self.stride), dw :: int(self.stride)] = False
        return Tensor(mask_2d.reshape(-1), ms.bool_)


# --------------------------- dataset ---------------------------

class MaskCDKDPretrainDataset:
    """
    Mask-CDKD pretraining dataset (unlabeled).

    Returns dict (paper-aligned):
      - images: (3, H, W) float32, SAM normalized
      - bool_masked_pos_student: (N,) bool, True=masked
      - bool_masked_pos_teacher: (N,) bool, True=masked (default identical to student)
      - name: file stem (for logging/debug)

    Note:
      This class is a Python-style dataset (like PyTorch Dataset) but returns MindSpore Tensors.
      You can wrap it by luojianet_ms.dataset.GeneratorDataset for training.
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

        if int(img_size) <= 0 or int(patch_size) <= 0:
            raise ValueError("img_size and patch_size must be positive.")
        if int(img_size) % int(patch_size) != 0:
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

        self.use_augmentation = bool(use_augmentation)
        self.rotation_degrees = float(rotation_degrees)
        self.jitter = tuple(float(x) for x in jitter)

    def __len__(self) -> int:
        return len(self.files)

    def _transform(self, img: Image.Image) -> Image.Image:
        # Resize first (paper code does this)
        img = _pil_resize(img, (self.img_size, self.img_size))

        if self.use_augmentation:
            b, c, s, h = self.jitter
            img = _pil_random_hflip(img, p=0.5)
            img = _pil_random_vflip(img, p=0.5)
            img = _pil_random_rotation(img, degrees=self.rotation_degrees, fill=0)
            img = _pil_color_jitter(img, brightness=b, contrast=c, saturation=s, hue=h)
        return img

    def __getitem__(self, idx: int) -> Dict[str, Union[str, Tensor]]:
        path = self.files[int(idx)]
        name = path.stem

        img = _load_rgb_image(
            path,
            tiff_backend=self.tiff_backend,
            swap_bgr_for_4band_tiff=self.swap_bgr_for_4band_tiff,
            bad_range=self.tiff_bad_range,
        )
        img = self._transform(img)

        images = _to_sam_normalized_tensor(img)  # (3,H,W), float32

        mask_s = self.mask_gen()  # (N,) bool
        mask_t = mask_s if self.same_mask_for_teacher else self.mask_gen()

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
