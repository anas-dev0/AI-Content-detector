"""Preprocessing that respects the fingerprint: crop (never resize) and uniform JPEG re-encoding.

Why it matters
- Resizing an image re-samples every pixel and wipes out the up-sampling traces we want to detect.
  So we always cut a 224x224 square out of the original pixels.
- In GenImage the real photos are JPEG and the fakes are PNG. A model can learn "JPEG vs PNG"
  instead of "real vs fake" (Grommelt et al. 2024). Re-encoding *everything* at the same JPEG
  quality removes that shortcut.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def iter_images(folder: Path | str, recursive: bool = True) -> list[Path]:
    """Sorted list of image files under `folder`."""
    folder = Path(folder)
    it = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _pad_to_min_size(im: Image.Image, size: int) -> Image.Image:
    """Reflect-pad an image that is smaller than `size` in either dimension. Never resizes."""
    w, h = im.size
    if w >= size and h >= size:
        return im
    arr = np.asarray(im)
    pad_h, pad_w = max(0, size - h), max(0, size - w)
    pads = ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2))
    if arr.ndim == 3:
        pads = pads + ((0, 0),)
    # 'reflect' needs pad < dim; fall back to 'symmetric' (pad <= dim) then 'edge' for tiny images
    for mode in ("reflect", "symmetric", "edge"):
        try:
            arr = np.pad(arr, pads, mode=mode)
            break
        except ValueError:
            continue
    return Image.fromarray(arr)


def center_crop(im: Image.Image, size: int = 224) -> Image.Image:
    """Central size x size crop from the original pixels (pads by reflection if too small)."""
    im = _pad_to_min_size(im, size)
    w, h = im.size
    left, top = (w - size) // 2, (h - size) // 2
    return im.crop((left, top, left + size, top + size))


def random_crop(im: Image.Image, size: int = 224, rng: np.random.Generator | None = None) -> Image.Image:
    """Random size x size crop from the original pixels (pads by reflection if too small)."""
    rng = rng if rng is not None else np.random.default_rng()
    im = _pad_to_min_size(im, size)
    w, h = im.size
    left = int(rng.integers(0, w - size + 1))
    top = int(rng.integers(0, h - size + 1))
    return im.crop((left, top, left + size, top + size))


class RandomCrop224:
    """Callable transform: PIL -> PIL 224x224 random crop, no resizing. Drop-in for torchvision."""

    def __init__(self, size: int = 224, seed: int | None = None):
        self.size = size
        self.rng = np.random.default_rng(seed)

    def __call__(self, im: Image.Image) -> Image.Image:
        return random_crop(im, self.size, self.rng)


def jpeg_reencode(
    src: Path | str,
    dst: Path | str,
    quality: int = 95,
    subsampling: int = 0,
    strip_metadata: bool = True,
) -> Path:
    """Open `src`, convert to RGB, save as baseline JPEG at a fixed quality.

    subsampling=0 means 4:4:4 (no chroma subsampling) so every file gets the same treatment.
    `dst` is given a .jpg suffix. Returns the written path.
    """
    src, dst = Path(src), Path(dst).with_suffix(".jpg")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        kwargs = dict(format="JPEG", quality=int(quality), subsampling=subsampling, optimize=False)
        if not strip_metadata and "exif" in im.info:
            kwargs["exif"] = im.info["exif"]
        im.save(dst, **kwargs)
    return dst


def reencode_folder(
    src_dir: Path | str,
    dst_dir: Path | str,
    quality: int = 95,
    skip_existing: bool = True,
    paths: Iterable[Path] | None = None,
) -> tuple[int, int]:
    """Re-encode every image under src_dir into dst_dir, mirroring the folder tree.

    Returns (written, skipped).
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    written = skipped = 0
    for p in paths if paths is not None else iter_images(src_dir):
        rel = Path(p).relative_to(src_dir)
        out = (dst_dir / rel).with_suffix(".jpg")
        if skip_existing and out.exists():
            skipped += 1
            continue
        jpeg_reencode(p, out, quality=quality)
        written += 1
    return written, skipped
