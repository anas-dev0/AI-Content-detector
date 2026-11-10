"""Lossless crops for training shards.

Why crops and not resizing: resizing resamples every pixel and wipes out the high-frequency
traces (up-sampling grids, noise statistics) that detectors rely on, and it does so
differently for images of different sizes, which is itself a shortcut. Cropping keeps
pixels untouched.

Why 8-aligned offsets: JPEG works on 8x8 blocks. Cropping on the block grid keeps the
blocking pattern of an already-JPEG image aligned, so the crop is not "a different
JPEG" from the original.
"""
from __future__ import annotations

import io
import random

from PIL import Image

GRID = 8


def aligned_crop_boxes(width: int, height: int, size: int, n: int, rng: random.Random) -> list[tuple[int, int, int, int]]:
    """Up to n distinct random crop boxes (x, y, x+size, y+size) with 8-aligned x, y.

    Returns [] if the image is smaller than the crop (caller logs and skips it).
    """
    if width < size or height < size:
        return []
    nx, ny = (width - size) // GRID + 1, (height - size) // GRID + 1
    cells = nx * ny
    picks = rng.sample(range(cells), k=min(n, cells))
    boxes = []
    for p in picks:
        x, y = (p % nx) * GRID, (p // nx) * GRID
        boxes.append((x, y, x + size, y + size))
    return boxes


def crop_to_png(img: Image.Image, box: tuple[int, int, int, int]) -> bytes:
    """Crop and store losslessly as PNG (RGB). compress_level only trades speed for size."""
    buf = io.BytesIO()
    img.crop(box).convert("RGB").save(buf, format="PNG", compress_level=6)
    return buf.getvalue()
