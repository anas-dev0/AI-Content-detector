"""Turn spectra and residuals into PNG panels using only numpy + Pillow (no matplotlib needed)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Small perceptual colour maps defined by anchor points (r, g, b in 0..1), linearly interpolated.
_CMAPS = {
    "magma": [
        (0.001, 0.000, 0.014),
        (0.251, 0.064, 0.418),
        (0.577, 0.148, 0.504),
        (0.866, 0.318, 0.391),
        (0.994, 0.624, 0.427),
        (0.987, 0.991, 0.750),
    ],
    "viridis": [
        (0.267, 0.005, 0.329),
        (0.253, 0.265, 0.530),
        (0.164, 0.471, 0.558),
        (0.134, 0.658, 0.517),
        (0.478, 0.821, 0.318),
        (0.993, 0.906, 0.144),
    ],
    "gray": [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
    # diverging: blue (negative) - white (zero) - red (positive)
    "bwr": [(0.13, 0.30, 0.75), (1.0, 1.0, 1.0), (0.80, 0.15, 0.15)],
}


def _lut(name: str) -> np.ndarray:
    anchors = np.array(_CMAPS[name], dtype=np.float32)
    xs = np.linspace(0, 1, len(anchors))
    t = np.linspace(0, 1, 256)
    lut = np.stack([np.interp(t, xs, anchors[:, c]) for c in range(3)], axis=1)
    return (lut * 255).round().astype(np.uint8)


def normalize(arr: np.ndarray, lo: float | None = None, hi: float | None = None, percentiles=(1.0, 99.5)) -> np.ndarray:
    """Scale an array to [0, 1] using fixed bounds or robust percentiles."""
    arr = np.asarray(arr, dtype=np.float32)
    if lo is None:
        lo = float(np.percentile(arr, percentiles[0]))
    if hi is None:
        hi = float(np.percentile(arr, percentiles[1]))
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def colorize(arr01: np.ndarray, cmap: str = "magma") -> np.ndarray:
    """(H, W) array in [0, 1] -> (H, W, 3) uint8 via the named colour map."""
    idx = (np.clip(arr01, 0, 1) * 255).round().astype(np.uint8)
    return _lut(cmap)[idx]


def spectrum_to_rgb(spec: np.ndarray, lo=None, hi=None, cmap: str = "magma") -> np.ndarray:
    return colorize(normalize(spec, lo, hi), cmap)


def diff_to_rgb(diff: np.ndarray, limit: float | None = None) -> np.ndarray:
    """Signed difference map -> blue/white/red image, symmetric around 0."""
    if limit is None:
        limit = float(np.percentile(np.abs(diff), 99.5)) or 1e-6
    return colorize(normalize(diff, -limit, limit), "bwr")


def residual_to_rgb(res: np.ndarray, gain: float = 8.0) -> np.ndarray:
    """Signed residual (H, W, C) around 0 -> viewable uint8 image centred on mid-grey."""
    res = np.asarray(res, dtype=np.float32)
    if res.ndim == 3 and res.shape[2] not in (1, 3):
        res = res[..., :3]
    out = np.clip(0.5 + gain * res, 0, 1)
    if out.ndim == 3 and out.shape[2] == 1:
        out = np.repeat(out, 3, axis=2)
    if out.ndim == 2:
        out = np.repeat(out[..., None], 3, axis=2)
    return (out * 255).round().astype(np.uint8)


def _font(size: int = 14):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def panel(
    images: Sequence[np.ndarray | Image.Image],
    labels: Sequence[str] | None = None,
    scale: int = 1,
    gap: int = 8,
    label_height: int = 22,
    bg=(255, 255, 255),
) -> Image.Image:
    """Lay out images side by side with a text label above each. Images may differ in size."""
    pil = [im if isinstance(im, Image.Image) else Image.fromarray(np.asarray(im)) for im in images]
    if scale != 1:
        pil = [im.resize((im.width * scale, im.height * scale), Image.NEAREST) for im in pil]
    labels = list(labels) if labels is not None else [""] * len(pil)
    top = label_height if any(labels) else 0
    width = sum(im.width for im in pil) + gap * (len(pil) + 1)
    height = max(im.height for im in pil) + top + 2 * gap
    canvas = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(canvas)
    font = _font(14)
    x = gap
    for im, text in zip(pil, labels):
        if text:
            draw.text((x, gap // 2), text, fill=(0, 0, 0), font=font)
        canvas.paste(im, (x, top + gap))
        x += im.width + gap
    return canvas


def save_png(img: np.ndarray | Image.Image, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    im = img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img))
    im.save(path, format="PNG")
    return path
