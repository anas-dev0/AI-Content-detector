"""Per-image metadata used by the manifest and the shortcut audit.

Everything here describes the ORIGINAL file (before any crop/re-encode), because these are
exactly the cues a detector could cheat with: format, size, JPEG quality, EXIF, bytes/pixel.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import asdict, dataclass

import imagehash
import numpy as np
from PIL import Image

# Standard IJG (libjpeg) luminance quantisation table, quality 50 (Annex K of the JPEG spec).
_IJG_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
], dtype=np.float64)


def _ijg_table(quality: int) -> np.ndarray:
    # Integer arithmetic exactly as libjpeg's jpeg_quality_scaling / jpeg_add_quant_table.
    scale = 5000 // quality if quality < 50 else 200 - 2 * quality
    base = _IJG_LUMA.astype(np.int64)
    return np.clip((base * scale + 50) // 100, 1, 255).astype(np.float64)


_IJG_SORTED = {q: np.sort(_ijg_table(q)) for q in range(1, 101)}


def estimate_jpeg_quality(img: Image.Image) -> tuple[float | None, float | None]:
    """Estimate IJG-equivalent JPEG quality from the luminance quantisation table.

    Returns (quality, fit_error). Tables are compared after sorting, which makes the
    comparison independent of zigzag/natural ordering. A large fit_error means a
    non-IJG table (camera / Photoshop); the quality is then only an approximation.
    """
    q = getattr(img, "quantization", None)
    if not q:
        return None, None
    luma = np.sort(np.asarray(q[min(q)], dtype=np.float64))
    if luma.size != 64:
        return None, None
    errs = {k: float(np.abs(v - luma).mean()) for k, v in _IJG_SORTED.items()}
    best = min(errs, key=errs.get)
    return float(best), errs[best]


@dataclass
class ImageMeta:
    format: str            # jpeg | png | webp | other   (original container)
    width: int
    height: int
    mode: str
    jpeg_q: float | None
    jpeg_q_fit_err: float | None
    exif_present: bool
    file_bytes: int
    bytes_per_pixel: float
    hash: str              # sha256 of the original bytes
    phash: str             # 64-bit perceptual hash, hex

    def to_dict(self) -> dict:
        return asdict(self)


_FORMATS = {"JPEG": "jpeg", "MPO": "jpeg", "PNG": "png", "WEBP": "webp"}


def extract_meta(data: bytes) -> ImageMeta:
    """Metadata from raw file bytes. Does not modify pixels."""
    sha = hashlib.sha256(data).hexdigest()
    with Image.open(io.BytesIO(data)) as im:
        fmt = _FORMATS.get((im.format or "").upper(), "other")
        w, h = im.size
        mode = im.mode
        jq, jerr = estimate_jpeg_quality(im) if fmt == "jpeg" else (None, None)
        exif = bool(im.info.get("exif")) or bool(len(im.getexif()))
        ph = str(imagehash.phash(im.convert("RGB")))
    return ImageMeta(
        format=fmt, width=w, height=h, mode=mode,
        jpeg_q=jq, jpeg_q_fit_err=jerr, exif_present=exif,
        file_bytes=len(data), bytes_per_pixel=len(data) / float(w * h),
        hash=sha, phash=ph,
    )


def phash_distance(a: str, b: str) -> int:
    return int(bin(int(a, 16) ^ int(b, 16)).count("1"))
