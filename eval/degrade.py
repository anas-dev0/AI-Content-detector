"""Offline, seeded degradations (docs/evaluation_protocol.md, section 3).

Each function maps original bytes -> (degraded bytes, file extension). Randomness comes
only from a per-image RNG seeded by hash(seed, sample_id), so results do not depend on
processing order, worker count or machine. Applied identically to real and fake.
Pillow's version is pinned in requirements.txt because codec output can change across versions.
"""
from __future__ import annotations

import hashlib
import io
import random
from typing import Callable

from PIL import Image


def image_rng(seed: int, sample_id: str) -> random.Random:
    h = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def _open_rgb(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as im:
        return im.convert("RGB")


def _jpeg(img: Image.Image, q: int) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q, subsampling=2)   # 2 = 4:2:0 chroma subsampling
    return buf.getvalue()


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def _resize(img: Image.Image, factor: float) -> Image.Image:
    w, h = img.size
    return img.resize((max(1, round(w * factor)), max(1, round(h * factor))), Image.Resampling.BICUBIC, reducing_gap=None)


def jpeg(q: int) -> Callable[[bytes, random.Random], tuple[bytes, str]]:
    def f(data: bytes, rng: random.Random) -> tuple[bytes, str]:
        return _jpeg(_open_rgb(data), q), "jpg"
    return f


def down50(data: bytes, rng: random.Random) -> tuple[bytes, str]:
    # Pillow's BICUBIC downscale is antialiased (filter support scales with the factor).
    return _png(_resize(_open_rgb(data), 0.5)), "png"


def screenshot(data: bytes, rng: random.Random) -> tuple[bytes, str]:
    """Approximate 'screenshot then share': display scaling -> PNG grab -> crop UI margins -> JPEG 85."""
    img = _open_rgb(data)
    scale = rng.choice([0.75, 1.0, 1.25])
    if scale != 1.0:
        img = _resize(img, scale)
    img = _open_rgb(_png(img))                       # the screenshot itself is lossless
    w, h = img.size
    l, r, t, b = (int(rng.uniform(0, 0.05) * d) for d in (w, w, h, h))
    img = img.crop((l, t, max(l + 1, w - r), max(t + 1, h - b)))
    return _jpeg(img, 85), "jpg"


def chain_social(data: bytes, rng: random.Random) -> tuple[bytes, str]:
    img = _resize(_open_rgb(data), 0.5)
    once = _open_rgb(_jpeg(img, 75))
    return _jpeg(once, 75), "jpg"


DEGRADATIONS: dict[str, Callable[[bytes, random.Random], tuple[bytes, str]]] = {
    "jpeg95": jpeg(95),
    "jpeg75": jpeg(75),
    "jpeg50": jpeg(50),
    "down50": down50,
    "screenshot": screenshot,
    "chain_social": chain_social,
}


def apply(name: str, data: bytes, sample_id: str, seed: int = 0) -> tuple[bytes, str]:
    return DEGRADATIONS[name](data, image_rng(seed, sample_id))
