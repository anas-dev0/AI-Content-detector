import io
import random

import numpy as np
import pytest
from PIL import Image

from dataprep.crops import aligned_crop_boxes, crop_to_png
from dataprep.metadata import extract_meta, phash_distance
from eval.degrade import DEGRADATIONS, apply


def _photo_like(w=320, h=240, seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, w)[None, :, None] * np.linspace(0, 1, h)[:, None, None]
    arr = (255 * x * np.array([1.0, 0.6, 0.3]) + rng.normal(0, 12, (h, w, 3))).clip(0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def _encode(img, fmt, **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


@pytest.mark.parametrize("q", [30, 50, 75, 90, 95])
def test_jpeg_quality_recovered(q):
    m = extract_meta(_encode(_photo_like(), "JPEG", quality=q))
    assert m.format == "jpeg"
    assert m.jpeg_q == q
    assert m.jpeg_q_fit_err == 0.0


def test_png_metadata():
    data = _encode(_photo_like(), "PNG")
    m = extract_meta(data)
    assert (m.format, m.width, m.height, m.jpeg_q, m.exif_present) == ("png", 320, 240, None, False)
    assert m.bytes_per_pixel == pytest.approx(len(data) / (320 * 240))
    assert len(m.hash) == 64 and len(m.phash) == 16


def test_phash_near_duplicate():
    img = _photo_like()
    a = extract_meta(_encode(img, "PNG")).phash
    b = extract_meta(_encode(img, "JPEG", quality=70)).phash
    c = extract_meta(_encode(_photo_like(seed=5).transpose(Image.Transpose.ROTATE_180), "PNG")).phash
    assert phash_distance(a, b) <= 6 < phash_distance(a, c)


def test_crops_aligned_and_inside():
    rng = random.Random(0)
    for w, h in [(256, 256), (300, 513), (1024, 768)]:
        for x0, y0, x1, y1 in aligned_crop_boxes(w, h, 256, 4, rng):
            assert x0 % 8 == 0 and y0 % 8 == 0
            assert x1 - x0 == 256 and y1 - y0 == 256 and x1 <= w and y1 <= h
    assert aligned_crop_boxes(200, 400, 256, 2, rng) == []


def test_crop_is_lossless():
    img = _photo_like()
    box = (8, 16, 8 + 128, 16 + 128)
    out = Image.open(io.BytesIO(crop_to_png(img, box)))
    assert np.array_equal(np.asarray(out), np.asarray(img.crop(box)))


@pytest.mark.parametrize("name", list(DEGRADATIONS))
def test_degradations_deterministic(name):
    data = _encode(_photo_like(), "PNG")
    a, ext_a = apply(name, data, "img_001", seed=0)
    b, ext_b = apply(name, data, "img_001", seed=0)
    assert a == b and ext_a == ext_b
    Image.open(io.BytesIO(a)).verify()


def test_screenshot_varies_with_id():
    data = _encode(_photo_like(), "PNG")
    outs = {apply("screenshot", data, f"img_{i}", seed=0)[0] for i in range(8)}
    assert len(outs) > 1
