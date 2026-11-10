"""Frequency-domain views of an image: log-magnitude FFT spectra, dataset averages, radial profiles.

Generators that up-sample step by step leave periodic artefacts that show up as bright dots or
grid lines in the spectrum (Durall et al. 2020, Frank et al. 2020, Corvi et al. 2023).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image

from .preprocess import center_crop, random_crop
from .residuals import to_float_array, to_gray


def log_fft_spectrum(img, remove_dc: bool = True) -> np.ndarray:
    """Grayscale -> 2-D FFT -> shift zero-frequency to the centre -> log(1 + |F|).

    `img` may be an RGB image, a (H, W) array, or a residual (H, W, C) array; multi-channel inputs
    are converted to luminance first. Removing the DC (mean) stops one huge centre spike from
    dominating the picture. Returns float32 (H, W).
    """
    arr = to_float_array(img)
    g = to_gray(arr) if arr.shape[2] in (1, 3) else arr.mean(axis=2)
    g = g.astype(np.float64)
    if remove_dc:
        g = g - g.mean()
    f = np.fft.fftshift(np.fft.fft2(g))
    return np.log1p(np.abs(f)).astype(np.float32)


def radial_profile(spec: np.ndarray, n_bins: int | None = None) -> np.ndarray:
    """Azimuthal average of a centred spectrum: mean value as a function of radius.

    Index 0 is the DC bin, the last index is the Nyquist corner. Used for the 1-D spectral plots
    of Durall et al. 2020. Returns float32 (n_bins,).
    """
    h, w = spec.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices(spec.shape)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    if n_bins is None:
        n_bins = int(r.max()) + 1
    bins = np.minimum((r / r.max() * (n_bins - 1)).round().astype(int), n_bins - 1)
    sums = np.bincount(bins.ravel(), weights=spec.ravel(), minlength=n_bins)
    counts = np.bincount(bins.ravel(), minlength=n_bins)
    return (sums / np.maximum(counts, 1)).astype(np.float32)


def band_energy(spec: np.ndarray, r_lo: float, r_hi: float) -> float:
    """Mean log-magnitude in the ring r_lo <= r/r_max < r_hi (fractions of the max radius)."""
    h, w = spec.shape
    yy, xx = np.indices(spec.shape)
    r = np.sqrt((yy - h // 2) ** 2 + (xx - w // 2) ** 2)
    r = r / r.max()
    mask = (r >= r_lo) & (r < r_hi)
    return float(spec[mask].mean())


def grid_peak_score(spec: np.ndarray, size: int = 5) -> float:
    """How much brighter the classic up-sampling spots are than their surroundings.

    Looks at the spectrum at the Nyquist edges/corner (factor-2 up-sampling) and at the half-
    Nyquist points on both axes (factor-4), takes the mean of a small window at each spot and
    subtracts the median of a ring at the same radius. Real photos score near 0; images with
    periodic up-sampling artefacts score clearly above 0. Units: log-magnitude.
    """
    h, w = spec.shape
    cy, cx = h // 2, w // 2
    spots = [
        (0, 0),  # corner (pi, pi) in the shifted spectrum
        (0, cx),  # (pi, 0)
        (cy, 0),  # (0, pi)
        (cy, cx // 2),
        (cy, cx + cx // 2),
        (cy // 2, cx),
        (cy + cy // 2, cx),
    ]
    yy, xx = np.indices(spec.shape)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    half = size // 2
    scores = []
    for y, x in spots:
        y0, y1 = max(0, y - half), min(h, y + half + 1)
        x0, x1 = max(0, x - half), min(w, x + half + 1)
        peak = float(spec[y0:y1, x0:x1].mean())
        rad = rr[y, x]
        ring = spec[(rr >= rad - 1.5) & (rr <= rad + 1.5)]
        scores.append(peak - float(np.median(ring)))
    return float(max(scores))


def iter_crops(
    paths: Iterable[Path | str],
    crop: int = 224,
    crops_per_image: int = 1,
    seed: int = 0,
    max_images: int | None = None,
):
    """Yield (path, PIL crop) pairs. The first crop is centred, the others are random; no resizing."""
    rng = np.random.default_rng(seed)
    for i, p in enumerate(paths):
        if max_images is not None and i >= max_images:
            break
        with Image.open(p) as im:
            im = im.convert("RGB")
            yield p, center_crop(im, crop)
            for _ in range(crops_per_image - 1):
                yield p, random_crop(im, crop, rng=rng)


def average_spectrum(
    paths: Iterable[Path | str],
    crop: int = 224,
    crops_per_image: int = 1,
    transform: Callable | None = None,
    seed: int = 0,
    max_images: int | None = None,
) -> tuple[np.ndarray, int]:
    """Mean log-FFT spectrum over many image crops.

    transform: optional function applied to each crop before the FFT (for example a residual
               such as residuals.npr_residual). Taking the spectrum of a residual rather than
               of the raw image makes the up-sampling peaks much easier to see.
    Returns (mean_spectrum float32 (crop, crop), number_of_crops_used).
    """
    acc = None
    n = 0
    for _, im in iter_crops(paths, crop, crops_per_image, seed, max_images):
        x = transform(im) if transform is not None else im
        s = log_fft_spectrum(x)
        acc = s.astype(np.float64) if acc is None else acc + s
        n += 1
    if acc is None:
        raise ValueError("no images found")
    return (acc / n).astype(np.float32), n
