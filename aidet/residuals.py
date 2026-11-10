"""Noise residuals: remove the image content, keep the tiny noise where generator fingerprints live.

All functions accept a PIL image or a numpy array (uint8 0..255 or float 0..1), and return a
float32 array of shape (H, W, C). Residuals are signed and small; see each docstring for the scale.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def to_float_array(img) -> np.ndarray:
    """PIL image or array -> float32 array of shape (H, W, C) with values in [0, 1]."""
    if isinstance(img, Image.Image):
        img = np.asarray(img.convert("RGB"))
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr


def to_gray(arr: np.ndarray) -> np.ndarray:
    """(H, W, C) float -> (H, W) float luminance. A single channel passes through."""
    arr = to_float_array(arr)
    if arr.shape[2] == 1:
        return arr[..., 0]
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def conv2d(x: np.ndarray, kernel: np.ndarray, pad_mode: str = "reflect") -> np.ndarray:
    """2-D correlation of a (H, W) array with a small (kh, kw) kernel, 'same' output size.

    Pure numpy (sliding windows + einsum). Fast enough for 224..1024 px images and 5x5 kernels.
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    xp = np.pad(x, ((ph, kh - 1 - ph), (pw, kw - 1 - pw)), mode=pad_mode)
    windows = np.lib.stride_tricks.sliding_window_view(xp, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, kernel.astype(x.dtype))


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    radius = max(1, int(round(3 * sigma)))
    t = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum()


def gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur of an (H, W, C) float array."""
    arr = to_float_array(arr)
    k = _gaussian_kernel1d(sigma)
    out = np.empty_like(arr)
    for c in range(arr.shape[2]):
        tmp = conv2d(arr[..., c], k[None, :])
        out[..., c] = conv2d(tmp, k[:, None])
    return out


# --------------------------------------------------------------------------------------
# 1) NPR: Neighboring Pixel Relationships (Tan et al., CVPR 2024)
# --------------------------------------------------------------------------------------

_PIL_RESAMPLE = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
}


def _resize_float(arr: np.ndarray, size_wh: tuple[int, int], mode: str) -> np.ndarray:
    """Resize an (H, W, C) float array with PIL, channel by channel, in float32 precision."""
    w, h = size_wh
    out = np.empty((h, w, arr.shape[2]), dtype=np.float32)
    for c in range(arr.shape[2]):
        im = Image.fromarray(arr[..., c].astype(np.float32), mode="F")
        out[..., c] = np.asarray(im.resize((w, h), _PIL_RESAMPLE[mode]), dtype=np.float32)
    return out


def npr_residual(img, factor: int = 2, mode: str = "nearest") -> np.ndarray:
    """NPR residual: x - up(down(x)).

    Generators build images with up-sampling layers, so their pixels are strongly tied to their
    neighbours and this operation almost reconstructs them (small residual). Real photos keep
    sensor noise and fine texture (larger residual). The paper uses nearest-neighbour resampling.

    Returns a float32 (H, W, C) array, roughly in [-1, 1]. H and W are cropped to a multiple of
    `factor` so down/up-sampling is exact.
    """
    arr = to_float_array(img)
    h, w, c = arr.shape
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    arr = arr[:h2, :w2]

    if mode == "nearest":
        down = arr[::factor, ::factor]
        up = np.repeat(np.repeat(down, factor, axis=0), factor, axis=1)
    else:
        down = _resize_float(arr, (w2 // factor, h2 // factor), mode)
        up = _resize_float(down, (w2, h2), mode)
    return (arr - up).astype(np.float32)


# --------------------------------------------------------------------------------------
# 2) SRM: three fixed high-pass filters from steganalysis (Fridrich & Kodovsky 2012),
#    as used for forensics by Zhou et al., CVPR 2018.
# --------------------------------------------------------------------------------------

SRM_KERNELS = np.stack(
    [
        # 2nd-order 3x3 "square" (embedded in 5x5), normalised by 4
        np.array(
            [
                [0, 0, 0, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        / 4.0,
        # 3rd-order 5x5 "square", normalised by 12
        np.array(
            [
                [-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1],
            ],
            dtype=np.float32,
        )
        / 12.0,
        # 1st-order horizontal 2nd derivative, normalised by 2
        np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -2, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        / 2.0,
    ]
)


def srm_residual(img, per_channel: bool = False, truncate: float | None = 2.0) -> np.ndarray:
    """Apply the 3 fixed SRM kernels.

    per_channel=False : kernels run on the luminance image -> (H, W, 3)
    per_channel=True  : kernels run on each RGB channel      -> (H, W, 9)

    Filtering is done on the 0..255 scale, then clipped to [-truncate, truncate] (the classic
    steganalysis truncation, T=2) and divided by `truncate`, so the output lies in [-1, 1].
    Pass truncate=None to keep raw, unclipped values (still divided by 255).
    """
    arr = to_float_array(img) * 255.0
    planes = [arr[..., c] for c in range(arr.shape[2])] if per_channel else [to_gray(arr / 255.0) * 255.0]
    outs = []
    for plane in planes:
        for k in SRM_KERNELS:
            outs.append(conv2d(plane, k))
    out = np.stack(outs, axis=-1)
    if truncate is None:
        return (out / 255.0).astype(np.float32)
    return (np.clip(out, -truncate, truncate) / truncate).astype(np.float32)


# --------------------------------------------------------------------------------------
# 3) Simplest residual: image minus a blurred copy of itself
# --------------------------------------------------------------------------------------


def denoise_subtract_residual(img, sigma: float = 1.0) -> np.ndarray:
    """x - gaussian_blur(x). Cheap high-pass residual, returns float32 (H, W, C) around 0."""
    arr = to_float_array(img)
    return (arr - gaussian_blur(arr, sigma)).astype(np.float32)


RESIDUALS = {
    "npr": npr_residual,
    "srm": srm_residual,
    "blur": denoise_subtract_residual,
}


def residual(img, kind: str = "npr", **kwargs) -> np.ndarray:
    """Dispatch by name: 'npr', 'srm' or 'blur'."""
    try:
        fn = RESIDUALS[kind]
    except KeyError as exc:
        raise ValueError(f"unknown residual {kind!r}; choose from {sorted(RESIDUALS)}") from exc
    return fn(img, **kwargs)
