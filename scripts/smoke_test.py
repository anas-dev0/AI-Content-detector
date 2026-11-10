"""Verify the week-1 pipeline without any dataset.

Fabricates two kinds of images and checks that the tools behave as the theory predicts:

  "real"  : 1/f ("pink") noise with a soft gradient, i.e. a natural-image-like spectrum with
            fine texture at every scale, like sensor noise.
  "fake"  : a 128x128 pink-noise image up-sampled x2 the way a generator's transposed
            convolution does it (zero insertion + small kernel). This bakes a period-2 pattern
            into the pixels: the classic "checkerboard" fingerprint.
  "fake_nearest" : plain nearest-neighbour x2 up-sampling (a second fingerprint flavour).

Checks
  1. NPR residual of a nearest-up-sampled image is exactly 0, of a real image clearly > 0.
  2. SRM gives 3 (or 9) channels in [-1, 1] with no NaNs.
  3. random_crop / center_crop return 224x224 from a 300x200 image with no resizing.
  4. jpeg_reencode writes a real JPEG that reopens as RGB at the same size.
  5. The average spectrum of the fakes has a strong grid peak that the reals lack.

Then it writes demo figures to figures/ so you can see the dots with your own eyes.

    python scripts/smoke_test.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from aidet import preprocess, residuals, spectrum, viz  # noqa: E402
import make_spectrum_figure  # noqa: E402  (same folder)

DEMO = ROOT / "demo_data"
FIG = ROOT / "figures"


# --------------------------------------------------------------------------- synthetic images
def pink_noise(rng, size, alpha=1.0):
    """2-D noise whose amplitude falls as 1/f^alpha. Returns float32 (size, size) in [0, 1]."""
    white = rng.standard_normal((size, size))
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    f = np.sqrt(fx**2 + fy**2)
    f[0, 0] = 1.0
    shaped = np.fft.ifft2(np.fft.fft2(white) / f**alpha).real
    shaped = (shaped - shaped.min()) / (shaped.max() - shaped.min() + 1e-9)
    return shaped.astype(np.float32)


def make_real(rng, size=256):
    base = pink_noise(rng, size)
    yy, xx = np.mgrid[0:size, 0:size] / size
    grad = 0.3 * (rng.uniform(-1, 1) * xx + rng.uniform(-1, 1) * yy)
    img = np.stack([base + grad + rng.uniform(-0.1, 0.1) for _ in range(3)], axis=-1)
    img += rng.normal(0, 0.01, img.shape)  # sensor-like noise, fine grain
    return (np.clip(img, 0, 1) * 255).round().astype(np.uint8)


def transposed_conv_upsample(x, kernel):
    """Zero-insertion x2 then correlate with `kernel` (what a stride-2 transposed conv does)."""
    h, w = x.shape
    z = np.zeros((2 * h, 2 * w), dtype=np.float32)
    z[::2, ::2] = x
    return residuals.conv2d(z, kernel)


def make_fake(rng, size=256, flavour="checkerboard"):
    low = make_real(rng, size // 2).astype(np.float32) / 255.0
    chans = []
    for c in range(3):
        if flavour == "nearest":
            up = np.repeat(np.repeat(low[..., c], 2, 0), 2, 1)
        else:  # 3x3 ones kernel with stride-2 zero insertion -> uneven overlap -> checkerboard
            up = transposed_conv_upsample(low[..., c], np.ones((3, 3), np.float32) / 2.25)
        chans.append(up)
    img = np.stack(chans, axis=-1)
    img = (img - img.min()) / (img.max() - img.min() + 1e-9)
    return (np.clip(img, 0, 1) * 255).round().astype(np.uint8)


def build_demo(n=20, seed=0):
    rng = np.random.default_rng(seed)
    if DEMO.exists():
        shutil.rmtree(DEMO)
    for name, fn in [
        ("real", lambda: make_real(rng)),
        ("fake", lambda: make_fake(rng, flavour="checkerboard")),
        ("fake_nearest", lambda: make_fake(rng, flavour="nearest")),
    ]:
        d = DEMO / name
        d.mkdir(parents=True)
        for i in range(n):
            Image.fromarray(fn()).save(d / f"{name}_{i:03d}.png")
    return n


# --------------------------------------------------------------------------- checks
def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    return bool(cond)


def main():
    n = build_demo()
    print(f"demo images written to {DEMO} ({n} per class)")
    ok = True
    rng = np.random.default_rng(1)

    real = Image.open(DEMO / "real" / "real_000.png").convert("RGB")
    fake_nn = Image.open(DEMO / "fake_nearest" / "fake_nearest_000.png").convert("RGB")
    fake_cb = Image.open(DEMO / "fake" / "fake_000.png").convert("RGB")

    print("1. NPR residual")
    r_real = np.abs(residuals.npr_residual(real)).mean()
    r_nn = np.abs(residuals.npr_residual(fake_nn)).mean()
    ok &= check(r_nn < 1e-6, f"nearest-upsampled image -> residual {r_nn:.2e} (expected 0)")
    ok &= check(r_real > 1e-3, f"real-like image -> residual {r_real:.2e} (expected > 0)")
    ok &= check(residuals.npr_residual(real, mode="bilinear").shape == (256, 256, 3), "bilinear mode works")

    print("2. SRM residual")
    s3 = residuals.srm_residual(real)
    s9 = residuals.srm_residual(real, per_channel=True)
    ok &= check(s3.shape == (256, 256, 3) and s9.shape == (256, 256, 9), f"shapes {s3.shape} / {s9.shape}")
    ok &= check(np.isfinite(s3).all() and s3.min() >= -1 and s3.max() <= 1, "values finite and in [-1, 1]")
    ok &= check(np.abs(s3).mean() > 0, "non-zero response")

    print("3. crops never resize")
    wide = Image.fromarray(rng.integers(0, 255, (200, 300, 3), dtype=np.uint8))
    c1, c2 = preprocess.random_crop(wide, 224, rng), preprocess.center_crop(wide, 224)
    ok &= check(c1.size == (224, 224) and c2.size == (224, 224), "300x200 input -> 224x224 (reflect-padded)")
    big = Image.fromarray(rng.integers(0, 255, (400, 500, 3), dtype=np.uint8))
    big_arr = np.asarray(big)
    cc = np.asarray(preprocess.center_crop(big, 224))
    ok &= check(np.array_equal(cc, big_arr[88:312, 138:362]), "centre crop is an exact pixel copy")

    print("4. JPEG re-encode")
    out = preprocess.jpeg_reencode(DEMO / "real" / "real_000.png", DEMO / "tmp" / "real_000.png", quality=95)
    with Image.open(out) as im:
        ok &= check(im.format == "JPEG" and im.mode == "RGB" and im.size == (256, 256), f"{out.name}: {im.format} {im.mode} {im.size}")
    w, s = preprocess.reencode_folder(DEMO / "real", DEMO / "tmp_folder", quality=95)
    w2, s2 = preprocess.reencode_folder(DEMO / "real", DEMO / "tmp_folder", quality=95)
    ok &= check(w == n and s == 0 and w2 == 0 and s2 == n, f"folder re-encode: wrote {w}, then skipped {s2} on rerun")
    shutil.rmtree(DEMO / "tmp"), shutil.rmtree(DEMO / "tmp_folder")

    print("5. average spectrum shows the grid")
    spec_real, _ = spectrum.average_spectrum(preprocess.iter_images(DEMO / "real"), 224, 2)
    spec_fake, _ = spectrum.average_spectrum(preprocess.iter_images(DEMO / "fake"), 224, 2)
    spec_nn, _ = spectrum.average_spectrum(preprocess.iter_images(DEMO / "fake_nearest"), 224, 2)
    g_real, g_fake, g_nn = (spectrum.grid_peak_score(s) for s in (spec_real, spec_fake, spec_nn))
    ok &= check(g_fake > g_real + 0.5, f"grid-peak score: real {g_real:.2f} vs checkerboard fake {g_fake:.2f}")
    # Nearest-neighbour x2 = zero insertion + 2-px box filter, whose response is 0 at Nyquist, so it
    # does NOT produce a spectral peak; its fingerprint is caught by NPR (check 1), not by the FFT.
    # This is the lesson of the project: different generators need different fingerprint views.
    ok &= check(g_nn < 1.0, f"grid-peak score: nearest fake {g_nn:.2f} stays low (no Nyquist peak, by theory)")
    hf = lambda s: spectrum.band_energy(s, 0.6, 1.0)  # noqa: E731
    print(f"        high-freq energy: real {hf(spec_real):.2f}, checkerboard {hf(spec_fake):.2f}, nearest {hf(spec_nn):.2f}")

    print("6. figures")
    for inp in ("gray", "npr"):
        make_spectrum_figure.main(
            ["--real", str(DEMO / "real"), "--fake", str(DEMO / "fake"), "--fake", str(DEMO / "fake_nearest"),
             "--input", inp, "--out", str(FIG / f"demo_spectrum_{inp}.png")]
        )
    crop = preprocess.center_crop(fake_cb, 224)
    panel = viz.panel(
        [np.asarray(crop),
         viz.residual_to_rgb(residuals.npr_residual(crop), gain=6),
         viz.residual_to_rgb(residuals.srm_residual(crop), gain=0.5),
         viz.spectrum_to_rgb(spectrum.log_fft_spectrum(crop))],
        ["fake crop", "NPR residual", "SRM residual", "log-FFT"],
    )
    viz.save_png(panel, FIG / "demo_single_image_views.png")
    print(f"  wrote {FIG / 'demo_single_image_views.png'}")

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
