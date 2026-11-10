"""Average log-FFT spectrum per folder, side by side, plus difference maps and radial profiles.

Example (real phone photos vs images from one AI tool):

    python scripts/make_spectrum_figure.py --real C:/data/demo/real --fake C:/data/demo/fake \
        --out figures/spectrum_raw.png
    python scripts/make_spectrum_figure.py --real ... --fake ... --input npr --out figures/spectrum_npr.png

--input gray : spectrum of the luminance image (what the study calls "frequency spectrum")
--input npr  : spectrum of the NPR residual (up-sampling peaks are much easier to see)
--input srm  : spectrum of the first SRM residual channel
--input blur : spectrum of (image - blurred image)

Writes: the PNG panel, a CSV with one radial profile per folder, and prints a grid-peak score
and high-frequency energy per folder. If the fake folders show bright dots at the edges /
corners / mid-axes that the real folder lacks, the pipeline works.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aidet import preprocess, residuals, spectrum, viz  # noqa: E402

TRANSFORMS = {
    "gray": None,
    "npr": lambda im: residuals.npr_residual(im),
    "srm": lambda im: residuals.srm_residual(im)[..., :1],
    "blur": lambda im: residuals.denoise_subtract_residual(im),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", required=True, type=Path, help="folder of real photos")
    ap.add_argument("--fake", required=True, type=Path, action="append", help="folder of AI images (repeatable)")
    ap.add_argument("--out", type=Path, default=Path("figures/spectrum.png"))
    ap.add_argument("--input", choices=sorted(TRANSFORMS), default="gray")
    ap.add_argument("--crop", type=int, default=224)
    ap.add_argument("--crops-per-image", type=int, default=2)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    folders = [("real", args.real)] + [(f"fake:{p.name}", p) for p in args.fake]
    transform = TRANSFORMS[args.input]

    specs, counts = {}, {}
    for name, folder in folders:
        paths = preprocess.iter_images(folder)
        if not paths:
            sys.exit(f"no images found in {folder}")
        specs[name], counts[name] = spectrum.average_spectrum(
            paths, args.crop, args.crops_per_image, transform, args.seed, args.max_images
        )

    # shared colour scale across all folders so they are comparable
    stack = np.stack(list(specs.values()))
    lo, hi = float(np.percentile(stack, 1)), float(np.percentile(stack, 99.5))

    images, labels = [], []
    for name, s in specs.items():
        images.append(viz.spectrum_to_rgb(s, lo, hi))
        labels.append(f"{name} (n={counts[name]})")
    real = specs["real"]
    diffs = {n: s - real for n, s in specs.items() if n != "real"}
    limit = max(float(np.percentile(np.abs(d), 99.5)) for d in diffs.values()) or 1e-6
    for n, d in diffs.items():
        images.append(viz.diff_to_rgb(d, limit))
        labels.append(f"{n} - real")

    fig = viz.panel(images, labels, scale=1 if args.crop >= 224 else 2)
    viz.save_png(fig, args.out)

    csv_path = args.out.with_suffix(".radial.csv")
    profiles = {n: spectrum.radial_profile(s) for n, s in specs.items()}
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["radius_bin"] + list(profiles))
        for i in range(len(next(iter(profiles.values())))):
            w.writerow([i] + [f"{profiles[n][i]:.5f}" for n in profiles])

    print(f"input={args.input} crop={args.crop}  -> {args.out}  and  {csv_path}")
    print(f"{'folder':<24}{'crops':>7}{'grid-peak':>12}{'HF energy':>12}{'LF energy':>12}")
    for n, s in specs.items():
        print(
            f"{n:<24}{counts[n]:>7}{spectrum.grid_peak_score(s):>12.3f}"
            f"{spectrum.band_energy(s, 0.6, 1.0):>12.3f}{spectrum.band_energy(s, 0.0, 0.2):>12.3f}"
        )
    print("grid-peak ~0 for real photos; clearly >0 means periodic up-sampling artefacts.")


if __name__ == "__main__":
    main()
