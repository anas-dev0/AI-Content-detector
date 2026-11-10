"""Shortcut audit (PLAN.md week 1, decision D1).

Question: can the label be predicted WITHOUT looking at image content? If a classifier on
file metadata alone (format, size, JPEG quality, EXIF, bytes/pixel, colour mode) reaches
AUROC clearly above 0.5, a detector can cheat the same way, and the data must be fixed
before training. Gate: AUROC <= 0.6 including the upper CI bound.

The split is grouped by generator, so the audit classifier cannot just memorise models.
A gradient-boosted tree is used even for the size-only baseline: a stronger adversary
than logistic regression makes the audit conservative.

Usage:
    python scripts/shortcut_audit.py --input C:/aidet_data/index/cf_small_index.parquet --name raw_cf_small
    python scripts/shortcut_audit.py --input data/manifests/manifest_vXXXX.parquet --split train --name v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataprep.paths import DOCS  # noqa: E402
from eval.metrics import auroc, bootstrap_ci  # noqa: E402

GATE = 0.6


def features(df: pd.DataFrame) -> pd.DataFrame:
    if "width" not in df and "resolution" in df:            # raw HF index
        df = df.assign(width=df.resolution.str[0], height=df.resolution.str[1])
    fmt = df["format"].astype(str).str.lower()
    X = pd.DataFrame({
        "is_jpeg": (fmt == "jpeg").astype(int),
        "width": df.width, "height": df.height,
        "log_mp": np.log(df.width * df.height),
        "aspect": df.width / df.height,
        "is_square": (df.width == df.height).astype(int),
        "w_mod64": df.width % 64, "h_mod64": df.height % 64,
    }, index=df.index)
    if "mode" in df:
        X["mode_not_rgb"] = (df["mode"] != "RGB").astype(int)
    # Cues that survive into a 256x256 crop: JPEG history (block traces in pixels) and coarse scale.
    X["jpeg_history"] = X.is_jpeg | (df["pre_jpeg"].astype(int) if "pre_jpeg" in df else 0)
    X["mp_bucket"] = pd.cut(df.width * df.height / 1e6, [0, 0.1, 0.8, 1e9], labels=False)
    for col in ("jpeg_q", "bytes_per_pixel", "exif_present"):
        if col in df:
            X[col] = df[col].astype(float)
    return X


FEATURE_SETS = {
    # GATED: what the detector can actually observe in a 256x256 crop (+ JPEG quality / EXIF
    # / bytes-per-pixel once a manifest with real bytes exists).
    "model_visible": ["jpeg_history", "mp_bucket", "mode_not_rgb", "jpeg_q"],
    # INFORMATIONAL: properties of the original file. Exact size/aspect do not survive a square
    # crop, but they show which cues exist in the sources.
    "metadata_all": None,
    "size_only": ["width", "height", "log_mp", "aspect", "is_square", "w_mod64", "h_mod64"],
    "format_only": ["is_jpeg"],
}
GATED = {"model_visible"}


def run(df: pd.DataFrame, seed: int = 0, n_boot: int = 300) -> list[dict]:
    gen = df["generator"] if "generator" in df else df["model_name"]
    groups = np.where(df.label == 1, gen.astype(str), "real_" + df.index.astype(str))
    tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(df, df.label, groups))
    X_all, y = features(df), df.label.to_numpy()
    out = []
    for name, cols in FEATURE_SETS.items():
        X = X_all if cols is None else X_all[[c for c in cols if c in X_all]]
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(X.iloc[tr], y[tr])
        s = clf.predict_proba(X.iloc[te])[:, 1]
        ci = bootstrap_ci(y[te], s, auroc, n_boot=n_boot, seed=seed)
        out.append({"baseline": name, "features": list(X.columns), "auroc": ci.value, "lo": ci.lo, "hi": ci.hi,
                    "n_test": len(te), "gated": name in GATED, "pass": ci.hi <= GATE})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default=None, help="restrict to one split of a manifest")
    ap.add_argument("--max-rows", type=int, default=200_000)
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    if args.split:
        df = df[df.split == args.split]
    if len(df) > args.max_rows:
        k = args.max_rows // 2
        df = pd.concat([g.sample(min(len(g), k), random_state=0) for _, g in df.groupby("label")])
    df = df.reset_index(drop=True)

    res = run(df)
    lines = [f"# Shortcut audit: {args.name}", "",
             f"Input: `{Path(args.input).name}`, rows used: {len(df)} (real {int((df.label == 0).sum())}, fake {int((df.label == 1).sum())}).",
             f"Split grouped by generator, 30% held out. Gate: AUROC upper 95% CI <= {GATE}.", "",
             "| Baseline | Role | AUROC [95% CI] | <= 0.6 | Features |", "|---|---|---|---|---|"]
    for r in res:
        role = "**gate**" if r["gated"] else "info"
        lines.append(f"| {r['baseline']} | {role} | {r['auroc']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] | "
                     f"{'yes' if r['pass'] else 'NO'} | {', '.join(r['features'])} |")
    verdict = "PASS" if all(r["pass"] for r in res if r["gated"]) else "FAIL: fix the data before training"
    lines += ["", "Gate = cues observable in a 256x256 crop (JPEG history, coarse scale, colour mode, JPEG quality). "
              "Informational rows describe the original files; exact size/aspect do not survive cropping. "
              "Pixel-level baselines (colour histogram, 64-px CLIP content probe) run on the built shards."]
    lines += ["", f"**Verdict: {verdict}**", ""]
    out = DOCS / f"audit_{args.name}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
