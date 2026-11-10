"""Select a shortcut-free, generator-split training/validation pool from the CF-Small index.

No image bytes are needed: this works on the metadata index (scripts/index_hf_parquet.py).
The shard builder (scripts/build_shards.py) later fetches exactly these rows.

Fixes for the shortcuts measured in docs/audit_raw_cf_small.md:
  1. scale matching   - equal real/fake counts per megapixel bucket (crop zoom carries no label info)
  2. JPEG equalising  - in each bucket, the same share of fakes as reals gets one JPEG pass
                        (quality drawn later from measured real JPEG qualities) -> `pre_jpeg`
  3. grayscale and tiny (<= 0.1 MP) images dropped
  4. nested subsets   - size_rank lets 10k subset of 50k subset of 200k (scaling study adds data only)
  5. split by generator (fakes); reals split 90/10 at random (CF-Small has no real_source for reals)

Usage:
    python scripts/build_pool.py --val-frac-systematic 0.1 --seed 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataprep.paths import INDEX  # noqa: E402

MP_BINS = [0.1, 0.8, 100.0]                  # buckets: M = (0.1, 0.8], L = (0.8, inf)
MP_LABELS = ["M", "L"]
VAL_MANUAL = ["DeepFloyd", "DiT", "StyleGAN3", "CIPS"]   # 4 families held out for validation

ARCH = {"LatDiff": "ldm", "PixDiff": "pixel-diffusion", "GAN": "gan", "Other": "other", "Real": "real"}


def canonical(name: str) -> str:
    return str(name).strip().lower().replace(" ", "-").replace("_", "-")


def load_index(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.assign(width=df.resolution.str[0].astype(int), height=df.resolution.str[1].astype(int))
    df["mp"] = df.width * df.height / 1e6
    df = df[(df["mode"] == "RGB") & (df.mp > MP_BINS[0])].copy()
    df["mp_bucket"] = pd.cut(df.mp, MP_BINS, labels=MP_LABELS).astype(str)
    df["format"] = df["format"].str.lower().replace({"jpg": "jpeg"})
    df["generator"] = np.where(df.label == 1, df.model_name.map(canonical), "real")
    df["architecture"] = df.architecture.map(ARCH).fillna("other")
    df["source_dataset"] = "cf_small"
    df["sample_id"] = "cf_small/" + df.hf_file.str.replace("data/", "", regex=False) + "#" + df.hf_row.astype(str)
    return df


def assign_splits(df: pd.DataFrame, val_frac: float, rng: np.random.Generator) -> pd.Series:
    split = pd.Series("train", index=df.index)
    fakes = df[df.label == 1]
    sys_gens = np.array(sorted(fakes[fakes.subset == "Systematic"].generator.unique()))
    val_sys = set(rng.choice(sys_gens, size=int(round(val_frac * len(sys_gens))), replace=False))
    val_gens = val_sys | {canonical(g) for g in VAL_MANUAL}
    split[(df.label == 1) & df.generator.isin(val_gens)] = "val"
    reals = df.index[df.label == 0]
    split[rng.choice(reals, size=int(round(0.1 * len(reals))), replace=False)] = "val"
    return split


def match(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Per split and bucket: equal real/fake counts; fakes spread evenly over generators."""
    parts = []
    for (_, _), g in df.groupby(["split", "mp_bucket"]):
        reals, fakes = g[g.label == 0], g[g.label == 1]
        n = min(len(reals), len(fakes))
        if n == 0:
            continue
        reals = reals.sample(n, random_state=int(rng.integers(1 << 31)))
        # round-robin over generators so one huge generator cannot dominate
        fakes = fakes.sample(frac=1, random_state=int(rng.integers(1 << 31)))
        fakes = fakes.assign(_r=fakes.groupby("generator").cumcount()).sort_values("_r").head(n).drop(columns="_r")
        # JPEG equalisation: same JPEG share among fakes as among the matched reals
        p_jpeg = float((reals.format == "jpeg").mean())
        fakes = fakes.assign(pre_jpeg=rng.random(len(fakes)) < p_jpeg)
        reals = reals.assign(pre_jpeg=False)
        parts += [reals, fakes]
    # nested subsets: a random rank per label, per split; take rank < N/2 per class for size N
    out = pd.concat(parts).sample(frac=1, random_state=int(rng.integers(1 << 31)))
    out["size_rank"] = out.groupby(["split", "label"]).cumcount()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(INDEX / "cf_small_index.parquet"))
    ap.add_argument("--val-frac-systematic", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(INDEX / "cf_small_pool.parquet"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = load_index(Path(args.index))
    df["split"] = assign_splits(df, args.val_frac_systematic, rng)
    pool = match(df, rng)
    keep = ["sample_id", "hf_file", "hf_row", "image_name", "label", "generator", "architecture", "subset",
            "source_dataset", "split", "format", "width", "height", "mp", "mp_bucket", "pre_jpeg", "size_rank", "nsfw_flag"]
    pool[keep].to_parquet(args.out, index=False)

    print(f"wrote {args.out}: {len(pool)} rows")
    print(pool.groupby(["split", "mp_bucket", "label"]).size().unstack("label"))
    f = pool[pool.label == 1]
    print("\nfake generators per split:", f.groupby("split").generator.nunique().to_dict())
    print("JPEG share  real:", round(float((pool[pool.label == 0].format == 'jpeg').mean()), 3),
          " fake(pre_jpeg):", round(float(f.pre_jpeg.mean()), 3))
    tr = set(f[f.split == "train"].generator); va = set(f[f.split == "val"].generator)
    print("train/val generator overlap:", len(tr & va))


if __name__ == "__main__":
    main()
