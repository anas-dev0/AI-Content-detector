"""Synthetic shards + manifest in the exact format of scripts/build_shards.py.

For tests and for smoke-running the training skeleton before real shards exist. NOT data:
toy "fakes" carry faint period-2 row stripes (an up-sampling-like artefact), so a working
pipeline must beat chance on them. Never report anything trained on this.

    python -m training.toy --out C:/aidet_data/toy --train-per-class 2000 --val-per-class 500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_shards import ShardWriter  # noqa: E402
from dataprep.crops import crop_to_png  # noqa: E402
from PIL import Image  # noqa: E402


def toy_image(rng: np.random.Generator, size: int, fake: bool) -> Image.Image:
    base = rng.uniform(40, 215, 3)
    grad = np.linspace(-1, 1, size)
    arr = base + 30 * grad[None, :, None] * rng.uniform(-1, 1, 3) + 30 * grad[:, None, None] * rng.uniform(-1, 1, 3)
    arr = arr + rng.normal(0, 8, (size, size, 3))
    if fake:
        # period-2 row stripes: phase survives h-flips and 8-aligned crops (a checkerboard's would not)
        rows = np.arange(size)[:, None, None]
        arr = arr + 8.0 * ((rows % 2) * 2 - 1)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def make_toy_dataset(out: Path, train_per_class: int, val_per_class: int, size: int = 256, seed: int = 0) -> Path:
    """Write toy shards to out/shards/toy/ and return the manifest path (out/toy_manifest.parquet)."""
    out = Path(out)
    rng = np.random.default_rng(seed)
    writers = {s: ShardWriter(out / "shards" / "toy", f"toy-{s}") for s in ("train", "val")}
    gens = {"train": ["toy-gan-a", "toy-gan-b"], "val": ["toy-gan-c"]}
    rows = []
    for split, n in (("train", train_per_class), ("val", val_per_class)):
        for label in (0, 1):
            for rank in range(n):
                gen = "real" if label == 0 else gens[split][rank % len(gens[split])]
                sid = f"toy/{split}_{label}_{rank:06d}"
                png = crop_to_png(toy_image(rng, size, bool(label)), (0, 0, size, size))
                key = sid.replace("/", "__")
                rec = {"label": label, "generator": gen, "architecture": "real" if label == 0 else "gan",
                       "source_dataset": "toy", "split": split, "format": "png", "width": size, "height": size,
                       "jpeg_q": None, "hash": f"{split}{label}{rank}", "sample_id": sid, "parent_id": sid,
                       "size_rank": rank, "degradation": "clean"}
                shard = writers[split].write(key, png, rec)
                rows.append({**rec, "path": f"shards/toy/{shard}#{key}", "shard": shard})
    for w in writers.values():
        w.close()
    manifest = out / "toy_manifest.parquet"
    pd.DataFrame(rows).to_parquet(manifest, index=False)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-per-class", type=int, default=2000)
    ap.add_argument("--val-per-class", type=int, default=500)
    ap.add_argument("--size", type=int, default=256)
    a = ap.parse_args()
    print(make_toy_dataset(Path(a.out), a.train_per_class, a.val_per_class, a.size))


if __name__ == "__main__":
    main()
