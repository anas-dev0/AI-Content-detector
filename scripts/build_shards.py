"""Build WebDataset shards of lossless 256x256 crops for the rows selected by build_pool.py.

One pass over the source Parquet files (each CF file is a single row group, so reading any
image downloads the whole image column of that file anyway). Each file is downloaded,
its selected rows processed, then deleted: disk use ~ one file + the output.

Per image:
  1. metadata of the ORIGINAL bytes (format, size, JPEG quality, EXIF, hashes)
  2. fakes flagged pre_jpeg get one JPEG pass, quality drawn from the measured qualities of
     real JPEGs in the pool (real files are processed first to build that distribution)
  3. one 8-aligned random 256x256 crop (seeded by sample_id), stored as lossless PNG
Output: SHARDS/cf_small/cf_small-<split>-<n>.tar (+ .json per sample) and a manifest part.

Usage:
    python scripts/build_shards.py --train-per-class 100000 --val-per-class 10000
    python scripts/build_shards.py --train-per-class 500 --val-per-class 100 --max-files 4   # smoke test
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataprep.crops import aligned_crop_boxes, crop_to_png  # noqa: E402
from dataprep.metadata import extract_meta  # noqa: E402
from dataprep.paths import DATA_ROOT, INDEX, SHARDS  # noqa: E402
from eval.degrade import image_rng  # noqa: E402

REPO = "OwensLab/CommunityForensics-Small"
CROP = 256
SHARD_BYTES = 1_000_000_000
LICENSE = "CC-BY-NC-SA-4.0"


class ShardWriter:
    def __init__(self, outdir: Path, prefix: str):
        self.outdir, self.prefix, self.idx, self.tar, self.size = outdir, prefix, -1, None, 0
        outdir.mkdir(parents=True, exist_ok=True)

    def _roll(self):
        if self.tar:
            self.tar.close()
        self.idx += 1
        self.name = f"{self.prefix}-{self.idx:05d}.tar"
        self.tar, self.size = tarfile.open(self.outdir / self.name, "w"), 0

    def write(self, key: str, png: bytes, meta: dict) -> str:
        if self.tar is None or self.size > SHARD_BYTES:
            self._roll()
        for ext, data in (("png", png), ("json", json.dumps(meta).encode())):
            info = tarfile.TarInfo(f"{key}.{ext}")
            info.size, info.mtime = len(data), 0
            self.tar.addfile(info, io.BytesIO(data))
            self.size += len(data) + 1024
        return self.name

    def close(self):
        if self.tar:
            self.tar.close()


def process(args: tuple) -> dict | None:
    """CPU work for one image (runs in a worker process)."""
    row, data, pre_q = args
    meta = extract_meta(data)
    src = data
    if row["pre_jpeg"]:
        buf = io.BytesIO()
        Image.open(io.BytesIO(data)).convert("RGB").save(buf, format="JPEG", quality=int(pre_q), subsampling=2)
        src = buf.getvalue()
    with Image.open(io.BytesIO(src)) as im:
        boxes = aligned_crop_boxes(im.width, im.height, CROP, 1, image_rng(0, row["sample_id"]))
        if not boxes:
            return None
        png = crop_to_png(im, boxes[0])
    import hashlib
    return {
        "row": row, "png": png,
        "meta": {**meta.to_dict(), "orig_hash": meta.hash, "hash": hashlib.sha256(png).hexdigest(),
                 "pre_jpeg_q": float(pre_q) if row["pre_jpeg"] else None,
                 "crop_xy": ",".join(map(str, (boxes[0][0], boxes[0][1], CROP, CROP)))},
    }


def download(fname: str, tmp: Path) -> Path:
    return Path(hf_hub_download(REPO, fname, repo_type="dataset", local_dir=tmp))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(INDEX / "cf_small_pool.parquet"))
    ap.add_argument("--train-per-class", type=int, required=True)
    ap.add_argument("--val-per-class", type=int, required=True)
    ap.add_argument("--max-files", type=int, default=None, help="smoke test: only the first N files (reals first)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--keep-files", action="store_true")
    args = ap.parse_args()

    pool = pd.read_parquet(args.pool)
    cap = pool.split.map({"train": args.train_per_class, "val": args.val_per_class})
    sel = pool[pool.size_rank < cap].copy()
    # reals first so the JPEG-quality distribution exists before fakes need it
    order = sel.groupby("hf_file").label.mean().sort_values().index.tolist()
    if args.max_files:                        # smoke test: half real-heavy, half fake-heavy files
        k = args.max_files
        order = order[: (k + 1) // 2] + order[len(order) - k // 2:]
    print(f"selected {len(sel)} rows in {sel.hf_file.nunique()} files; processing {len(order)} files", flush=True)

    tmp = DATA_ROOT / "tmp_download"
    outdir = SHARDS / "cf_small"
    writers = {s: ShardWriter(outdir, f"cf_small-{s}") for s in ("train", "val")}
    real_q: list[float] = []
    rng = random.Random(0)
    manifest, t0, n_done, n_bytes = [], time.time(), 0, 0

    with ProcessPoolExecutor(args.workers) as cpu, ThreadPoolExecutor(1) as net:
        nxt = net.submit(download, order[0], tmp)
        for i, fname in enumerate(order):
            path = nxt.result()
            if i + 1 < len(order):
                nxt = net.submit(download, order[i + 1], tmp)          # prefetch the next file
            rows = sel[sel.hf_file == fname]
            col = pq.read_table(path, columns=["image_data"]).column("image_data")
            jobs = []
            for r in rows.to_dict("records"):
                q = None
                if r["pre_jpeg"]:
                    if not real_q:
                        raise RuntimeError("no real JPEG qualities measured yet; process real files first")
                    q = rng.choice(real_q)
                jobs.append((r, col[int(r["hf_row"])].as_py(), q))
            for res in cpu.map(process, jobs, chunksize=32):
                if res is None:
                    continue
                r, m = res["row"], res["meta"]
                if r["label"] == 0 and m["jpeg_q"] is not None:
                    real_q.append(m["jpeg_q"])
                key = r["sample_id"].replace("/", "__").replace("#", "_").replace(".", "-")
                rec = {
                    "path": None, "label": int(r["label"]), "generator": r["generator"],
                    "architecture": r["architecture"], "source_dataset": "cf_small", "split": r["split"],
                    "format": m["format"], "width": m["width"], "height": m["height"], "jpeg_q": m["jpeg_q"],
                    "hash": m["hash"], "orig_hash": m["orig_hash"], "sample_id": r["sample_id"],
                    "parent_id": r["sample_id"], "phash": m["phash"], "exif_present": m["exif_present"],
                    "bytes_per_pixel": m["bytes_per_pixel"], "file_bytes": m["file_bytes"], "mode": m["mode"],
                    "pre_jpeg_q": m["pre_jpeg_q"], "crop_xy": m["crop_xy"], "mp_bucket": r["mp_bucket"],
                    "size_rank": int(r["size_rank"]), "subset": r["subset"], "nsfw_flag": bool(r["nsfw_flag"]),
                    "degradation": "clean", "license": LICENSE, "real_source": None,
                }
                shard = writers[r["split"]].write(key, res["png"], rec)
                rec["path"], rec["shard"] = f"shards/cf_small/{shard}#{key}", shard
                manifest.append(rec)
                n_done += 1
                n_bytes += len(res["png"])
            size_gb = path.stat().st_size / 1e9
            if not args.keep_files:
                path.unlink()
            el = time.time() - t0
            print(f"[{i + 1}/{len(order)}] {fname} ({size_gb:.2f} GB): {len(rows)} rows | total {n_done} crops, "
                  f"{n_bytes / max(n_done, 1) / 1e3:.0f} KB/crop, {el / 60:.1f} min elapsed", flush=True)

    for w in writers.values():
        w.close()
    out = INDEX / "cf_small_shards_manifest.parquet"
    pd.DataFrame(manifest).to_parquet(out, index=False)
    np.save(INDEX / "real_jpeg_q.npy", np.array(real_q))
    print(f"wrote {out} ({len(manifest)} rows); avg crop {n_bytes / max(n_done, 1) / 1e3:.0f} KB; "
          f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
