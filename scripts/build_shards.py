"""Build WebDataset shards of lossless 256x256 crops for the rows selected by build_pool.py.

One pass over the source Parquet files (each CF file is a single row group, so reading any
image downloads the whole file anyway). Each file is downloaded, its selected rows
processed, then deleted: disk use ~ two files (current + prefetched next) + the output.

Memory: images are STREAMED from each file in small batches (pyarrow buffered reads), and
only one batch at a time is handed to the worker processes, so RAM stays bounded
(roughly a few hundred MB + the workers), whatever the file size.

Per image:
  1. metadata of the ORIGINAL bytes (format, size, JPEG quality, EXIF, hashes)
  2. fakes flagged pre_jpeg get one JPEG pass, quality drawn from the measured qualities of
     real JPEGs in the pool (real files are processed first to build that distribution)
  3. one 8-aligned random 256x256 crop (seeded by sample_id), stored as lossless PNG
<<<<<<< HEAD
Output: SHARDS/cf_small/cf_small-<split>-<n>.tar (+ .json per sample) and
        SHARDS/cf_small/cf_small_shards_manifest.parquet
=======
Output: SHARDS/cf_small/cf_small-<split>-<file>-<n>.tar (+ .json per sample) and a manifest part.

Resumable: every source file gets its own shards and its own manifest part in
SHARDS/cf_small/_progress/. A file counts as done only once its part is written, so Ctrl+C or a
crash loses at most the file in progress; rerun the same command to continue. Results do not
depend on interruptions: JPEG qualities for pre_jpeg fakes are drawn with a per-sample RNG from
the qualities of the real files, which are all processed before any fake-heavy file.
>>>>>>> fdf6b3f4d0493d4148c0d4d847f43b43a5e147f6

Usage:
    python scripts/build_shards.py --train-per-class 25000 --val-per-class 5000
    python scripts/build_shards.py --train-per-class 25000 --val-per-class 5000 --max-files 2   # smoke test
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
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
from dataprep.paths import MANIFESTS, SHARDS, TMP  # noqa: E402
from eval.degrade import image_rng  # noqa: E402

REPO = "OwensLab/CommunityForensics-Small"
CROP = 256
SHARD_BYTES = 1_000_000_000
BATCH = 128
LICENSE = "CC-BY-NC-SA-4.0"


def rss_gb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except ImportError:
        return float("nan")


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
    return {
        "row": row, "png": png,
        "meta": {**meta.to_dict(), "orig_hash": meta.hash, "hash": hashlib.sha256(png).hexdigest(),
                 "pre_jpeg_q": float(pre_q) if row["pre_jpeg"] else None,
                 "crop_xy": ",".join(map(str, (boxes[0][0], boxes[0][1], CROP, CROP)))},
    }


def download(fname: str) -> Path:
    return Path(hf_hub_download(REPO, fname, repo_type="dataset", local_dir=TMP))


def iter_selected(path: Path, wanted: dict[int, dict]):
    """Yield (row, image_bytes) for the wanted row positions, streaming BATCH rows at a time."""
    pf = pq.ParquetFile(path, buffer_size=8 << 20, pre_buffer=False)
    offset = 0
    for batch in pf.iter_batches(batch_size=BATCH, columns=["image_data"]):
        col = batch.column(0)
        for j in range(len(col)):
            r = wanted.get(offset + j)
            if r is not None:
                yield r, col[j].as_py()
        offset += len(col)


def chunks(it, n):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(MANIFESTS / "cf_small_pool_v1.parquet"))
    ap.add_argument("--train-per-class", type=int, required=True)
    ap.add_argument("--val-per-class", type=int, required=True)
    ap.add_argument("--max-files", type=int, default=None, help="smoke test: first N files (half real, half fake)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--keep-files", action="store_true")
    args = ap.parse_args()

    pool = pd.read_parquet(args.pool)
    cap = pool.split.map({"train": args.train_per_class, "val": args.val_per_class})
    sel = pool[pool.size_rank < cap].copy()
    # reals first so the JPEG-quality distribution exists before fakes need it
    order = sel.groupby("hf_file").label.mean().sort_values(kind="stable").index.tolist()
    if args.max_files:
        k = args.max_files
        real_f = stats[stats["mean"] < 0.5].sort_values("size", ascending=False).index[: (k + 1) // 2]
        fake_f = stats[stats["mean"] >= 0.5].sort_values("size", ascending=False).index[: k // 2]
        order = [f for f in order if f in set(real_f) | set(fake_f)]
    print(f"selected {len(sel)} rows in {sel.hf_file.nunique()} files; processing {len(order)} files", flush=True)

    outdir = SHARDS / "cf_small"
    writers = {s: ShardWriter(outdir, f"cf_small-{s}") for s in ("train", "val")}
    real_q: list[float] = []
    rng = random.Random(0)
    manifest, t0, n_done, n_bytes, peak = [], time.time(), 0, 0, 0.0

    with ProcessPoolExecutor(args.workers) as cpu, ThreadPoolExecutor(1) as net:
        nxt = net.submit(download, order[0])
        for i, fname in enumerate(order):
            path = nxt.result()
            if i + 1 < len(order):
                nxt = net.submit(download, order[i + 1])          # prefetch the next file
            rows = sel[sel.hf_file == fname]
            wanted = {int(r["hf_row"]): r for r in rows.to_dict("records")}
            for chunk in chunks(iter_selected(path, wanted), BATCH):
                jobs = []
                for r, data in chunk:
                    q = None
                    if r["pre_jpeg"]:
                        if not real_q:
                            raise RuntimeError("no real JPEG qualities measured yet; process real files first")
                        q = rng.choice(real_q)
                    jobs.append((r, data, q))
                for res in cpu.map(process, jobs):
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
                peak = max(peak, rss_gb())
            size_gb = path.stat().st_size / 1e9
            if not args.keep_files:
                path.unlink()
            el = time.time() - t0
            print(f"[{i + 1}/{len(order)}] {fname} ({size_gb:.2f} GB): {len(rows)} rows | total {n_done} crops, "
                  f"{n_bytes / max(n_done, 1) / 1e3:.0f} KB/crop, {el / 60:.1f} min, RAM peak {peak:.2f} GB", flush=True)

    for w in writers.values():
        w.close()
    out = outdir / "cf_small_shards_manifest.parquet"
    pd.DataFrame(manifest).to_parquet(out, index=False)
    np.save(outdir / "real_jpeg_q.npy", np.array(real_q))
    total_gb = sum(p.stat().st_size for p in outdir.glob("*.tar")) / 1e9
    print(f"DONE: {len(manifest)} crops, avg {n_bytes / max(n_done, 1) / 1e3:.0f} KB/crop, "
          f"shards {total_gb:.2f} GB, {(time.time() - t0) / 60:.1f} min -> {outdir}")


if __name__ == "__main__":
    main()
