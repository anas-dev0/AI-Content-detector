"""Build a metadata-only index of a Parquet dataset hosted on the Hugging Face Hub.

Parquet is columnar, so we can fetch only the small metadata columns and skip the
`image_data` column (which holds ~all the bytes). This indexes the 260 GB
Community Forensics-Small in minutes, without downloading images.

Example:
    python scripts/index_hf_parquet.py --repo OwensLab/CommunityForensics-Small --name cf_small
    python scripts/index_hf_parquet.py --repo OwensLab/CommunityForensics-Eval --name cf_eval
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataprep.paths import INDEX, ensure_dirs  # noqa: E402

SKIP_COLUMNS = {"image_data", "image", "prompt"}


def index_file(fs: HfFileSystem, path: str) -> pd.DataFrame:
    with fs.open(path, "rb") as fh:
        pf = pq.ParquetFile(fh)
        cols = [c for c in pf.schema_arrow.names if c not in SKIP_COLUMNS]
        df = pf.read(columns=cols).to_pandas()
    df["hf_file"] = path.split("/", 3)[-1]          # e.g. data/HFCF_small_0.parquet
    df["hf_row"] = range(len(df))                   # row position inside that file
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--name", required=True, help="output name, e.g. cf_small")
    ap.add_argument("--glob", default="data/**/*.parquet")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="index only the first N files (debug)")
    args = ap.parse_args()

    ensure_dirs()
    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{args.repo}/{args.glob}"))
    if args.limit:
        files = files[: args.limit]
    print(f"{args.repo}: {len(files)} parquet files", flush=True)

    parts, failed = [], []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(index_file, fs, f): f for f in files}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                parts.append(fut.result())
            except Exception as e:  # keep going; report at the end
                failed.append((futs[fut], repr(e)))
            if i % 20 == 0 or i == len(files):
                print(f"  {i}/{len(files)} files indexed", flush=True)

    df = pd.concat(parts, ignore_index=True)
    df["source_repo"] = args.repo
    out = INDEX / f"{args.name}_index.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out}  rows={len(df)}  failed_files={len(failed)}")
    for f, e in failed:
        print("  FAILED", f, e)


if __name__ == "__main__":
    main()
