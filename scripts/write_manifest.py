"""Validate the shard builder's manifest part and write it as an immutable versioned manifest.

    python scripts/write_manifest.py --note "CF-Small 50k train + 10k val crops, pool v1, seed 0"

Input (default): AIDET_DATA_ROOT/index/cf_small_shards_manifest.parquet (from build_shards.py).
Output: data/manifests/manifest_v<YYYYMMDD>_<n>.parquet + changelog .md (tracked in git).

Checks, in addition to dataprep.manifest.validate (docs/manifest_schema.md invariants):
  - every fake generator per split is in that split's committed generator list (configs/splits.yaml),
    and each list file still has the SHA-256 recorded in splits.yaml;
  - exact duplicates (same crop SHA-256) are removed BEFORE writing, keeping the first by sample_id,
    and the count is written to the changelog. Nothing else is ever dropped silently.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataprep.manifest import load_splits, validate, write_version  # noqa: E402
from dataprep.paths import INDEX, REPO_ROOT  # noqa: E402


def generator_list_errors(df: pd.DataFrame, splits: dict) -> list[str]:
    errs = []
    for split in ("train", "val"):
        spec = splits.get(split, {}).get("fake", {}).get("cf_small", {})
        path = REPO_ROOT / spec["generator_list"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec["generator_list_sha256"]:
            errs.append(f"{path.name}: SHA-256 differs from configs/splits.yaml")
        allowed = {ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")}
        got = set(df[(df.label == 1) & (df.split == split) & (df.source_dataset == "cf_small")].generator)
        extra = got - allowed
        if extra:
            errs.append(f"{split}: {len(extra)} generators not in {path.name}, e.g. {sorted(extra)[:5]}")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(INDEX / "cf_small_shards_manifest.parquet"))
    ap.add_argument("--splits", default=str(REPO_ROOT / "configs" / "splits.yaml"))
    ap.add_argument("--note", required=True, help="changelog text: what this version contains and why")
    ap.add_argument("--dry-run", action="store_true", help="validate only, write nothing")
    args = ap.parse_args()

    df = pd.read_parquet(args.input).sort_values("sample_id").reset_index(drop=True)
    splits = load_splits(args.splits)

    dup = df.hash.duplicated(keep="first")
    dropped = df[dup].groupby(["split", "label"]).size().to_dict()
    df = df[~dup].reset_index(drop=True)

    errs = validate(df, splits) + generator_list_errors(df, splits)
    print(f"{len(df)} rows; exact duplicates removed: {int(dup.sum())} {dropped}")
    print(df.groupby(["split", "label"]).size().unstack("label").to_string())
    if errs:
        print("\nINVALID, nothing written:")
        for e in errs:
            print("  -", e)
        sys.exit(1)
    print("valid")
    if args.dry_run:
        return
    note = args.note + f"\n\nSource: `{Path(args.input).name}`. Exact duplicates removed (same crop SHA-256, kept first by sample_id): {int(dup.sum())} {dropped}."
    out = write_version(df, note, args.splits)
    print(f"wrote {out.relative_to(REPO_ROOT)} (+ .md). Point data.manifest in configs/exp/*.yaml at it.")


if __name__ == "__main__":
    main()
