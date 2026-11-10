"""Manifest schema, invariant checks and versioned writing (docs/manifest_schema.md)."""
from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

import pandas as pd
import yaml

from dataprep.paths import MANIFESTS

REQUIRED = ["path", "label", "generator", "architecture", "source_dataset", "split",
            "format", "width", "height", "jpeg_q", "hash"]
SPLITS = {"train", "val", "test"}
FORMATS = {"jpeg", "png", "webp", "other"}


def splits_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_splits(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def validate(df: pd.DataFrame, splits_cfg: dict | None = None) -> list[str]:
    """Return a list of human-readable violations (empty list = valid)."""
    errs: list[str] = []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return [f"missing required columns: {missing}"]

    if df.hash.duplicated().any():
        errs.append(f"duplicate hash values: {int(df.hash.duplicated().sum())} rows")
    if not set(df.split) <= SPLITS:
        errs.append(f"invalid split values: {set(df.split) - SPLITS}")
    if not set(df.format) <= FORMATS:
        errs.append(f"invalid format values: {set(df.format) - FORMATS}")
    if not set(df.label.unique()) <= {0, 1}:
        errs.append("label must be 0/1")
    if ((df.label == 0) != (df.generator == "real")).any():
        errs.append("label==0 must coincide with generator=='real'")
    jq_ok = df.jpeg_q.isna() == (df.format != "jpeg")
    if not jq_ok.all():
        errs.append(f"jpeg_q must be set iff format=='jpeg' ({int((~jq_ok).sum())} rows)")

    for key in ("parent_id", "video_id"):
        if key in df.columns:
            sub = df.dropna(subset=[key])
            n_split = sub.groupby(key).split.nunique()
            if (n_split > 1).any():
                errs.append(f"{key} spans several splits: {list(n_split[n_split > 1].index[:5])} ...")

    fakes = df[df.label == 1]
    gens = {s: set(fakes[fakes.split == s].generator) for s in SPLITS}
    allowed = set((splits_cfg or {}).get("test", {}).get("seen_in_test", []) or [])
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = (gens[a] & gens[b]) - allowed
        if overlap:
            errs.append(f"generators in both {a} and {b}: {sorted(overlap)[:10]}")

    if splits_cfg and "real_source" in df.columns:
        held = set(splits_cfg.get("held_out_real_sources", []))
        bad = df[(df.split == "train") & (df.label == 0) & df.real_source.isin(held)]
        if len(bad):
            errs.append(f"held-out real sources in train: {sorted(set(bad.real_source))}")
    return errs


def summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["split", "source_dataset", "label"]).size().unstack("label", fill_value=0)


def write_version(df: pd.DataFrame, note: str, splits_path: str | Path | None = None) -> Path:
    """Write an immutable manifest version + a changelog .md next to it."""
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    day = dt.date.today().strftime("%Y%m%d")
    n = 1 + len(list(MANIFESTS.glob(f"manifest_v{day}_*.parquet")))
    out = MANIFESTS / f"manifest_v{day}_{n}.parquet"
    if out.exists():
        raise FileExistsError(out)
    df.to_parquet(out, index=False)
    lines = [f"# {out.name}", "", note, "", f"rows: {len(df)}"]
    if splits_path:
        lines.append(f"splits_sha256: {splits_sha256(splits_path)}")
    lines += ["", "```", summary(df).to_string(), "```", ""]
    out.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    return out
