"""Central path handling.

Large data lives OUTSIDE the repo (the repo sits in a OneDrive-synced folder and the
data is hundreds of GB and mostly non-commercial). Set AIDET_DATA_ROOT to override.
Small, tracked artefacts (manifests, configs) stay in the repo.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("AIDET_DATA_ROOT", "C:/aidet_data" if os.name == "nt" else "/kaggle/working/aidet_data"))

RAW = DATA_ROOT / "raw"            # streamed / downloaded source data
INDEX = DATA_ROOT / "index"        # metadata-only indexes of remote datasets
SHARDS = DATA_ROOT / "shards"      # WebDataset tar shards (lossless crops)
DEGRADED = DATA_ROOT / "degraded"  # offline, seeded degraded test sets
OWN_2026 = DATA_ROOT / "own_2026"  # our generated + real photos

MANIFESTS = REPO_ROOT / "data" / "manifests"   # tracked in git
CONFIGS = REPO_ROOT / "configs"
DOCS = REPO_ROOT / "docs"


def ensure_dirs() -> None:
    for p in (RAW, INDEX, SHARDS, DEGRADED, OWN_2026, MANIFESTS):
        p.mkdir(parents=True, exist_ok=True)
