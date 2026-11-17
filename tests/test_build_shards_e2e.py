"""End-to-end run of scripts/build_shards.py on two synthetic source files (no network):
reals-first ordering, JPEG equalisation, resumability, and the final manifest."""
import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from dataprep.manifest import validate

# import as a normal module so worker processes (spawn) can import it too
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_shards as bs  # noqa: E402


def _img(seed, fmt, q=90, w=320, h=288):
    arr = np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt, **({"quality": q} if fmt == "JPEG" else {}))
    return buf.getvalue()


def _setup(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    n = 20
    files = {"data/HFCF_small_1.parquet": [_img(i, "JPEG", 70 + i) for i in range(n)],      # reals
             "data/HFCF_small_2.parquet": [_img(100 + i, "PNG") for i in range(n)]}        # fakes
    for name, imgs in files.items():
        pq.write_table(pa.table({"image_data": imgs}), src / Path(name).name, row_group_size=n)
    rows = []
    for name, label in (("data/HFCF_small_1.parquet", 0), ("data/HFCF_small_2.parquet", 1)):
        for i in range(n):
            rows.append({"sample_id": f"cf_small/{Path(name).name}#{i}", "hf_file": name, "hf_row": i,
                         "label": label, "generator": "real" if label == 0 else ("gen-a" if i < 15 else "gen-b"),
                         "architecture": "real" if label == 0 else "ldm", "subset": "x",
                         "split": "train" if i < 15 else "val", "mp_bucket": "M",
                         "pre_jpeg": bool(label == 1 and i % 2 == 0), "size_rank": i % 15, "nsfw_flag": False})
    pool = tmp_path / "pool.parquet"
    pd.DataFrame(rows).to_parquet(pool)

    calls = []

    def fake_download(fname):
        calls.append(fname)
        dst = tmp_path / "dl" / Path(fname).name
        dst.parent.mkdir(exist_ok=True)
        dst.write_bytes((src / Path(fname).name).read_bytes())
        return dst

    monkeypatch.setattr(bs, "download", fake_download)
    monkeypatch.setattr(bs, "SHARDS", tmp_path / "shards")
    monkeypatch.setattr(bs, "INDEX", tmp_path / "index")
    argv = ["build_shards.py", "--pool", str(pool), "--train-per-class", "100", "--val-per-class", "100", "--workers", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    return calls


def test_build_resume_and_manifest(tmp_path, monkeypatch):
    calls = _setup(tmp_path, monkeypatch)
    bs.main()
    assert calls == ["data/HFCF_small_1.parquet", "data/HFCF_small_2.parquet"]       # reals first

    out = tmp_path / "shards" / "cf_small" / "cf_small_shards_manifest.parquet"
    m = pd.read_parquet(out)
    assert len(m) == 40 and (tmp_path / "index" / "cf_small_shards_manifest.parquet").exists()
    assert validate(m) == []
    fakes = m[m.label == 1]
    assert fakes.pre_jpeg_q.notna().sum() == 10                                     # JPEG equalisation applied
    assert set(fakes.pre_jpeg_q.dropna()) <= set(m[m.label == 0].jpeg_q)           # qualities come from reals

    # every manifest row points at a real PNG inside a shard
    for shard, g in m.groupby("shard"):
        with tarfile.open(tmp_path / "shards" / "cf_small" / shard) as t:
            names = set(t.getnames())
        assert all(p.split("#")[1] + ".png" in names for p in g.path)

    # second run: nothing downloaded again, identical manifest
    calls.clear()
    bs.main()
    assert calls == []
    pd.testing.assert_frame_equal(pd.read_parquet(out), m)
