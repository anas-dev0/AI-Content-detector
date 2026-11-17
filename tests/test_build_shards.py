"""Checks the streaming reader and per-image processing of scripts/build_shards.py
on a synthetic Parquet file with the same layout as Community Forensics (one row group)."""
import io
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

# import as a normal module so worker processes (spawn) can import it too
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_shards as bs  # noqa: E402


def _png(seed: int, w=320, h=288) -> bytes:
    arr = np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_iter_selected_streams_the_right_rows(tmp_path):
    n = 300                                                   # > 2 batches of 128
    imgs = [_png(i, 32, 32) for i in range(n)]
    path = tmp_path / "f.parquet"
    pq.write_table(pa.table({"image_data": imgs, "idx": list(range(n))}), path, row_group_size=n)
    assert pq.ParquetFile(path).num_row_groups == 1
    wanted = {i: {"hf_row": i} for i in (0, 5, 127, 128, 129, 255, 256, 299)}
    got = list(bs.iter_selected(path, wanted))
    assert [r["hf_row"] for r, _ in got] == sorted(wanted)
    for r, data in got:
        assert data == imgs[r["hf_row"]]


def test_process_crop_and_pre_jpeg():
    data = _png(1)
    row = {"sample_id": "cf_small/x#1", "pre_jpeg": False}
    res = bs.process((row, data, None))
    crop = Image.open(io.BytesIO(res["png"]))
    assert crop.size == (256, 256)
    x, y, *_ = map(int, res["meta"]["crop_xy"].split(","))
    assert x % 8 == 0 and y % 8 == 0
    orig = np.asarray(Image.open(io.BytesIO(data)))
    assert np.array_equal(np.asarray(crop), orig[y:y + 256, x:x + 256])      # lossless
    assert res["meta"]["format"] == "png" and res["meta"]["pre_jpeg_q"] is None

    res_j = bs.process(({**row, "pre_jpeg": True}, data, 80.0))
    assert res_j["meta"]["pre_jpeg_q"] == 80.0
    assert res_j["meta"]["orig_hash"] == res["meta"]["orig_hash"]          # metadata is of the original
    assert res_j["png"] != res["png"]                                        # JPEG pass changed pixels


def test_tiny_image_skipped():
    assert bs.process(({"sample_id": "s", "pre_jpeg": False}, _png(2, 200, 200), None)) is None
