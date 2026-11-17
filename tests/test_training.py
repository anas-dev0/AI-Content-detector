"""Training skeleton: data access, label-independent augmentation, exact resume."""
import io
import json
from itertools import islice

import numpy as np
import pytest
import torch
from PIL import Image

from training.data import CropDataset, StreamSampler, aligned_offset, load_split
from training.toy import make_toy_dataset
from training.train import DEFAULTS, lr_at, merge, train


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    root = tmp_path_factory.mktemp("toy")
    manifest = make_toy_dataset(root, train_per_class=48, val_per_class=16, size=64)
    return root, manifest


def _rows(toy, split="train", n=None):
    root, manifest = toy
    return load_split(manifest, split, n, shards_root=root, cache_dir=root / "tar_index")


def test_tar_index_reads_exact_png(toy):
    rows = _rows(toy)
    ds = CropDataset(rows, input_size=32)
    for i in (0, 7, len(ds) - 1):
        img = Image.open(io.BytesIO(ds.read_bytes(i)))
        assert img.format == "PNG" and img.size == (64, 64)


def test_nested_subsets(toy):
    small, big = _rows(toy, n=20), _rows(toy, n=60)
    assert small.label.value_counts().to_dict() == {0: 10, 1: 10}
    assert set(small.sample_id) <= set(big.sample_id)
    with pytest.raises(ValueError):
        _rows(toy, n=1000)


def test_aligned_offsets():
    import random
    rng = random.Random(0)
    offs = {aligned_offset(256, 224, rng) for _ in range(500)}
    assert offs == {0, 8, 16, 24, 32}
    assert aligned_offset(256, 224) == 16 and aligned_offset(224, 224) == 0


def test_augmentation_is_label_independent_and_deterministic(toy):
    rows = _rows(toy).head(4).copy()
    aug = {"jpeg_p": 1.0, "jpeg_q": [60, 100], "blur_p": 0.5, "hflip": True}
    a = CropDataset(rows, input_size=32, train=True, aug=aug)
    flipped = rows.assign(label=1 - rows.label)
    b = CropDataset(flipped, input_size=32, train=True, aug=aug)
    for i in range(4):
        xa, ya = a[(i, 3)]
        xb, yb = b[(i, 3)]
        assert torch.equal(xa, xb) and ya != yb            # same pixels whatever the label
        assert torch.equal(xa, a[(i, 3)][0])               # deterministic
    assert any(not torch.equal(a[(i, 0)][0], a[(i, 1)][0]) for i in range(4))   # varies by epoch


def test_eval_transform_has_no_randomness(toy):
    ds = CropDataset(_rows(toy, "val"), input_size=32, train=False)
    assert torch.equal(ds[0][0], ds[0][0])


def test_stream_sampler_resume_is_exact():
    full = list(islice(iter(StreamSampler(10, seed=3)), 35))
    for k in (0, 4, 10, 23):
        assert list(islice(iter(StreamSampler(10, seed=3, start=k)), 35 - k)) == full[k:]
    first_epoch = [i for i, e in full if e == 0]
    assert sorted(first_epoch) == list(range(10))


def test_lr_schedule():
    cfg = merge(DEFAULTS, {"optim": {"lr": 1e-3, "lr_ref_batch": 128, "warmup_samples": 1000},
                           "train": {"batch_size": 128, "total_samples": 10000}})
    assert lr_at(0, cfg) < lr_at(512, cfg) < lr_at(896, cfg) == pytest.approx(1e-3)
    assert lr_at(1000, cfg) == pytest.approx(1e-3)
    assert lr_at(9999, cfg) < 1e-5


def _cfg(toy, total, **train_kw):
    root, manifest = toy
    return merge(DEFAULTS, {
        "run": "pytest_toy", "seed": 0,
        "data": {"manifest": str(manifest), "shards_root": str(root), "tar_index_cache": str(root / "tar_index"),
                 "input_size": 32, "num_workers": 0},
        "model": {"name": "vit_tiny_patch16_224", "kwargs": {"img_size": 32, "patch_size": 8, "depth": 2}},
        "optim": {"lr": 1e-3, "lr_ref_batch": 16, "warmup_samples": 32},
        "train": {"batch_size": 16, "total_samples": total, "eval_every_samples": 64,
                  "log_every_steps": 2, "amp": False, **train_kw},
    })


def test_interrupted_run_resumes_to_identical_weights(toy, tmp_path):
    a = tmp_path / "a"
    train(_cfg(toy, 160), a / "exp", a / "ckpt", device="cpu")

    b = tmp_path / "b"
    cfg = _cfg(toy, 160, max_minutes=1e-9)                 # stops right after the first step
    r = train(cfg, b / "exp", b / "ckpt", device="cpu")
    assert not r["finished"] and r["samples_seen"] == 16
    r = train(_cfg(toy, 160), b / "exp", b / "ckpt", device="cpu")
    assert r["finished"] and r["samples_seen"] == 160

    wa = torch.load(a / "ckpt" / "last.pt", weights_only=False)["model"]
    wb = torch.load(b / "ckpt" / "last.pt", weights_only=False)["model"]
    assert all(torch.equal(wa[k], wb[k]) for k in wa)

    info = json.loads((b / "exp" / "run_info.json").read_text())
    assert info["manifest_sha256"] and len(info["resumes"]) == 1
    assert info["reportable"] is False                     # toy manifest, splits not frozen
    assert (b / "exp" / "val_best.json").exists() and (b / "ckpt" / "best.pt").exists()
    # rerunning a finished run does nothing
    assert train(_cfg(toy, 160), b / "exp", b / "ckpt", device="cpu")["samples_seen"] == 160


def test_resume_refuses_changed_manifest(toy, tmp_path):
    root, manifest = toy
    train(_cfg(toy, 32), tmp_path / "exp", tmp_path / "ckpt", device="cpu")
    import pandas as pd
    other = tmp_path / "other.parquet"
    pd.read_parquet(manifest).assign(note="x").to_parquet(other)
    cfg = _cfg(toy, 64)
    cfg["data"]["manifest"] = str(other)
    with pytest.raises(RuntimeError, match="manifest_sha256"):
        train(cfg, tmp_path / "exp", tmp_path / "ckpt", device="cpu")


def test_toy_pipeline_beats_chance(toy, tmp_path):
    """Pipeline sanity (labels reach the loss, scores reach the metric), not augmentation robustness:
    the toy artefact is a Nyquist-frequency pattern that JPEG/blur largely erase, so aug is off here."""
    cfg = _cfg(toy, 1536)
    cfg["aug"] = {"hflip": True, "jpeg_p": 0.0, "blur_p": 0.0}
    r = train(cfg, tmp_path / "exp", tmp_path / "ckpt", device="cpu")
    assert r["best"]["val_auroc"] > 0.75
