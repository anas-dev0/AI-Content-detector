import pandas as pd

from dataprep.manifest import validate


def _row(**kw):
    base = dict(path="p", label=1, generator="sd-1.4", architecture="ldm-unet", source_dataset="cf_small",
                split="train", format="png", width=512, height=512, jpeg_q=None, hash="h0",
                real_source=None, parent_id=None)
    base.update(kw)
    return base


def _df(rows):
    return pd.DataFrame(rows)


def test_valid_manifest():
    df = _df([
        _row(hash="a"),
        _row(hash="b", label=0, generator="real", architecture="real", format="jpeg", jpeg_q=90.0, real_source="coco"),
        _row(hash="c", generator="dalle3", architecture="commercial", split="test"),
    ])
    assert validate(df, {"held_out_real_sources": ["raise"]}) == []


def test_generator_leak_detected():
    df = _df([_row(hash="a"), _row(hash="b", split="test")])
    errs = validate(df)
    assert any("train and test" in e for e in errs)
    assert validate(df, {"test": {"seen_in_test": ["sd-1.4"]}}) == []


def test_duplicates_and_parent_and_heldout():
    df = _df([
        _row(hash="a", parent_id="p1"),
        _row(hash="a", parent_id="p1", split="val", generator="other"),
        _row(hash="c", label=0, generator="real", architecture="real", real_source="raise"),
    ])
    errs = " | ".join(validate(df, {"held_out_real_sources": ["raise"]}))
    assert "duplicate hash" in errs
    assert "parent_id spans several splits" in errs
    assert "held-out real sources in train" in errs


def test_jpeg_q_consistency():
    errs = validate(_df([_row(format="jpeg", jpeg_q=None)]))
    assert any("jpeg_q" in e for e in errs)
