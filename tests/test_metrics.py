import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from eval.metrics import (
    aggregate, auroc, average_precision, bootstrap_ci, metrics_at_threshold,
    per_generator, threshold_at_fpr, tpr_at_fpr,
)
from eval.tagging import tag_seen


def test_perfect_and_reversed():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert auroc(y, s) == 1.0
    assert auroc(y, -s) == 0.0
    assert average_precision(y, s) == 1.0
    assert tpr_at_fpr(y, s, 0.01) == 1.0


def test_matches_sklearn_random():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    s = rng.normal(size=2000) + 0.8 * y
    assert auroc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_tpr_at_fpr_known_case():
    # 100 reals at 0..99, 10 fakes at 98.5 + 0..9 -> threshold above 98 flags 1 real (1% FPR)
    reals = np.arange(100.0)
    fakes = 98.5 + np.arange(10.0)
    y = np.r_[np.zeros(100), np.ones(10)]
    s = np.r_[reals, fakes]
    assert tpr_at_fpr(y, s, 0.01) == 1.0
    assert tpr_at_fpr(y, s, 0.0) == pytest.approx(0.9)   # fake at 98.5 is below real 99


def test_val_threshold_respects_fpr():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(1000), np.ones(1000)]
    s = np.r_[rng.normal(size=1000), rng.normal(2, 1, 1000)]
    thr = threshold_at_fpr(y, s, 0.01)
    m = metrics_at_threshold(y, s, thr)
    assert m["fpr"] <= 0.01
    assert 0.0 < m["tpr"] <= 1.0


def test_bootstrap_contains_point_estimate():
    rng = np.random.default_rng(2)
    y = np.r_[np.zeros(300), np.ones(300)]
    s = np.r_[rng.normal(size=300), rng.normal(1, 1, 300)]
    ci = bootstrap_ci(y, s, auroc, n_boot=200, seed=0)
    assert ci.lo <= ci.value <= ci.hi
    assert ci.hi - ci.lo < 0.15
    assert bootstrap_ci(y, s, auroc, n_boot=200, seed=0) == ci   # deterministic


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        auroc([0, 1, 2], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        auroc([0, 1], [0.1, float("nan")])


def test_per_generator_and_tagging():
    rng = np.random.default_rng(3)
    train = pd.DataFrame({"label": [1, 1, 0], "generator": ["sd-1.4", "biggan", "real"],
                          "architecture": ["ldm-unet", "gan", "real"]})
    n = 200
    test = pd.DataFrame({
        "source_dataset": "x",
        "label": [0] * n + [1] * n + [1] * n + [1] * n,
        "generator": ["real"] * n + ["sd-1.4"] * n + ["sdxl"] * n + ["dalle3"] * n,
        "architecture": ["real"] * n + ["ldm-unet"] * n + ["ldm-unet"] * n + ["commercial"] * n,
    })
    test["score"] = rng.normal(size=len(test)) + 1.5 * test.label
    test["seen_tag"] = tag_seen(test, train)
    assert set(test.seen_tag) == {"real", "seen", "unseen_same_arch", "unseen_new_arch"}
    pg = per_generator(test, n_boot=50)
    assert len(pg) == 3 and (pg.n_real == n).all()
    agg = aggregate(pg)
    assert set(agg.index) == {"seen", "unseen_same_arch", "unseen_new_arch"}
