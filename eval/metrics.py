"""Metrics of the frozen evaluation protocol (docs/evaluation_protocol.md, sections 2 and 6).

Convention: label 1 = fake, 0 = real; higher score = more likely fake.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def _check(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=np.float64)
    if y.shape != s.shape:
        raise ValueError("labels and scores must have the same shape")
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("labels must be 0 (real) / 1 (fake)")
    if not np.isfinite(s).all():
        raise ValueError("scores contain NaN/inf")
    return y, s


def auroc(y, s) -> float:
    y, s = _check(y, s)
    return float(roc_auc_score(y, s))


def average_precision(y, s) -> float:
    y, s = _check(y, s)
    return float(average_precision_score(y, s))


def tpr_at_fpr(y, s, target_fpr: float = 0.01) -> float:
    """Highest TPR achievable with FPR <= target (threshold chosen on this same data)."""
    y, s = _check(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= target_fpr + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def threshold_at_fpr(y_val, s_val, target_fpr: float = 0.01) -> float:
    """Score threshold on VALIDATION reals so that at most `target_fpr` of them are flagged.

    Predict fake iff score > threshold. Applied unchanged to test sets.
    """
    y, s = _check(y_val, s_val)
    real = np.sort(s[y == 0])
    if real.size == 0:
        raise ValueError("no real images in validation scores")
    k = int(np.floor(target_fpr * real.size))          # number of reals allowed above threshold
    return float(real[real.size - 1 - k]) if k < real.size else float(real[0])


def metrics_at_threshold(y, s, thr: float) -> dict:
    y, s = _check(y, s)
    pred = s > thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    return {
        "thr": thr,
        "tpr": tp / pos if pos else float("nan"),
        "fpr": fp / neg if neg else float("nan"),
        "acc": float((pred == (y == 1)).mean()),
        "balanced_acc": 0.5 * ((tp / pos if pos else 0) + (1 - fp / neg if neg else 0)),
    }


METRICS = {"auroc": auroc, "ap": average_precision, "tpr@1%fpr": lambda y, s: tpr_at_fpr(y, s, 0.01)}


@dataclass
class CI:
    value: float
    lo: float
    hi: float
    n_boot: int

    def fmt(self, digits: int = 3) -> str:
        return f"{self.value:.{digits}f} [{self.lo:.{digits}f}, {self.hi:.{digits}f}]"


def bootstrap_ci(y, s, metric=auroc, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> CI:
    """Stratified bootstrap: resample reals and fakes separately (keeps class balance fixed)."""
    y, s = _check(y, s)
    rng = np.random.default_rng(seed)
    idx0, idx1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    if idx0.size == 0 or idx1.size == 0:
        raise ValueError("bootstrap needs both classes")
    vals = np.empty(n_boot)
    for b in range(n_boot):
        i = np.concatenate([rng.choice(idx0, idx0.size), rng.choice(idx1, idx1.size)])
        vals[b] = metric(y[i], s[i])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return CI(metric(y, s), float(lo), float(hi), n_boot)


def per_generator(df: pd.DataFrame, score_col: str = "score", n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Per (source_dataset, generator): that generator's fakes vs ALL reals of the same source.

    Expects columns: source_dataset, generator, label, <score_col> (+ optional seen_tag).
    """
    rows = []
    for src, g in df.groupby("source_dataset"):
        reals = g[g.label == 0]
        if reals.empty:
            continue
        for gen, fakes in g[g.label == 1].groupby("generator"):
            sub = pd.concat([reals, fakes])
            y, s = sub.label.to_numpy(), sub[score_col].to_numpy()
            row = {"source_dataset": src, "generator": gen, "n_fake": len(fakes), "n_real": len(reals)}
            if "seen_tag" in fakes:
                row["seen_tag"] = fakes.seen_tag.iloc[0]
            for name, fn in METRICS.items():
                ci = bootstrap_ci(y, s, fn, n_boot=n_boot, seed=seed)
                row[name], row[f"{name}_lo"], row[f"{name}_hi"] = ci.value, ci.lo, ci.hi
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate(per_gen: pd.DataFrame, by: str = "seen_tag") -> pd.DataFrame:
    """Unweighted mean over generators (large generators must not dominate), per group."""
    cols = list(METRICS)
    return per_gen.groupby(by)[cols].mean().assign(n_generators=per_gen.groupby(by).size())
