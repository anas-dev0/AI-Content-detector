"""Pixel-level shortcut audit on the built shards (PLAN.md week 1, decision D1; CLAUDE.md rule 6).

Question: can a trivially weak view of the crop's pixels predict the label? If yes, a detector
can cheat the same way, and the data must be fixed before any training.

Protocol: fit on TRAIN crops, score VAL crops (val generators are unseen in train, so this
is generator-disjoint by construction). AUROC with stratified bootstrap 95% CI.

  colour_hist    GATE  3 x 32-bin RGB histogram -> logistic regression AND gradient boosting.
                       Both must have upper CI <= 0.6 (the stronger adversary keeps it conservative).
  global_stats   info  per-channel mean/std + mean saturation (gradient boosting)
  clip64         info  CLIP linear probe on crops downsampled to 64 px (content-bias proxy:
                       semantics survive, high-frequency artefacts mostly do not). Optional, GPU.
                       If > 0.6, discuss and mitigate (caption matching) as PLAN.md week 1 says.

    python scripts/pixel_audit.py --manifest data/manifests/manifest_v20261116_1.parquet --name v1
    python scripts/pixel_audit.py --manifest ... --name v1 --clip          # + 64-px CLIP probe
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataprep.paths import DOCS  # noqa: E402
from eval.metrics import auroc, bootstrap_ci  # noqa: E402
from training.data import CropDataset, load_split  # noqa: E402

GATE = 0.6
BINS = 32


def sample(rows: pd.DataFrame, per_class: int) -> pd.DataFrame:
    return rows[rows.size_rank < per_class].reset_index(drop=True) if per_class else rows


def pixel_features(ds: CropDataset, workers: int) -> tuple[np.ndarray, np.ndarray]:
    """(normalised 3x32 histogram, global stats) per crop. Threads open their own file handle per
    item: CropDataset's cached handles are not safe to share across threads (seek + read)."""
    def safe(i):
        with open(ds.tar[i], "rb") as fh:
            fh.seek(ds.offset[i])
            data = fh.read(ds.size[i])
        arr = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        hist = np.concatenate([np.bincount(arr[..., c].ravel() // (256 // BINS), minlength=BINS) for c in range(3)])
        hsv = np.asarray(Image.fromarray(arr).convert("HSV"))
        f = arr.reshape(-1, 3).astype(np.float32)
        return hist / hist[:BINS].sum(), np.concatenate([f.mean(0), f.std(0), [hsv[..., 1].mean()]])
    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(safe, range(len(ds))))
    return np.stack([h for h, _ in out]), np.stack([s for _, s in out])


def clip64_features(ds: CropDataset, model_name: str, pretrained: str, batch: int) -> np.ndarray:
    import open_clip
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=dev)
    model.eval()
    feats = []
    with torch.no_grad():
        for s in range(0, len(ds), batch):
            ims = []
            for i in range(s, min(s + batch, len(ds))):
                im = Image.open(io.BytesIO(ds.read_bytes(i))).convert("RGB")
                ims.append(preprocess(im.resize((64, 64), Image.Resampling.BICUBIC)))   # 64 px, then CLIP's own resize
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                f = model.encode_image(torch.stack(ims).to(dev))
            feats.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
    return np.concatenate(feats)


def score(name, clf, Xtr, ytr, Xva, yva, gated, n_boot, features) -> dict:
    s = clf.fit(Xtr, ytr).predict_proba(Xva)[:, 1]
    ci = bootstrap_ci(yva, s, auroc, n_boot=n_boot, seed=0)
    return {"baseline": name, "auroc": ci.value, "lo": ci.lo, "hi": ci.hi, "gated": gated,
            "pass": ci.hi <= GATE, "features": features, "n_val": int(len(yva))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--shards-root", default=None)
    ap.add_argument("--train-per-class", type=int, default=10000, help="0 = all")
    ap.add_argument("--val-per-class", type=int, default=5000, help="0 = all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--clip", action="store_true")
    ap.add_argument("--clip-model", default="ViT-B-32")
    ap.add_argument("--clip-pretrained", default="openai")
    args = ap.parse_args()

    tr = sample(load_split(args.manifest, "train", None, args.shards_root), args.train_per_class)
    va = sample(load_split(args.manifest, "val", None, args.shards_root), args.val_per_class)
    ds_tr, ds_va = CropDataset(tr, 256), CropDataset(va, 256)
    ytr, yva = ds_tr.label, ds_va.label
    print(f"train {len(tr)} {tr.label.value_counts().to_dict()}, val {len(va)} {va.label.value_counts().to_dict()}", flush=True)

    Htr, Str = pixel_features(ds_tr, args.workers)
    Hva, Sva = pixel_features(ds_va, args.workers)
    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))   # noqa: E731
    hgb = lambda: HistGradientBoostingClassifier(max_iter=200, random_state=0)                # noqa: E731
    res = [
        score("colour_hist (logreg)", lr(), Htr, ytr, Hva, yva, True, args.n_boot, f"3x{BINS}-bin RGB histogram"),
        score("colour_hist (boosting)", hgb(), Htr, ytr, Hva, yva, True, args.n_boot, f"3x{BINS}-bin RGB histogram"),
        score("global_stats (boosting)", hgb(), Str, ytr, Sva, yva, False, args.n_boot, "RGB mean/std, mean saturation"),
    ]
    if args.clip:
        print("CLIP 64-px features ...", flush=True)
        Ctr = clip64_features(ds_tr, args.clip_model, args.clip_pretrained, 256)
        Cva = clip64_features(ds_va, args.clip_model, args.clip_pretrained, 256)
        res.append(score(f"clip64 ({args.clip_model}/{args.clip_pretrained}, logreg)", lr(), Ctr, ytr, Cva, yva,
                         False, args.n_boot, "CLIP embedding of the crop downsampled to 64 px"))

    gate_ok = all(r["pass"] for r in res if r["gated"])
    msha = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    lines = [f"# Pixel-level shortcut audit: {args.name}", "",
             f"Manifest: `{Path(args.manifest).name}` (sha256 `{msha[:16]}…`). Fit on train crops "
             f"({len(tr)}: {int((ytr == 0).sum())} real / {int((ytr == 1).sum())} fake), scored on val crops "
             f"({len(va)}: {int((yva == 0).sum())} real / {int((yva == 1).sum())} fake). Val generators are unseen in train.",
             f"Gate: AUROC upper 95% CI <= {GATE} (stratified bootstrap, {args.n_boot} resamples).", "",
             "| Baseline | Role | AUROC [95% CI] | <= 0.6 | Features |", "|---|---|---|---|---|"]
    for r in res:
        lines.append(f"| {r['baseline']} | {'**gate**' if r['gated'] else 'info'} | {r['auroc']:.3f} "
                     f"[{r['lo']:.3f}, {r['hi']:.3f}] | {'yes' if r['pass'] else 'NO'} | {r['features']} |")
    lines += ["", f"**Verdict: {'PASS' if gate_ok else 'FAIL: fix the data before training'}**", ""]
    if args.clip and res[-1]["auroc"] > GATE:
        lines.insert(-2, "\nThe 64-px CLIP content probe is above 0.6: content differs between classes. "
                         "Discuss and mitigate (caption matching) before reporting results.")
    out = DOCS / f"audit_pixel_{args.name}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"manifest": str(args.manifest), "manifest_sha256": msha,
                                                    "results": res, "pass": gate_ok}, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {out}")
    sys.exit(0 if gate_ok else 2)


if __name__ == "__main__":
    main()
