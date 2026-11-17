"""Shared training skeleton for the trained tracks (T3 ResNet-50, T4 NPR, T5 scratch ViT).

    python -m training.train --config configs/exp/s0_vit_ti16_10k.yaml

- Config YAML -> experiments/<run>/ (resolved config, run_info.json, log.jsonl, val_best.json).
  Checkpoints go to AIDET_DATA_ROOT/checkpoints/<run>/ (never inside the OneDrive repo).
- fp16 autocast + GradScaler on CUDA (the T4 has no bf16). fp32 on CPU.
- Checkpoint every `ckpt_minutes`, at every eval and at the end; resumes automatically from
  last.pt with the exact data order (see training/data.py: StreamSampler).
- `max_minutes` stops cleanly (checkpoint saved) before Kaggle's 12-h session limit.
- Model selection on val only (rule 4): best.pt = highest val AUROC. Test is never touched here.
- run_info.json records git commit, manifest path + sha256, splits.yaml sha256 (rule 7). A run is
  flagged reportable only if splits.yaml is frozen, the manifest is a versioned one and git was clean.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
import yaml
from timm.optim import create_optimizer_v2
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataprep.manifest import load_splits, splits_sha256  # noqa: E402
from dataprep.paths import DATA_ROOT, INDEX, REPO_ROOT  # noqa: E402
from eval.metrics import auroc, average_precision  # noqa: E402
from training.data import CropDataset, StreamSampler, load_split  # noqa: E402

DEFAULTS: dict = {
    "run": None,
    "seed": 0,
    "splits": "configs/splits.yaml",
    "data": {
        "manifest": None, "shards_root": None, "tar_index_cache": None,
        "train_images": None, "val_images": None,            # None = whole split
        "input_size": 224, "geometry": "crop",
        "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
        "num_workers": 4,
    },
    "aug": {"hflip": True, "jpeg_p": 0.5, "jpeg_q": [60, 100], "blur_p": 0.1, "blur_sigma": [0.1, 3.0]},
    "model": {"name": "vit_tiny_patch16_224", "kwargs": {}},
    "optim": {"lr": 5.0e-4, "lr_ref_batch": 512, "weight_decay": 0.05, "betas": [0.9, 0.999],
              "warmup_samples": 0, "clip_grad": 1.0, "label_smoothing": 0.1, "min_lr_ratio": 0.0},
    "train": {"batch_size": 128, "total_samples": 1_000_000, "eval_every_samples": 100_000,
              "ckpt_minutes": 30, "max_minutes": None, "amp": True, "log_every_steps": 50},
}


def merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "kwargs" else v
    return out


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        cfg = merge(DEFAULTS, yaml.safe_load(fh))
    if not cfg["run"] or not cfg["data"]["manifest"]:
        raise ValueError("config needs `run` and `data.manifest`")
    return cfg


def resolve(p: str | Path | None) -> Path | None:
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


def provenance(cfg: dict) -> dict:
    manifest, splits = resolve(cfg["data"]["manifest"]), resolve(cfg["splits"])
    sp = load_splits(splits) if splits and splits.exists() else {}
    git = git_state()
    info = {
        "manifest": str(manifest), "manifest_sha256": file_sha256(manifest),
        "splits": str(splits), "splits_sha256": splits_sha256(splits) if sp else None,
        "splits_frozen": bool(sp.get("frozen")), **{f"git_{k}": v for k, v in git.items()},
    }
    reasons = []
    if not info["splits_frozen"]:
        reasons.append("splits.yaml not frozen")
    if not manifest.name.startswith("manifest_v"):
        reasons.append("manifest is not a versioned data/manifests/manifest_v*.parquet")
    if git["dirty"] is not False:
        reasons.append("git working tree dirty or unknown")
    info["reportable"], info["not_reportable_because"] = not reasons, reasons
    return info


def lr_at(samples: int, cfg: dict) -> float:
    o, t = cfg["optim"], cfg["train"]
    peak = o["lr"] * t["batch_size"] / o["lr_ref_batch"]
    w = o["warmup_samples"]
    if w and samples < w:
        return peak * min(1.0, (samples + t["batch_size"]) / w)
    frac = min(1.0, (samples - w) / max(1, t["total_samples"] - w))
    lo = peak * o["min_lr_ratio"]
    return lo + 0.5 * (peak - lo) * (1 + math.cos(math.pi * frac))


def build_model(cfg: dict) -> torch.nn.Module:
    m = cfg["model"]
    return timm.create_model(m["name"], pretrained=False, num_classes=2, **(m.get("kwargs") or {}))


def score_fn(logits: torch.Tensor) -> torch.Tensor:
    """Higher = more likely fake: logit(fake) - logit(real)."""
    logits = logits.float()
    return logits[:, 1] - logits[:, 0]


@torch.no_grad()
def evaluate(model, loader, device, amp: bool) -> dict:
    model.eval()
    ys, ss, loss, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
        loss += torch.nn.functional.cross_entropy(logits.float(), y, reduction="sum").item()
        n += len(y)
        ys.append(y.cpu().numpy())
        ss.append(score_fn(logits).cpu().numpy())
    model.train()
    y, s = np.concatenate(ys), np.concatenate(ss)
    return {"val_loss": loss / n, "val_auroc": auroc(y, s), "val_ap": average_precision(y, s), "n_val": int(n)}


def atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(cfg: dict, exp_dir: Path | None = None, ckpt_dir: Path | None = None, device: str | None = None) -> dict:
    exp_dir = exp_dir or REPO_ROOT / "experiments" / cfg["run"]
    ckpt_dir = ckpt_dir or DATA_ROOT / "checkpoints" / cfg["run"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    d, t = cfg["data"], cfg["train"]
    amp = bool(t["amp"]) and device.type == "cuda"
    torch.manual_seed(cfg["seed"])

    prov = provenance(cfg)
    last = ckpt_dir / "last.pt"
    state = torch.load(last, map_location="cpu", weights_only=False) if last.exists() else None
    if state is not None and state["samples_seen"] >= t["total_samples"]:
        print(f"{cfg['run']} already finished ({state['samples_seen']} samples); nothing to do", flush=True)
        return {"samples_seen": state["samples_seen"], "finished": True, "best": state["best"]}
    info_path = exp_dir / "run_info.json"
    if state is not None:                                     # resume: data/protocol must not change
        info = json.loads(info_path.read_text(encoding="utf-8"))
        for k in ("manifest_sha256", "splits_sha256"):
            if info[k] != prov[k]:
                raise RuntimeError(f"cannot resume {cfg['run']}: {k} changed since the run started")
        if state["config"]["train"]["batch_size"] != t["batch_size"]:
            raise RuntimeError("cannot resume with a different batch_size (data order would change)")
        info.setdefault("resumes", []).append({"at_samples": state["samples_seen"], "time": dt.datetime.now().isoformat(),
                                               "git_commit": prov["git_commit"], "git_dirty": prov["git_dirty"]})
        info["reportable"] = info["reportable"] and prov["reportable"]
        info["not_reportable_because"] = sorted(set(info["not_reportable_because"]) | set(prov["not_reportable_because"]))
    else:
        info = {"run": cfg["run"], "started": dt.datetime.now().isoformat(), **prov,
                "torch": torch.__version__, "timm": timm.__version__,
                "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}
        (exp_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    cache = resolve(d["tar_index_cache"]) if d["tar_index_cache"] else INDEX / "tar_index"
    manifest = resolve(d["manifest"])
    tr_rows = load_split(manifest, "train", d["train_images"], d["shards_root"], cache)
    va_rows = load_split(manifest, "val", d["val_images"], d["shards_root"], cache)
    common = dict(input_size=d["input_size"], geometry=d["geometry"], mean=d["mean"], std=d["std"])
    tr_ds = CropDataset(tr_rows, train=True, aug=cfg["aug"], seed=cfg["seed"], **common)
    va_ds = CropDataset(va_rows, train=False, **common)

    model = build_model(cfg).to(device)
    o = cfg["optim"]
    opt = create_optimizer_v2(model, "adamw", lr=o["lr"], weight_decay=o["weight_decay"], betas=tuple(o["betas"]))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    samples, best = 0, {"val_auroc": -1.0}
    if state is not None:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        samples, best = state["samples_seen"], state["best"]
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(state["cuda_rng"], device)
        print(f"resumed {cfg['run']} at {samples} samples", flush=True)

    nw, bs = d["num_workers"], t["batch_size"]
    loader_kw = dict(num_workers=nw, pin_memory=device.type == "cuda", persistent_workers=nw > 0)
    tr_dl = DataLoader(tr_ds, batch_size=bs, sampler=StreamSampler(len(tr_ds), cfg["seed"], samples),
                       drop_last=False, **loader_kw)
    va_dl = DataLoader(va_ds, batch_size=bs, shuffle=False, **loader_kw)
    print(f"{cfg['run']}: train {len(tr_ds)} ({tr_rows.label.value_counts().to_dict()}), val {len(va_ds)}, "
          f"device {device}, amp {amp}, reportable {info['reportable']}", flush=True)

    log = (exp_dir / "log.jsonl").open("a", encoding="utf-8")

    def checkpoint():
        atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                     "samples_seen": samples, "best": best, "config": cfg, "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}, last)

    def do_eval():
        nonlocal best
        r = {"samples_seen": samples, **evaluate(model, va_dl, device, amp), "time": dt.datetime.now().isoformat()}
        log.write(json.dumps({"event": "eval", **r}) + "\n")
        log.flush()
        print(f"  eval @ {samples}: val AUROC {r['val_auroc']:.4f}  AP {r['val_ap']:.4f}  loss {r['val_loss']:.4f}", flush=True)
        if r["val_auroc"] > best["val_auroc"]:
            best = r
            atomic_save({"model": model.state_dict(), "samples_seen": samples, "config": cfg, "val": r}, ckpt_dir / "best.pt")
            (exp_dir / "val_best.json").write_text(json.dumps({**r, **{k: info[k] for k in (
                "manifest", "manifest_sha256", "splits_sha256", "git_commit", "reportable")}}, indent=2), encoding="utf-8")

    crit = torch.nn.CrossEntropyLoss(label_smoothing=o["label_smoothing"])
    t_start = t_ckpt = t_log = time.time()
    n_log, stopped_early = 0, False
    model.train()
    it = iter(tr_dl)
    while samples < t["total_samples"]:
        x, y = next(it)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        lr = lr_at(samples, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            loss = crit(model(x).float(), y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at {samples} samples")
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if o["clip_grad"]:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), o["clip_grad"])
        scaler.step(opt)
        scaler.update()
        prev, samples = samples, samples + len(y)
        n_log += len(y)

        step = samples // bs
        if step % t["log_every_steps"] == 0:
            now = time.time()
            rec = {"event": "train", "samples_seen": samples, "loss": loss.item(), "lr": lr,
                   "img_per_s": n_log / max(now - t_log, 1e-9),
                   "vram_gb": torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"[{samples}/{t['total_samples']}] loss {rec['loss']:.4f} lr {lr:.2e} {rec['img_per_s']:.0f} img/s", flush=True)
            t_log, n_log = now, 0
        if samples // t["eval_every_samples"] > prev // t["eval_every_samples"] and samples < t["total_samples"]:
            do_eval()
            checkpoint()
            t_ckpt = time.time()
        if time.time() - t_ckpt > 60 * t["ckpt_minutes"]:
            checkpoint()
            t_ckpt = time.time()
        if t["max_minutes"] and time.time() - t_start > 60 * t["max_minutes"]:
            stopped_early = True
            break

    if not stopped_early:
        do_eval()
    checkpoint()
    log.write(json.dumps({"event": "stop", "samples_seen": samples, "finished": not stopped_early,
                          "wall_minutes": (time.time() - t_start) / 60}) + "\n")
    log.close()
    print(("finished" if not stopped_early else "stopped at max_minutes; rerun the same command to resume")
          + f": best val AUROC {best['val_auroc']:.4f} @ {best.get('samples_seen')}", flush=True)
    return {"samples_seen": samples, "finished": not stopped_early, "best": best}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    train(load_config(args.config), device=args.device)


if __name__ == "__main__":
    main()
