# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project
A benchmark of detector families for AI-generated images, later extended to video. Research question: **which detector family generalises best to UNSEEN generators and survives real-world degradation (JPEG, resize, screenshot chains), and at what compute cost?**

- 5-week student project, 2 people: Person A (models), Person B (data, eval, demo).
- Also a portfolio piece, so README clarity matters.
- **Status: planning phase.** Docs and skeleton only; no model code yet. The current week's tasks are in `PLAN.md`.

## Read these first
- `PLAN.md`: week-by-week tasks, GPU budget, decision points (D1–D5), cut list, risk register.
- `docs/evaluation_protocol.md`: the **frozen** protocol (metrics, test sources, degradations, CIs).
- `docs/manifest_schema.md`: manifest columns and invariants.
- `docs/data_sources.md`: dataset access, licences, storage plan, lossless vs lossy rules.
- `docs/literature.md`: verified citations per track.
- `configs/splits.example.yaml`: template for the generator-disjoint split file.

## Hard rules (do not break these)
1. **Never fabricate metrics, results or citations.**
   - Results stay `[X]` until produced by `eval/` and stored in `experiments/<run>/metrics.json`.
   - Unverifiable facts are tagged `UNVERIFIED`.
   - Paper numbers are never presented as our results.
2. **Split by generator, never by random image.** Video is split by `video_id`. Once `configs/splits.yaml` is frozen, do not edit it: changing it invalidates every result.
3. **The evaluation protocol is frozen after week 1.** Any change needs a change-log entry in `docs/evaluation_protocol.md`, and every affected track must be re-evaluated.
4. **No test-set peeking.** Model selection, thresholds and early stopping use `val` only.
5. **Identical processing for real and fake.**
   - No resizing by default: use native-resolution crops at 8-pixel-aligned offsets. Resize is an ablation only.
   - JPEG/blur augmentation is applied the same way to both classes.
   - Degraded test sets are generated once, offline, with fixed seeds. Nothing is degraded randomly at test time.
6. **No training until the shortcut audit passes:** metadata-only, size-only and colour-histogram AUROC must all be ≤ 0.6 on val.
7. **Every experiment records** the manifest version, the SHA-256 of `configs/splits.yaml` and the git commit. Runs without them are not reportable.
8. **Data and licences.**
   - Never commit images, shards or checkpoints. Most data is non-commercial (CC BY-NC-SA 4.0 etc.).
   - Our own photos and the 2026 test set contain no people. Our own real photos are test-only.
   - RAISE never enters training.

## Environment
- **Python venv** lives outside OneDrive at `C:\Users\anasw\.venvs\aidet` (run `C:/Users/anasw/.venvs/aidet/Scripts/python.exe`).
- **Large data** lives outside the repo in `AIDET_DATA_ROOT` (default `C:\aidet_data`; on Kaggle `/kaggle/working/aidet_data`). The repo is in a OneDrive-synced folder, so never write datasets, shards or checkpoints inside it.
- **Local GPU on this machine:** GTX 1660 Ti 6 GB.
- **Run tests:** `C:/Users/anasw/.venvs/aidet/Scripts/python.exe -m pytest tests -q -p no:cacheprovider`

## Data pipeline (week 1)
1. `scripts/index_hf_parquet.py`: metadata-only index of a HF Parquet dataset. Reads columns, skips `image_data`.
2. `scripts/shortcut_audit.py`: metadata classifiers grouped by generator; the gate is `model_visible` ≤ 0.6.
3. `scripts/build_pool.py`: scale-matched, JPEG-equalised, generator-split pool with nested `size_rank`.
4. `scripts/build_shards.py`: one pass over the source files → lossless crop shards + manifest part.
5. `scripts/benchmark_throughput.py`: images/s and VRAM per architecture on the current GPU → `docs/throughput.md`.

## Compute constraints
- **Training** runs on free Kaggle/Colab T4 GPUs. Use fp16 AMP + GradScaler: the T4 has no bf16. Kaggle sessions end at 12 h, so checkpoint every ~30 min and make runs resumable.
- **Local NVIDIA T1000 8 GB** (no tensor cores): debugging on tiny subsets, latency/VRAM measurement and the demo only. Never plan training on it.
- **A40 48 GB**: maybe available. It unlocks stretch goals only (500k scaling, ViT p8@224, MAE). Nothing may depend on it.
- **Community Forensics-Small** is ~260 GB. Stream it and write lossless 256×256 PNG crops into WebDataset shards. Never download it whole.

## Detector tracks → folders
| Track | Folder |
|---|---|
| 1 CLIP ViT-L/14 probe (+ DINOv3 probe) | `detectors/clip_probe/` |
| 2 LoRA on the probe winner | `detectors/lora_ft/` |
| 3 ResNet-50 from scratch, blur+JPEG aug | `detectors/resnet50_scratch/` |
| 4 NPR re-implementation | `detectors/npr/` |
| 5 ViT from scratch (first-class, never cut) | `detectors/vit_scratch/` |
| 6 AEROBLADE (optional, training-free) | `detectors/aeroblade/` |
| Video (frame-level + pooling) | `detectors/video/` |

Other folders:
- `dataprep/`: data library (`paths.py`, `metadata.py`, `crops.py`, `manifest.py`)
- `eval/`: metrics, bootstrap CIs, degradations, cost benchmarks
- `api/`: FastAPI + ONNX Runtime
- `app/`: Gradio demo
- `experiments/`: one folder per run
- `scripts/`: data pipeline entry points
- `tests/`: split disjointness, manifest invariants, metric correctness
- `data/`: raw / shards / degraded / own_2026 are git-ignored; `manifests/` is tracked

## Conventions
- **Stack:** Python, PyTorch, timm, open_clip, peft, webdataset, pyarrow/Parquet, scikit-learn, onnxruntime, FastAPI, Gradio. Prefer existing libraries and official reference code over custom implementations; person-hours are the tightest budget.
- **Configs:** YAML in `configs/`. Experiment configs reference a manifest path and `splits_sha256`.
- **Manifests:** immutable, versioned `data/manifests/manifest_v<YYYYMMDD>_<n>.parquet`, each with a changelog `.md`.
- **Generator names:** canonical lowercase (`sd-1.4`, `qwen-image`, `real`), mapped via `configs/generator_aliases.yaml`.
- **Labels:** `0` = real, `1` = fake.
- **Results tables** are generated by scripts from `metrics.json`, never typed by hand. Single-run cells are marked `†`.
- **Tests:** `pytest tests/` must pass before freezing splits or reporting results.
- **Platform:** Windows dev machine (PowerShell or Git Bash); Kaggle notebooks run Linux. Keep paths portable (`pathlib`).

## History
Earlier exploratory code (`aidet/residuals.py`, `aidet/spectrum.py`: high-pass residuals and spectrum views) was removed from the working tree. It can be recovered with `git show bb61be2:aidet/<file>` for the SRM-stem ablation or the demo's residual view.
