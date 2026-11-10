# PLAN: 5-week project plan

**Research question:** *Which detector family generalises best to UNSEEN generators and survives real-world degradation (JPEG, resize, screenshot chains), and at what compute cost?*

- **Team:** Person A (models), Person B (data, evaluation, deployment/demo). About 8–12 h/week each, so **~100–120 person-hours in total**.
- **Compute:** free Kaggle/Colab T4 for training. Local NVIDIA T1000 8 GB is used for debugging, latency measurement and the demo only. An A40 48 GB may appear; it **only unlocks stretch goals**.
- **Companion docs:** [evaluation protocol](docs/evaluation_protocol.md) · [data sources](docs/data_sources.md) · [manifest schema](docs/manifest_schema.md) · [literature](docs/literature.md) · [splits template](configs/splits.example.yaml)

---

## 0. Feasibility: what is realistic and what we changed

| Original goal | Verdict | What we do instead |
|---|---|---|
| Scratch-ViT data scaling up to **500k** images | **Not realistic on Kaggle.** ~65 GB of lossless crops (estimate) plus ~30+ GPU-h for one run. Kaggle disk/output limits are UNVERIFIED but likely too small | Scale **10k → 50k → 200k** on Kaggle. **500k only if the A40 arrives** (decision D3) |
| Patch size 8 vs 16 at 224 px | p8@224 = 784 tokens, roughly 4–5× the cost of p16 on a T4 | **Equal-token ablation: p16@224 vs p8@112** (196 tokens each). This isolates patch granularity from compute. p8@224 is A40-only |
| Download Community Forensics | CF-Small is listed at ~260 GB, and the card warns of ~600 GB during re-indexing | **Stream** and write only the lossless crops we need |
| 5 tracks + video + demo in ~110 person-hours | **Aggressive.** It only works if we lean on existing libraries (timm, open_clip, peft, webdataset, official NPR/AEROBLADE code) and GPU runs happen unattended overnight | Hard cut list (§6). Video is the first to go |
| Video with a temporal model | Not enough time for a real temporal architecture | **Best image detector per frame + mean/max logit pooling.** A tiny temporal head only if ahead of schedule |
| AI-GenBench, full | Its reals need ImageNet (registration), LAION scraping and RAISE | **Fake part (HF) + COCO reals.** Use only the newest windows as test |
| Training on the T1000 | Turing without tensor cores, 8 GB, roughly GTX-1650 class | Never train there. Debug on tiny subsets, measure latency, run the demo |

Re-usable earlier code: `aidet/residuals.py` and `aidet/spectrum.py` (high-pass residuals and spectrum views) are deleted in the working tree but recoverable from commit `bb61be2` (`git show bb61be2:aidet/residuals.py`). Use them for the SRM-stem ablation and the demo's "what the detector sees" panel.

---

## 1. Week-by-week plan

Weekly rhythm: **Monday** 30-min sync (plan and blockers). **Thursday** async check-in. **Sunday** demo of the week's deliverables against the acceptance criteria. If a criterion fails, the cut list is applied the next Monday, not later.

### Week 1: Data, protocol, freeze (nothing is trained on real data until D1 passes)

**Person A: models infrastructure**
- [ ] Throughput/VRAM benchmark on a Kaggle T4 (fp16 AMP; **no bf16 on T4**) and on the local T1000:
  - ViT-Ti/16, ViT-S/16 and ViT-S/8@112 (timm)
  - ResNet-50
  - CLIP ViT-L/14 and DINOv3 ViT-L/16, in forward-only and LoRA modes
  - NPR
  - Record images/s, peak VRAM and the batch size that fits. Results go in `docs/throughput.md` and **replace the GPU-hour estimates in §3**.
- [ ] Shared training skeleton:
  - config → `experiments/<run>/` folder, WebDataset loader
  - online augmentation applied **identically to both classes**: random JPEG re-encode (p = 0.5, Q ~ U[60, 100]), Gaussian blur (p = 0.1)
  - fp16 AMP + GradScaler, checkpoint every 30 min (Kaggle sessions end at 12 h), resume support
- [ ] Accept the DINOv3 licence on HF (gated). Fallback: DINOv2.
- [ ] Qwen-Image feasibility on a T4 (CPU offload / quantised; one 1024² image). If it fails or takes more than 3 min per image, fall back to FLUX.1-schnell, after checking whether CF already contains it.
- [ ] Extract CLIP features of a 5k train/val sample for B's content-bias probe.

**Person B: data and evaluation**
- [ ] Day 1, access requests and checks:
  - email the Chameleon request
  - check whether the NTIRE 2026 data is public
  - read the GenImage LICENSE
  - read and archive the ToS of the 1–2 commercial generators (free web UI) under `docs/tos/`
  - update `docs/data_sources.md` (UNVERIFIED → verified / dropped)
- [ ] Streaming ingester:
  - CF-Small (HF streaming) and the GenImage subset → per-image metadata → manifest rows
  - 1–2 random **256×256 crops at 8-aligned offsets**, stored as lossless PNG in WebDataset shards (~1 GB each)
  - build the 10k and 50k sets first; 200k runs in the background
- [ ] Metadata per image: format, width/height, `jpeg_q` (from quantisation tables), EXIF presence, bytes/pixel, SHA-256, pHash, source, generator, architecture.
- [ ] Near-duplicate removal across train/val/test (pHash Hamming ≤ 6 and CLIP cosine ≥ 0.95; tune the thresholds on 100 hand-checked pairs). Violators are dropped from test.
- [ ] **Shortcut audit** (`docs/audit_v1.md`). On val, train:
  - (a) a gradient-boosted classifier on metadata only
  - (b) logistic regression on (width, height) only
  - (c) logistic regression on a 3×32-bin colour histogram
  - (d) a CLIP linear probe on *downsampled 64 px* images (content-bias proxy: semantics but few artefacts)
  - Report AUROC with bootstrap CIs.
- [ ] Fixed CF-Eval ID list (~10k, seed 0). AI-GenBench newest-window subset + COCO reals.
- [ ] Degraded test sets generated **once**, seeded: `jpeg95/75/50`, `down50`, `screenshot`. Stored in `data/degraded/`.
- [ ] Eval harness (`eval/`):
  - AUROC, AP, TPR@1%FPR, val-threshold metrics
  - stratified bootstrap CIs, seen / unseen-same-arch / unseen-new-arch tagging
  - unit tests on synthetic scores, checked against scikit-learn
- [ ] Copy `configs/splits.example.yaml` → `configs/splits.yaml`, fill it in, **freeze** (read-only, hash recorded). Freeze the evaluation protocol v1.0.

**Both:** take ~150 real photos each (no people in frame, varied scenes, phone camera, original files kept) and write captions for them, for the 2026 set.

**Deliverables:** `manifest_v<date>_1.parquet` + changelog, frozen `splits.yaml`, `docs/audit_v1.md`, degraded test sets, eval harness with passing tests, `docs/throughput.md`, a working training skeleton.

**Acceptance criteria:**
- Metadata-only, size-only and colour-only audit AUROC are all **≤ 0.6** on val, upper CI bound included. The 64-px CLIP content probe is reported, and if it is > 0.6 the content imbalance is discussed and mitigated by caption-matching where possible.
- `pytest tests/`: generator disjointness, parent/video integrity, held-out reals not in train, and metric correctness all pass.
- The 10k and 50k shards load at ≥ the throughput needed to keep a T4 busy, as measured.

**Dependencies:** training (week 2) is blocked on D1. The CLIP content probe needs A's features.

### Week 2: Baselines

**Person A**
- [ ] **T1:** CLIP ViT-L/14 and DINOv3 ViT-L/16 frozen features for train (10k/50k/200k) and all test sets including the degraded ones. Logistic-regression probe (C tuned on val). 3 seeds on the probe.
- [ ] **T4 NPR:** re-implement it. Run a parity test against the official repo's preprocessing on 20 images (same NPR map up to floating-point error). Train on 200k (or 50k if slow).
- [ ] **T5 sanity:** ViT-Ti/16 @224 crops, RGB, 10k. It must beat chance on val. If not, debug (lr, warmup, labels) before the ablations.
- [ ] Start the **T5 ablations** on ViT-Ti, 50k (§4).

**Person B**
- [ ] Launch **T3 ResNet-50 from scratch** from A's config (CNNDetection-style blur+JPEG aug, crops, 200k).
- [ ] Generate the 2026 set: Qwen-Image (or fallback) + 1–2 commercial UIs, ~300 images per generator from our captions. Store original bytes and record format. Add it to the manifest as `own_2026` / test.
- [ ] Evaluate every finished baseline on all test sources, clean and degraded. Write `experiments/<run>/metrics.json`.
- [ ] Draft the API (FastAPI, one `/score` endpoint per detector) using the CLIP probe.

**Decision D2 (end of week 2):** the LoRA backbone is whichever **probe has the higher val AUROC** on val generators, which are unseen in training. Tie → CLIP (cheaper licence situation, matches UnivFD). **Never decided on test.**

**Deliverables:** T1 (both probes), T3, T4 evaluated. First ViT-Ti numbers. 2026 set in the manifest.

**Acceptance:** each baseline has a `metrics.json` produced by the frozen harness on every available test source × degradation. Probes report 3-seed mean ± std.

### Week 3: Main models

**Person A**
- [ ] **T2 LoRA** on the D2 backbone:
  - r = 16, α = 32, on the attention q/v projections (+ MLP if val improves)
  - fp16, gradient checkpointing, lr sweep over 2–3 values on val
  - same online JPEG/blur augmentation
  - train at 200k, then rerun at **10k and 50k** for the equal-data comparison
- [ ] **T5:** finish the ablations on ViT-Ti 50k and pick the best config **by val**. Launch the ViT-S scaling runs at 10k / 50k / 200k with that config.

**Person B**
- [ ] **T6 AEROBLADE** (if not cut): inference-only with the official code / diffusers VAEs, on the test sets clean and degraded. No training.
- [ ] Video prep (if not cut): download the GenVideo-100K subset, or GenVidBench if needed:
  - ~1.5k real + 1.5k fake training clips and the official test generators
  - re-encode **both classes** to H.264 CRF 23, short side 512, 8 fps, 16 frames
  - run the video metadata audit with the same gate as images
- [ ] ONNX export dry run (CLIP probe), with an onnxruntime-CPU parity check (max |Δlogit| < 1e-3).
- [ ] Gradio demo v0: upload → score bars per detector.

**Decision D3 (end of week 3): has the A40 been granted?**
- **Yes:**
  - build 500k shards on the local disk and run ViT-S 500k
  - ViT-S p8@224
  - MAE pretraining stretch: ViT-S, unlabeled train crops, ~100–200 epochs-equivalent within 24–36 A40 hours, then fine-tune on 50k
  - extra seeds for LoRA / ViT-S 200k
- **No:** stop at 200k and report 500k as future work. No other plan item changes.

**Deliverables:** T2 evaluated at 10k/50k/200k. T5 ablation table complete. ViT-S scaling runs done or running.

**Acceptance:** the ablation table has one row per ablation, all with the same data, steps and seed, each with a val AUROC CI. Equal-data rows exist for the probe, LoRA and ViT-S at 10k and 50k.

### Week 4: Video, robustness, deployment

**Decision D4 (Monday):** video goes ahead **only if** every week 1–3 acceptance criterion is met. Otherwise video is cut and the time goes to analysis.

**Person A**
- [ ] **Video:** apply the best image detector per frame → mean- and max-pooled clip score. Evaluate on test generators split by video. Optional: a 1-layer temporal transformer on frozen per-frame features, if ahead.
- [ ] Finish the ViT-S 200k run (and the A40 runs if D3 = yes). Run extra seeds on the cheap tracks.

**Person B**
- [ ] Full robustness evaluation of all tracks × all degradations × all test sources.
- [ ] **Cost table:** params, training GPU-h (from logs), latency at batch 1 and 32 and peak VRAM on the T1000 (fp32 and fp16), CPU latency (ONNX Runtime for the best model, PyTorch for the rest).
- [ ] **Decision D5 (deploy model):** the best **unseen AUROC under `jpeg50`** among models with CPU-ONNX batch-1 latency under 1 s. Export that model to ONNX.
- [ ] Demo v1:
  - "real-world" slider (JPEG Q, downscale, screenshot chain) re-scores live
  - threshold line at the val 1%-FPR operating point
  - "evidence, not a verdict" note
  - optional residual/spectrum view from the recovered `aidet/` code

**Deliverables:** robustness and cost tables, video results (or a documented cut), ONNX model, demo v1.

**Acceptance:** the demo runs on a CPU-only laptop with no GPU, end to end, in under 60 s for one image including the slider. All tables are generated by scripts from `metrics.json`, never typed by hand.

### Week 5: Analysis, write-up, demo video, freeze

**Both**
- [ ] Failure analysis (protocol §7): detector × generator and detector × degradation heatmaps, format/size and real-source breakdowns, hardest FP/FN.
- [ ] Bootstrap CIs on every reported number. Mark single-run cells with `†`.
- [ ] Fill in the README tables (replace `[X]`) from scripts. Write the 5-bullet "findings" section, including negative results (e.g. scratch ViT needing more data than we have).
- [ ] Record a 60-second demo video. Update the README GIF/screenshot.
- [ ] **Reproducibility check:** a fresh Kaggle notebook runs the 10k pipeline end to end from the README instructions (manifest → train ViT-Ti 10k → eval).
- [ ] Code freeze on day 3. No new experiments after day 2, only reruns of broken ones. Tag `v1.0`.

**Deliverables:** final README with results, failure-analysis section, demo video, tagged release.

**Acceptance:** every number in the README traces to an `experiments/<run>/metrics.json` with a commit hash, manifest version and splits hash. The reproducibility notebook passes.

---

## 2. Dependencies (critical path)

```
W1 metadata+manifest ──► W1 shortcut audit (D1) ──► W2 all training
W1 throughput bench ───► GPU budget confirmed ───► W3 scaling runs
W1 degraded sets + harness ─► every evaluation
W2 probes (D2) ──► W3 LoRA ──► W4 deploy choice (D5) ──► W4 ONNX + demo
W2 ViT-Ti sanity ──► W2-3 ablations ──► W3 ViT-S scaling ──► W5 scaling plot
W3 video prep ──► W4 video (D4)
```
The critical path is **audit → ablations → ViT-S 200k**. The 200k shards must exist by the end of week 2.

## 3. Kaggle GPU-hours budget (estimates; replaced by measured throughput in week 1)

Assumption: Kaggle gives ~30 GPU-h/week/account. This is **UNVERIFIED** and comes from third-party reports; Kaggle documents a 12-h session cap. 2 accounts × 4 training weeks gives ~240 h in theory. **We plan for ~140 h (≈60%)** to absorb queueing, crashes and quota variance. Colab free is overflow only.

| Track | Runs | Est. T4 GPU-h | Notes |
|---|---|---|---|
| T1 probes (CLIP + DINOv3) | feature extraction for train 200k + all test × 6 variants; LR probes on CPU | 6 | forward-only, fp16 |
| T2 LoRA | 200k main + 10k + 50k + 2–3 lr points | 25 | ViT-L at 224, gradient checkpointing |
| T3 ResNet-50 scratch | 200k, 1 seed (+50k) | 15 | |
| T4 NPR | 200k (or 50k), 1–3 seeds | 6 | small network |
| **T5 scratch ViT** | see §4 | **50** | largest share, on purpose |
| T6 AEROBLADE | inference on test subsets × variants | 4 | VAE + LPIPS |
| Video | frame-feature extraction + pooling (+ tiny head) | 10 | |
| Evaluation (degraded sets, all tracks) | | 10 | |
| Reserve | crashes, reruns | 15 | |
| **Total** | | **~141** | |

Measure our own throughput, not paper numbers. If week-1 throughput comes in **more than 1.5× slower** than these imply, apply the cut list immediately (§6).

## 4. Track 5 (ViT from scratch): experiment matrix

**Recipe** (DeiT-style, forensics-safe):
- AdamW (β = 0.9/0.999, wd 0.05), lr 5e-4 × batch/512, 5-epoch-equivalent linear warmup, cosine decay
- label smoothing 0.1, stochastic depth 0.1 (Ti) / 0.1–0.2 (S), fp16 AMP, gradient clipping 1.0
- **Augmentation:** random crop at native resolution, horizontal flip, and the shared JPEG/blur augmentation on both classes
- **No** colour jitter, RandAugment colour ops, Mixup/CutMix or RandomResizedCrop: they destroy or fake forensic traces. Mixup/CutMix is an optional one-off ablation only.
- Sample budget scales with data: 10k → 1M samples seen, 50k → 2M, 200k → 3M (500k → 5M on the A40). Both "equal epochs" and "equal samples" views go in the report.

| ID | Model | Data | Variable | Settings compared | Est. GPU-h |
|---|---|---|---|---|---|
| S0 | ViT-Ti/16 | 10k | sanity | RGB, crop 224 | 0.5 |
| A1 | ViT-Ti | 50k | **patch size** | p16@224 vs **p8@112** (equal tokens) | 3 |
| A2 | ViT-Ti/16 | 50k | **input geometry** | native random crop (default) vs resize-to-224 | 1.5 (+1 run) |
| A3 | ViT-Ti/16 | 50k | **input stem** | RGB vs fixed SRM/high-pass residual stem (3 filters, frozen) vs *(optional)* Bayar constrained conv | 3 |
| A4 | ViT-Ti/16 | 50k | **position embedding** (cheap) | learned absolute vs fixed 2D sin-cos + GAP | 1.5 |
| R | ViT-Ti best | 50k | seeds | +2 seeds of the best config | 3 |
| SC1–3 | **ViT-S** best config | 10k / 50k / 200k | **data scaling** | | 1 + 4 + 15 |
| SC4 | ViT-S | 500k | scaling | **A40 only** | n/a (A40) |
| P8 | ViT-S/8@224 | 50k | patch at full FOV | **A40 only** | n/a (A40) |
| MAE | ViT-S | unlabeled crops → fine-tune 50k | SSL pretraining | **A40 only, stretch** | n/a (A40) |
| | | | | **Kaggle total** | **~33 (+eval/buffer → 50)** |

**Comparison at equal data:** the CLIP probe, the LoRA model and ViT-S are all evaluated at 10k and 50k (and 200k). The headline plot is "AUROC unseen vs training images" with one line per track.

Selection rule: the best config is chosen on **val AUROC (unseen val generators)**, then evaluated on test once.

## 5. Decision points

| ID | When | Question | If yes | If no |
|---|---|---|---|---|
| D1 | end W1 | Shortcut audit ≤ 0.6? | start training | fix the data (equalise JPEG for both classes, crop-only, re-balance sources, drop the worst-biased source, e.g. raw GenImage), re-audit. Training waits. |
| D1b | W1 | Qwen-Image feasible on a T4? | use it | FLUX.1-schnell (Apache-2.0) or another Apache/MIT model not in CF |
| D2 | end W2 | Which probe is better on val? | LoRA on that backbone | tie → CLIP |
| D3 | end W3 | **A40 granted?** | 500k scaling, p8@224, MAE stretch, extra seeds | stop at 200k; 500k listed as future work |
| D4 | start W4 | All W1–W3 acceptance met? | do video | cut video, use the time for analysis |
| D5 | mid W4 | Deploy model | best unseen `jpeg50` AUROC with CPU latency < 1 s | next best |

## 6. CUT LIST (drop in this order when behind)

1. **Video extension.** Fallback: score a few clips frame-by-frame in the demo only, no benchmark.
2. **AEROBLADE (T6).**
3. **A40 stretch goals** (500k, p8@224, MAE). These drop automatically if there is no A40.
4. Position-embedding ablation (A4).
5. Bayar learnable stem variant (keep the fixed SRM stem).
6. Extra seeds on expensive runs (report them as single-run `†`).
7. Optional test sources (Chameleon, NTIRE 2026).
8. DINOv3 probe (keep the CLIP probe; LoRA then goes on CLIP).
9. ResNet-50 reduced to 50k only.
10. NPR (T4).

**Never cut:**
- the scratch ViT at ≥ 50k with the patch-size, crop-vs-resize and RGB-vs-residual ablations, plus the 10k/50k/200k scaling
- the CLIP probe and LoRA
- the shortcut audit, frozen protocol, robustness evaluation, cost table and demo

## 7. Risk register

| ID | Risk | Likelihood | Impact | Mitigation | Owner | Early signal |
|---|---|---|---|---|---|---|
| R1 | **Kaggle quota / disk lower than assumed**, or HF streaming too slow, so the scaling study is starved | High | High | Week-1 throughput benchmark. Shards built once and attached as a Kaggle Dataset. 2 accounts. Cap at 200k. Cut list | B (data), A (compute) | W1 benchmark > 1.5× slower than §3 |
| R2 | **Shortcut / format bias** (e.g. GenImage JPEG-vs-PNG) inflates results | High | High | Audit gate D1. Identical JPEG handling for both classes. Crops not resizing. Format/size breakdown in the failure analysis | B | metadata-only AUROC > 0.6 |
| R3 | **Test sets unavailable** (NTIRE not released, Chameleon not answered, commercial ToS forbids research use) | Medium | Medium | Core test = CF-Eval + AI-GenBench newest + own 2026 set (open weights). Optional sources are never on the critical path | B | no Chameleon reply by end of W2 |
| R4 | "Unseen" is not really unseen: CF's thousands of SD derivatives make test generators near-duplicates of train | High | Medium | Architecture-level tagging (`unseen_same_arch` vs `unseen_new_arch`), canonical generator aliases, own 2026 set with new architectures | B | most "unseen" generators are `ldm-unet` |
| R5 | Scratch ViT stays near chance at 10k–50k | Medium | Medium | This is a legitimate finding and gets reported. Remedies: lower lr / longer warmup, SPT+LSA (arXiv 2112.13492), residual stem | A | S0 val AUROC ≈ 0.5 after the full budget |
| R6 | **Person-hours overrun** (~110 h for 5 tracks + video + demo) | High | High | Reuse libraries and official code. Unattended overnight runs. Weekly acceptance check triggers cuts on Monday | both | any W1 acceptance item slips |
| R7 | Licence breach (redistributing NC-SA images/shards, commercial-ToS outputs) | Low | High | Shards private. Repo ships code + manifests only. Licence column in the manifest. ToS archived | B | n/a |
| R8 | Kaggle 12-h session limit kills long runs | Medium | Medium | Checkpoint every 30 min, resumable loader state | A | n/a |
| R9 | DINOv3 gated access delayed / licence unclear | Low | Low | Request on day 1. Fallback DINOv2 (arXiv 2304.07193) | A | no access by W1 end |
| R10 | Qwen-Image (20B) too heavy for a T4 | Medium | Low | D1b fallback (FLUX.1-schnell, others) | A | > 3 min/image |
| R11 | Results not reproducible (library drift, seeds) | Medium | Medium | Pinned `requirements.txt`, seeds, commit + manifest + splits hash in every `metrics.json`, W5 fresh-notebook test | B | metric differs on rerun beyond the CI |
| R12 | A40 never arrives | High | Low | Nothing depends on it | n/a | n/a |
| R13 | A teammate unavailable for a week | Medium | Medium | Every task documented in `experiments/` and here. Each person can launch the other's configs | both | n/a |

## 8. Definition of done (portfolio)
- The README is understandable in 60 s and the results tables are fully filled from scripts with CIs and `†` marks.
- The demo runs on CPU and the slider visibly changes scores.
- There is an honest limitations and failure-analysis section, including negative results.
- Anyone can reproduce the 10k pipeline from the README on a free Kaggle notebook.
