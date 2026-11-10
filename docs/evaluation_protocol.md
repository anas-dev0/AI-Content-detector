# Evaluation protocol (v1.0)

> **Frozen at the end of week 1.** After the freeze, the protocol does not change. Any unavoidable change goes in the change log at the bottom, with a date and reason, and every affected track is re-evaluated. Detectors are compared only on results produced under the same protocol version.

## 1. Test sources (never used for training or model selection)

| ID | Source | Subset we use | Seen/unseen |
|---|---|---|---|
| `cf_eval` | Community Forensics-Eval (`CompEval`) | Fixed list of ~10k image IDs, streamed once, frozen in `data/manifests/` | Per generator, computed against the train list |
| `aigb_new` | AI-GenBench | The newest chronological windows (generators released after our newest training generator) | Unseen by construction |
| `own_2026` | Our 2026 set | ~300 fake per generator × 2–3 generators, plus caption-matched real photos taken by us, no people in frame | Unseen |
| `ntire26` | NTIRE 2026 robust AIGI data | Only if publicly released (**UNVERIFIED**) | Per generator |
| `chameleon` | Chameleon (AIDE) | Only if access is granted by email | Unseen |
| `video_test` | GenVideo / GenVidBench test generators | Split by video, test generators disjoint from train | Unseen |

**Validation** (model selection, threshold calibration, early stopping) uses only the `val` generators listed in `configs/splits.yaml`.

**Seen vs unseen** is not assumed. `eval/` intersects each test image's `generator` (and `architecture`) with the training generator list in the frozen manifest, and tags the image `seen`, `unseen_same_arch` or `unseen_new_arch`.

## 2. Metrics

Each metric is reported **per test source, per generator, and aggregated separately over seen and unseen generators**:

- **AUROC**: threshold-free ranking quality.
- **Average precision (AP)**.
- **TPR at 1% FPR**: the operationally relevant number (how many fakes we catch while flagging only 1% of real images). The threshold is computed **on the test set** for this metric. We also report:
- **Accuracy and TPR/FPR at the validation threshold**: threshold fixed on `val` at 1% FPR and then applied unchanged to test. This exposes calibration drift under distribution shift.

Aggregation rules:
- Mean over generators is unweighted, so large generators do not dominate. Pooled-sample AUROC is reported as a secondary number.
- Real images are shared between generators within a source. Per-generator AUROC = that generator's fakes vs all real images of that source.

## 3. Robustness: degraded test sets

Degraded variants are **generated once, offline, with fixed seeds**, stored in `data/degraded/<variant>/` and recorded in the manifest (`degradation` column). Nothing is degraded randomly at test time. The same operation is applied to real and fake images.

| Variant | Operation (Pillow / OpenCV, versions pinned) |
|---|---|
| `clean` | none (original bytes) |
| `jpeg95`, `jpeg75`, `jpeg50` | decode → re-encode JPEG at quality 95 / 75 / 50, 4:2:0 chroma subsampling |
| `down50` | bicubic downscale ×0.5 (antialiased) → PNG |
| `screenshot` | chain: resize to a display scale drawn per image from {0.75, 1.0, 1.25} (seeded) → PNG → crop a random UI margin of 0–5% per side (seeded) → JPEG 85. Approximates "screenshot then share" |
| `chain_social` *(optional)* | down50 → JPEG 75 → JPEG 75 (double compression) |

Robustness summary per detector: AUROC drop from `clean` to each variant, on unseen generators.

## 4. Input handling at test time (identical for every detector)

- **No resizing by default.** Detectors with a fixed input size take a **centre crop** (or the mean of a 5-crop) at native resolution. Images smaller than the crop are reflect-padded. Resize-based inference is evaluated only as an ablation.
- CLIP/DINOv3 tracks use the backbone's own normalisation. The crop size follows the patch grid (224 for /14 and /16 backbones).

## 5. Compute cost (reported for every track)

| Metric | How |
|---|---|
| Parameters | total / trainable |
| Training GPU-hours | wall-clock × GPUs, with the GPU type stated (T4, or A40 if available) |
| Inference latency | batch 1 and batch 32, median of 200 runs after 20 warm-up, at 224×224 input. **NVIDIA T1000 8 GB** (fp32 and fp16) and **CPU** (stating the model; ONNX Runtime for the best model, PyTorch for others) |
| Peak VRAM | `torch.cuda.max_memory_allocated` at batch 1 and 32 on the T1000 |
| Throughput | images/s |

## 6. Statistical honesty

- **95% bootstrap CIs** (1,000 resamples, stratified by class, seed 0) for AUROC, AP and TPR@1%FPR on every test source.
- **Seeds**: cheap tracks (probes, NPR, ViT-Tiny 10k) run with **3 seeds**, reported as mean ± std. Expensive runs (LoRA, ViT-S 200k, ResNet-50) are **single-run unless budget allows**, and every table marks single-run cells with `†`.
- No test-set peeking. The checkpoint and hyperparameters are chosen on `val` only. Test evaluation happens once per final checkpoint and is logged in `experiments/<run>/metrics.json` with the git commit, manifest version and splits hash.
- Differences smaller than the CI overlap are not claimed as wins.

## 7. Failure analysis (required section of the final report)

- A heatmap of detector × generator AUROC on unseen generators, clean and `jpeg50`.
- A heatmap of detector × degradation.
- Breakdown by real source and by image format/size bucket (catches residual shortcuts).
- 20 hardest false positives and false negatives per detector, shown as thumbnails where the licence allows, otherwise by ID.
- Short written conclusions: which generator families and degradations break which detector family, and a hypothesis why.

## Change log

| Date | Version | Change | Reason | Re-evaluated tracks |
|---|---|---|---|---|
| (week 1 freeze) | 1.0 | initial | n/a | n/a |
