# Which AI-image detectors survive the real world?

**A reproducible benchmark of 5 detector families (plus a training-free contrast) for AI-generated images, tested on generators they have never seen and on images that have been JPEG-compressed, resized and screenshotted. The best model is then extended to video.**

> Status: **planning phase**. Every number below is a placeholder `[X]` until measured. See [PLAN.md](PLAN.md).

---

## In 60 seconds

| | |
|---|---|
| **Problem** | Detectors that look near-perfect on their own benchmark often collapse on images from a *new* generator, or after one round of WhatsApp-style recompression. |
| **Research question** | *Which detector family generalises best to **unseen** generators, survives real-world degradation (JPEG, resize, screenshot chains), and at what compute cost?* |
| **Approach** | Train 5 detector families on the **same** data. Evaluate them under **one frozen protocol**. Splits are by *generator*, not by image, so the test generators are never seen in training and include newer ones. |
| **Why it is credible** | A shortcut audit runs before any training (file format, size and JPEG quality must not give away the label). Degraded test sets are fixed and seeded. Seen and unseen results are reported separately, with confidence intervals. Compute cost is reported for every model. |
| **Demo** | Upload an image or clip and see every detector's score side by side. Then drag a slider to recompress or "screenshot" it and watch the scores move. |

## Detector tracks

| # | Track | Idea | Why it is in the benchmark |
|---|---|---|---|
| 1 | **CLIP ViT-L/14 probe** (+ DINOv3 probe) | Frozen foundation-model features + a linear classifier (UnivFD-style) | Cheap, strong generalisation baseline |
| 2 | **LoRA fine-tuned backbone** | Low-rank adapters on the probe winner (CLIP or DINOv3) | Expected best; tests whether adapting beats freezing |
| 3 | **ResNet-50 from scratch** | Classic CNN with blur+JPEG augmentation (CNNDetection recipe) | Historical baseline |
| 4 | **NPR CNN** | Small CNN on neighbouring-pixel relations left by up-sampling layers | Paper re-implementation, artefact-focused |
| 5 | **ViT from scratch** | ViT-Tiny/Small with forensics-specific design (patch size, crops not resizing, noise-residual stem), plus a data-scaling study | How much data does a transformer need without pretraining? |
| 6 | *AEROBLADE (optional)* | Training-free: autoencoder reconstruction error of latent-diffusion VAEs | Contrast: no training at all |

## Results (placeholders: nothing has been measured yet)

Protocol: [docs/evaluation_protocol.md](docs/evaluation_protocol.md). *Seen* = generator family present in training. *Unseen* = never seen in training. All AUROC/AP/TPR values are means over the test sources, with 95% bootstrap CIs.

| Track | Params | Train GPU-h | AUROC seen | AUROC unseen | AP unseen | TPR@1%FPR unseen | AUROC unseen JPEG-50 | AUROC unseen screenshot | Latency T1000 (ms) | Latency CPU ONNX (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 CLIP probe | [X] | [X] | [X] | [X] | [X] | [X] | [X] | [X] | [X] | [X] |
| 1b DINOv3 probe | [X] | [X] | [X] | [X] | [X] | [X] | [X] | [X] | [X] | [X] |
| 2 LoRA | [X] | [X] | [X] | [X] | [X] | [X] | [X] | [X] | [X] | [X] |
| 3 ResNet-50 scratch | [X] | [X] | [X] | [X] | [X] | [X] | [X] | [X] | [X] | [X] |
| 4 NPR | [X] | [X] | [X] | [X] | [X] | [X] | [X] | [X] | [X] | [X] |
| 5 ViT-S scratch (best config) | [X] | [X] | [X] | [X] | [X] | [X] | [X] | [X] | [X] | [X] |
| 6 AEROBLADE | 0 trained | 0 | [X] | [X] | [X] | [X] | [X] | [X] | [X] | [X] |

**Scratch-ViT data scaling** (AUROC unseen, clean):

| Train images | ViT-S scratch | CLIP probe | LoRA |
|---|---|---|---|
| 10k | [X] | [X] | [X] |
| 50k | [X] | [X] | [X] |
| 200k | [X] | [X] | [X] |
| 500k (only if an A40 is available) | [X] | n/a | n/a |

**Video extension** (best image detector + temporal pooling): AUROC unseen video generators [X], per-clip latency [X].

Which numbers are single-run and which have several seeds will be stated next to each table.

## Demo (planned)

`app/` (Gradio UI) talks to `api/` (FastAPI):
1. Upload an image or a short clip.
2. See one score bar per detector, with a calibrated threshold line at the validation 1%-FPR operating point.
3. Use the **"Real-world" slider**: JPEG quality 95 → 50, downscale, or a screenshot chain. Scores update live, so you can watch which detector breaks first.
4. A per-image note explains that a score is **evidence, not a verdict**.

The best model ships as an ONNX Runtime export, so the demo runs on CPU.

## Repository structure

```
configs/            experiment configs + splits.yaml (generator-disjoint, read-only once frozen)
dataprep/           data library: metadata (JPEG-quality estimate, hashes), lossless crops, manifest validation
data/               (in the repo only manifests are used; large data lives under AIDET_DATA_ROOT, outside OneDrive)
  raw/              downloaded / streamed source data (git-ignored)
  manifests/        versioned Parquet manifests (tracked)
  shards/           WebDataset tar shards of lossless crops (git-ignored)
  degraded/         offline, seeded degraded test sets (git-ignored)
  own_2026/         our 2026 test set: generated + our own real photos (git-ignored)
detectors/
  clip_probe/       Track 1 (+ DINOv3 probe)
  lora_ft/          Track 2
  resnet50_scratch/ Track 3
  npr/              Track 4
  vit_scratch/      Track 5
  aeroblade/        Track 6 (optional)
  video/            frame sampling + temporal pooling
eval/               metrics, bootstrap CIs, degradation generation, cost/latency benchmarks
api/                FastAPI inference service (ONNX Runtime)
app/                Gradio demo
experiments/        one folder per run: config, manifest version, metrics JSON, logs
scripts/            data download / sharding / audit entry points
tests/              unit tests (splits disjointness, manifest schema, metric correctness)
docs/               protocol, manifest schema, data sources, literature
```

## How to reproduce (planned; commands will be filled in as they are implemented)

1. `pip install -r requirements.txt` (pinned versions)
2. Build the manifest and shards from public sources. See [docs/data_sources.md](docs/data_sources.md) for access and licences.
3. Run the shortcut audit. It must pass (metadata-only AUROC ≤ 0.6) before any training.
4. Generate the degraded test sets once, with fixed seeds.
5. Train each track from its config in `configs/` (Kaggle/Colab T4 notebooks are provided).
6. Run `eval/` to produce `experiments/<run>/metrics.json`, the tables above and the failure-analysis plots.

Every experiment records the manifest version and the hash of `configs/splits.yaml` it used.

## Limitations and ethics

- **Scores are evidence, not verdicts.** No detector here can prove an image is real or fake. Use the output as one signal among others: provenance, C2PA metadata, context.
- **No reliable detection on arbitrary internet content.** Our test sets are a sample of generators and degradations. A new generator or an unusual processing chain can break any detector, and the failure analysis shows where.
- **Compute and data limits.** These are student-scale budgets (free T4 GPUs). Some results are single-run, and these are labelled.
- **People and GDPR.** Public datasets used here contain faces (e.g. FFHQ). We do not redistribute them. Data we collected ourselves contains no people. Do not use these detectors to make decisions about individuals.
- **Licences.** Most datasets are **non-commercial** (e.g. CC BY-NC-SA 4.0, CC BY-NC 4.0, academic-only). Trained weights and shards inherit these restrictions: research use only.
- **Dual use.** Publishing detector weaknesses could help evasion. We report aggregate failure modes, not recipes for evading a specific detector.

## Citations

Full verified list with URLs: [docs/literature.md](docs/literature.md). Datasets: [docs/data_sources.md](docs/data_sources.md).

Key works: UnivFD (Ojha et al., CVPR 2023), DINOv3 (Siméoni et al., 2025), LoRA (Hu et al., 2021), CNNDetection (Wang et al., CVPR 2020), NPR (Tan et al., CVPR 2024), DeiT (Touvron et al., 2021), MAE (He et al., 2022), AEROBLADE (Ricker et al., CVPR 2024), Community Forensics (Park & Owens, CVPR 2025), GenImage (Zhu et al., NeurIPS 2023), AI-GenBench (Pellegrini et al., IJCNN 2025).

## Team

- **Person A**: models (Tracks 1–6, video model).
- **Person B**: data, evaluation, deployment and demo.
