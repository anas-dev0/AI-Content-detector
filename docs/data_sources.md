# Data sources: access, licence, size, role

Facts were checked by web search on **2026-11-10**. Anything that could not be confirmed from the primary page is marked **UNVERIFIED**. Before downloading, re-check the licence on the dataset card. Cards change.

**Role legend:** `TRAIN` = may appear in train/val. `TEST` = test only, never trained on. `HELD-OUT` = never in training even as reals.

## Image datasets

### Community Forensics (CF): `TRAIN` (Small) / `TEST` (Eval)
- Paper: Park & Owens, *Community Forensics: Using Thousands of Generators to Train Fake Image Detectors*, CVPR 2025. https://arxiv.org/abs/2411.04125
- **CF-Small**: https://huggingface.co/datasets/OwensLab/CommunityForensics-Small
  - The card states 278K generated + 278K real images, but the dataset viewer estimates ~356K rows. The discrepancy is unexplained; count ourselves in week 1.
  - Listed size **260 GB**. The card warns downloads "may use up to ~600GB".
  - Generators come from 4,803 models, the vast majority Stable Diffusion derivatives (the authors note this bias). Subsets: systematic, manual, commercial.
  - Real images come from FFHQ, VISION, COCO and Landscapes HQ.
  - Fields: `image_name, format, resolution, mode, image_data, model_name, nsfw_flag, prompt, real_source, subset, split, label (1 = fake), architecture`.
  - **Streaming supported.**
  - Licence: **CC BY-NC-SA 4.0** (Small card). The full-dataset card has been reported as CC BY 4.0 (**UNVERIFIED**). We treat everything as **CC BY-NC-SA 4.0**, so derived shards stay private.
- **CF-Eval**: https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval
  - Split `CompEval`, 51,836 images, ~206 GB, reals from **RAISE**, COCO, FFHQ, LAION. CC BY-NC-SA 4.0.
  - We stream a **fixed ~10k ID list** once.
- Plan: never download CF-Small whole. Stream it, filter it by generator split, and write lossless crops to shards (see Storage below).

### GenImage: `TRAIN` (subset) + `VAL` (Midjourney)
- Paper: Zhu et al., *GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image*, NeurIPS 2023 Datasets & Benchmarks. https://arxiv.org/abs/2306.08571
- Official repo: https://github.com/GenImage-Dataset/GenImage
  - Downloads: Baidu Netdisk, plus an **official Google Drive folder** added 2024-03-15. The Drive link is on the repo README, so we don't need Baidu.
- Unofficial HF mirrors (convenient but **not official**, so verify file counts against the paper): https://huggingface.co/datasets/bitmind/GenImage_ADM (and the sibling `bitmind/GenImage_*` repos), https://huggingface.co/datasets/nebula/GenImage-arrow
- Generators: Midjourney, SD v1.4, SD v1.5, ADM, GLIDE, Wukong, VQDM, BigGAN. Reals: ImageNet (1,000 classes).
- Licence: the repo has a LICENSE file whose terms we have not read, so **UNVERIFIED**. Assume research-only.
- **Known bias**: reals are ImageNet JPEGs, while most fakes are PNG, and image sizes differ. Detectors learn "JPEG vs PNG" (Grommelt et al., https://arxiv.org/abs/2403.17608). Our shortcut audit must catch this. Mitigation: identical JPEG re-encoding for both classes, and crops instead of resizing.

### AI-GenBench: `TEST` (newest windows only)
- Paper: Pellegrini et al., *AI-GenBench: A New Ongoing Benchmark for AI-Generated Image Detection*, IJCNN 2025. https://arxiv.org/abs/2504.20865
- Code: https://github.com/MI-BioLab/AI-GenBench (BSD-3 code). Dataset build instructions: https://github.com/MI-BioLab/AI-GenBench/blob/main/dataset_creation/README.md
- 36 generators (2017–2024) ordered chronologically. The fake part is pre-packaged at https://huggingface.co/datasets/lrzpellegrini/AI-GenBench-fake_part
- Reals require LAION-400M (ELSA_D3 subset), COCO 2017, ImageNet and RAISE. **Licences are mixed, per source.**
- Use: only the **newest windows** as a temporal "future generators" test. Per-generator release dates come from the paper/repo, not from us.
- Cheaper variant: use only the fake part + COCO reals. We skip ImageNet (registration needed) and RAISE (held out anyway).

### NTIRE 2026 Robust AIGI Detection in the Wild: `TEST` (if released)
- Overview: Gushchin et al., CVPRW 2026. https://arxiv.org/abs/2604.11487. Challenge page: https://cvlai.net/ntire/2026/
- 108,750 real + 185,750 generated images, 42 generators, 36 transformations (crop, resize, compression, blur). Test generators include recent proprietary models.
- **Public release after the challenge: UNVERIFIED.** The paper gives no dataset URL. Week-1 task: check the challenge page / Codabench. If unavailable, drop it. No plan item depends on it.

### Chameleon (AIDE): `TEST` (if access granted)
- Paper: Yan et al., *A Sanity Check for AI-generated Image Detection*, ICLR 2025. https://arxiv.org/abs/2406.19435. Repo: https://github.com/shilinyan99/AIDE
- Access: **by email request** (address in the repo README). "Only used for academic research. Commercial use in any form is prohibited."
- Week-1 task: send the request on day 1. Size: **UNVERIFIED**.

### Our own "2026 set": `TEST`
- **Fakes**: ~300 images per generator, 2–3 generators.
  - **Qwen-Image**: Apache-2.0, 20B params, weights released 2025-08-04, https://huggingface.co/Qwen/Qwen-Image. Run on a Kaggle T4 with CPU offload/quantisation. Feasibility is **UNVERIFIED** and gets tested in week 1.
  - Fallback: **FLUX.1-schnell** (Apache-2.0, https://huggingface.co/black-forest-labs/FLUX.1-schnell). It is likely already "seen" via CF/NTIRE, so it is less useful as "unseen". Check CF's `model_name` list.
  - Plus **1–2 commercial generators through their free web UI** (e.g. Gemini or ChatGPT image generation), **only if their current terms allow research use of outputs**. Read and archive the ToS text in `docs/` before generating.
- **Reals**: our own phone photos. **No people in frame.** EXIF is kept in the raw copy and stripped in the shared copy.
- **Caption matching**: write a caption for each real photo, then generate one fake per caption per generator. Both classes go through the same storage path: original bytes kept, format recorded.
- **Our real photos are test-only.** They are never used in training.

## Video datasets (week 4 only)

### GenVideo (DeMamba): primary candidate
- Paper: *DeMamba: AI-Generated Video Detection on Million-Scale GenVideo Benchmark*, arXiv 2405.19707. https://arxiv.org/abs/2405.19707. Repo: https://github.com/chenhaoxing/DeMamba
- Generated videos are on ModelScope. **GenVideo-100K** (10k samples per forgery category): https://modelscope.cn/datasets/cccnju/GenVideo-100K
- Test reals are MSR-VTT, inside GenVideo-Val. Training reals are Kinetics-400 / Youku-mPLUG, downloaded separately.
- The README splits train and test **by generator** (e.g. test: ModelScope, MorphStudio, MoonValley, HotShot, Show_1, Gen2, Crafter, Lavie, Sora, WildScrape).
- Licence: **not stated (UNVERIFIED)**. Assume research-only.

### GenVidBench: alternative
- Paper: Ni et al., AAAI 2026. https://arxiv.org/abs/2501.11340. Site: https://genvidbench.github.io/. HF: https://huggingface.co/datasets/jian-0/GenVidBench
- 6.78M videos, 11 generators, cross-source and cross-generator splits, prompt-paired real/fake. **CC BY-NC 4.0**.
- Caveat: the "pair 1" videos must be fetched from Vript / HD-VG-130M / VidProM by ID, which is extra work. Use only if GenVideo access fails.

Video budget: **a few thousand clips** (e.g. 1,500 real + 1,500 fake for training, plus the official test generators), **split by video ID**.

## Storage budget

| Item | Size | Where |
|---|---|---|
| CF-Small, full | ~260 GB (card) | **never downloaded whole**, streamed |
| CF-Eval, full | ~206 GB (card) | stream a fixed ~10k subset only |
| Lossless training crops, 256×256 PNG | **estimate** 100–150 KB/crop (measure in week 1) | WebDataset shards |
| → 10k / 50k / 200k / 500k crops | ≈ 1.3 / 6.5 / 26 / 65 GB (estimates) | Kaggle Dataset or HF private repo |
| Test sets (clean, native resolution, ~25k images) | ≈ 10–20 GB (estimate) | Parquet/WebDataset |
| Degraded variants (5 × test) | JPEG/PNG; JPEG variants much smaller than clean | generated once |
| Video: 3k–5k clips × 16 frames | ≈ 5–15 GB (estimate) | frames as PNG or clip as H.264 |

Kaggle limits (**UNVERIFIED**, third-party sources): ~20 GB persistent `/kaggle/working` output, larger temporary disk, attached Kaggle Datasets mounted read-only. Kaggle Dataset size limits and HF private-storage quotas also need checking in week 1. **Consequence: the 500k scale is only realistic with the A40 (local disk) or by streaming. On Kaggle we stop at 200k.**

### Format
- **Training: WebDataset `.tar` shards (~1 GB each)**, one sample = `{key}.png` + `{key}.json` (manifest row). Sequential reads stream fast on Kaggle's limited CPUs (4 vCPU).
- **Test sets and manifests: Parquet** (`image_bytes` column + metadata), easy to filter and audit.

### What is lossless and what is not
| Lossless (allowed for preprocessing) | Lossy (augmentation/degradation only, applied to **both** classes) |
|---|---|
| Cropping on the pixel grid. Crop offsets are multiples of 8 so JPEG block alignment is preserved | JPEG re-encoding (any quality) |
| PNG, lossless WebP | Resizing / resampling (any kernel) |
| Keeping the original JPEG bytes unchanged | Lossy WebP / AVIF |
| Colour-space *metadata* changes without pixel changes | Screenshot chains, social-media chains |

Rule: for a source image that is already JPEG, the stored crop is the decoded JPEG pixels, which is lossless relative to the decode. The original `jpeg_q` estimate is kept in the manifest so the audit can check quality is not a label shortcut.
