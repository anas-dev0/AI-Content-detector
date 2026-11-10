# Manifest schema (v1)

The manifest is the **single source of truth** for what data exists, which split it belongs to, and how it was processed. Every experiment config references **one manifest version** and **the SHA-256 of `configs/splits.yaml`**. A run that cannot name both is not reportable.

- Format: **Parquet** (pyarrow), one row per stored sample: an image crop, a full test image, or a video frame/clip.
- Location: `data/manifests/manifest_v<YYYYMMDD>_<n>.parquet` + `manifest_v<YYYYMMDD>_<n>.md` (changelog: what was added/removed and why, row counts per split × label × source).
- Versions are **immutable**. Changing anything means writing a new version. Old versions are never edited.
- Manifests are committed to git (small). Images/shards are not.

## Required columns

| Column | Type | Allowed values / format | Notes |
|---|---|---|---|
| `path` | string | relative URI, e.g. `shards/train-000042.tar#cf_000123_c0` or `degraded/jpeg50/own_2026/img_0001.jpg` | Locates the bytes |
| `label` | int8 | `0` = real, `1` = fake | Same convention as Community Forensics |
| `generator` | string | canonical lowercase name, e.g. `sd-1.4`, `midjourney-v5`, `qwen-image`; `real` for real images | Canonical names are mapped in `configs/generator_aliases.yaml` so the same model is never counted under two names |
| `architecture` | string | `gan`, `ldm-unet`, `ldm-dit`, `pixel-diffusion`, `autoregressive`, `unknown`, `real` | Lets us report "unseen same-architecture" vs "unseen new-architecture" |
| `source_dataset` | string | `cf_small`, `cf_eval`, `genimage`, `aigenbench`, `ntire26`, `chameleon`, `own_2026`, `genvideo`, `genvidbench` | |
| `split` | string | `train`, `val`, `test` | Must agree with `configs/splits.yaml` (enforced by test) |
| `format` | string | `jpeg`, `png`, `webp`, `other`: the **original** file format before our processing | Shortcut-audit feature |
| `width` | int32 | original width in px | Before cropping |
| `height` | int32 | original height in px | Before cropping |
| `jpeg_q` | float32 / null | estimated JPEG quality 1–100 from the quantisation tables; null if not JPEG | Shortcut-audit feature |
| `hash` | string | SHA-256 of the stored bytes | Exact-duplicate detection, integrity |

## Recommended extra columns

| Column | Type | Notes |
|---|---|---|
| `sample_id` | string | stable unique key, e.g. `cf_small/<image_name>` (+ `_c0`, `_c1` for crops) |
| `parent_id` | string | original image ID (several crops share one parent; splits are assigned per parent) |
| `real_source` | string | for reals: `coco`, `ffhq`, `vision`, `lhq`, `imagenet`, `laion`, `raise`, `own_phone`, … |
| `phash` | string (16 hex) | 64-bit perceptual hash of the full image, for near-duplicate removal |
| `clip_emb_id` | int64 / null | row index into a separate embedding file (`.npy`), for embedding-based dedup and content-bias checks |
| `crop_xy` | string / null | `"x,y,w,h"` in original pixels; x and y are multiples of 8 |
| `exif_present` | bool | any EXIF block in the original |
| `bytes_per_pixel` | float32 | original file size / (width × height) |
| `file_bytes` | int64 | original file size |
| `mode` | string | `RGB`, `RGBA`, `L`, … (converted to RGB on read; recorded for the audit) |
| `degradation` | string | `clean`, `jpeg95`, `jpeg75`, `jpeg50`, `down50`, `screenshot`, `chain_social` |
| `degradation_seed` | int32 / null | seed used for the degraded copy |
| `prompt` | string / null | generation prompt or caption, if known |
| `license` | string | SPDX-style identifier of the source, e.g. `CC-BY-NC-SA-4.0` |
| `release_date` | date / null | generator release date (for temporal splits); only from cited sources |
| `video_id` | string / null | video datasets: split key, so all frames of a video stay in one split |
| `frame_idx` | int32 / null | frame index within the clip after fps resampling |
| `nsfw_flag` | bool / null | passed through from CF; NSFW samples are excluded from demo examples |
| `shard` | string | WebDataset shard name |

## Invariants (enforced by `tests/test_manifest.py`)

1. `hash` is unique within the manifest. Exact duplicates are removed before splitting.
2. **Generator disjointness:** the set of fake `generator` values in `train`, `val` and `test` are pairwise disjoint, except generators explicitly listed as `seen_in_test` in `configs/splits.yaml` (used to measure seen-generator performance).
3. **Parent integrity:** all rows with the same `parent_id` (and all frames with the same `video_id`) have the same `split`.
4. **Near-duplicates:** no `test` row has pHash Hamming distance ≤ 6 (threshold to tune in week 1) to any `train` row, and no CLIP cosine similarity ≥ 0.95 to any `train` row. Violators are dropped from **test**, not train.
5. **Held-out real sources** (`raise`, `own_phone`) never appear with `split == train`.
6. `label == 0` ⇔ `generator == "real"`.
7. `jpeg_q` is null iff `format != "jpeg"`.
8. Class balance per split is reported, not forced. The audit flags any split with real:fake outside 40:60–60:40.

## Experiment config reference (example)

```yaml
# configs/exp/vit_s16_crop224_rgb_50k.yaml
manifest: data/manifests/manifest_v20261120_1.parquet
splits_sha256: "<sha256 of configs/splits.yaml>"
train_subset: {max_images: 50000, seed: 0, stratify_by: [generator, label]}
```
