# Shortcut audit: raw_cf_small

Input: `cf_small_index.parquet`, rows used: 200000 (real 100000, fake 100000).
Split grouped by generator, 30% held out. Gate: AUROC upper 95% CI <= 0.6.

| Baseline | Role | AUROC [95% CI] | <= 0.6 | Features |
|---|---|---|---|---|
| model_visible | **gate** | 0.838 [0.836, 0.841] | NO | jpeg_history, mp_bucket, mode_not_rgb |
| metadata_all | info | 0.997 [0.996, 0.997] | NO | is_jpeg, width, height, log_mp, aspect, is_square, w_mod64, h_mod64, mode_not_rgb, jpeg_history, mp_bucket |
| size_only | info | 0.997 [0.996, 0.997] | NO | width, height, log_mp, aspect, is_square, w_mod64, h_mod64 |
| format_only | info | 0.727 [0.724, 0.730] | NO | is_jpeg |

Gate = cues observable in a 256x256 crop (JPEG history, coarse scale, colour mode, JPEG quality). Informational rows describe the original files; exact size/aspect do not survive cropping. Pixel-level baselines (colour histogram, 64-px CLIP content probe) run on the built shards.

**Verdict: FAIL: fix the data before training**
