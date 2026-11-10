# AI-generated image detection with ViTs

Course project following the study "AI-generated image detection with ViTs: project study"
(two-branch detector: CLIP ViT-B/16 on the RGB image + small ViT on a noise/frequency fingerprint,
trained on GenImage SD1.4, tested on 8 generators).

## Status: week 1, classic CV pieces (no dataset, no deep-learning framework needed)

Everything here runs with the system Python (3.14) plus `numpy` and `Pillow` only.

```
aidet/
  residuals.py    NPR residual (x - up(down(x))), 3 fixed SRM filters, blur-and-subtract
  spectrum.py     log-FFT spectrum, dataset average, radial profile, grid-peak score
  preprocess.py   224x224 random/centre crop (never resizes), JPEG re-encode, folder listing
  viz.py          spectra / residuals -> PNG panels without matplotlib
scripts/
  smoke_test.py            verifies all of the above on fabricated real/fake images
  make_spectrum_figure.py  average spectrum per folder + difference maps + radial CSV
  reencode_folder.py       re-encode a folder tree to JPEG at one quality (JPEG-vs-PNG fix)
figures/                   outputs of the scripts
demo_data/                 synthetic images written by the smoke test (safe to delete)
```

### Run

```
python scripts/smoke_test.py
python scripts/make_spectrum_figure.py --real <real_dir> --fake <fake_dir> --input gray --out figures/spectrum_gray.png
python scripts/make_spectrum_figure.py --real <real_dir> --fake <fake_dir> --input npr  --out figures/spectrum_npr.png
python scripts/reencode_folder.py --src <raw_dir> --dst <out_dir> --quality 95
```

### The week-1 test with your own images

1. Put 20+ photos from your phone in a folder, and 20+ images from any AI tool in another.
2. Run `make_spectrum_figure.py` on both folders with `--input gray` and `--input npr`.
3. Expect the fake panel to show bright dots / lines at the edges, corners or mid-axes that the
   real panel lacks, and a `grid-peak` score clearly above the real one. Note: phone photos are
   JPEG and downloaded AI images are usually PNG, so re-encode both with `reencode_folder.py`
   first, otherwise you are also looking at JPEG blocking artefacts.

### What the smoke test teaches

- A transposed-convolution ("checkerboard") up-sampler leaves a strong peak at the Nyquist
  corner of the spectrum: the FFT view catches it.
- A nearest-neighbour up-sampler leaves *no* spectral peak (its box filter is zero at Nyquist),
  but its NPR residual is exactly zero: the residual view catches it.
- That is why the detector gets both the residual and the spectrum as fingerprint inputs.

## Hardware notes for this machine

- GPU: RTX 5050 laptop, 8 GB. Use mixed precision, batch 32, keep CLIP frozen at first.
- Disk: ~42 GB free. Keep datasets *outside* OneDrive (e.g. `C:\data\`), download only the
  GenImage SD1.4 split + 2k/2k per other generator, re-encode to JPEG q95, delete originals.

## Next steps (weeks 2+)

1. Install the deep-learning stack: `pip install -r requirements.txt` (PyTorch with CUDA 12.8).
2. Download the GenImage subset, run `reencode_folder.py` on real and fake alike.
3. Dataset class + loaders using `preprocess.RandomCrop224` and `residuals.residual`.
4. Baselines: frozen CLIP + linear probe, ResNet-50, NPR.
