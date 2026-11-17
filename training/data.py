"""Training/val data: map-style access to the WebDataset shards written by scripts/build_shards.py.

Why not stream the tars with webdataset's iterable pipeline:
  - the 10k/50k subsets are nested inside the 200k shards (size_rank < N/2 per class), so a
    streaming loader would read all 200k crops to train on 10k;
  - exact resume after a Kaggle session ends needs "skip to sample k", which a shuffled
    stream cannot do cheaply.
So we index each tar once (byte offset + size of every .png member) and read crops directly.
The shard files stay standard WebDataset tars.

Processing is identical for real and fake (CLAUDE.md rule 5): every random decision of the
augmentation comes from an RNG seeded by (seed, epoch, sample_id), never from the label.
Order: Gaussian blur -> JPEG re-encode (on the stored 256 crop) -> 8-aligned crop -> h-flip.
The crop offset is a multiple of 8, so a JPEG applied before the crop stays grid-aligned.
"""
from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, Sampler

from dataprep.crops import GRID
from dataprep.paths import DATA_ROOT, INDEX
from eval.degrade import image_rng

# ---------------------------------------------------------------- tar index


def index_tar(tar_path: Path, cache_dir: Path | None = INDEX / "tar_index") -> pd.DataFrame:
    """key -> (offset, size) of every .png member. Cached per tar (keyed by name + file size)."""
    tar_path = Path(tar_path)
    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / f"{tar_path.parent.name}__{tar_path.stem}__{tar_path.stat().st_size}.parquet"
        if cache.exists():
            return pd.read_parquet(cache)
    rows = []
    with tarfile.open(tar_path, "r:") as tar:
        for m in tar:
            if m.isfile() and m.name.endswith(".png"):
                rows.append((m.name[:-4], m.offset_data, m.size))
    df = pd.DataFrame(rows, columns=["key", "offset", "size"])
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    return df


def load_split(manifest: str | Path, split: str, n_images: int | None = None,
               shards_root: str | Path | None = None, cache_dir: Path | None = INDEX / "tar_index") -> pd.DataFrame:
    """Rows of one split, joined with tar offsets.

    n_images: total images (half per class) using the nested size_rank subsets;
    None = the whole split. Raises if the subset is not available in the manifest.
    """
    root = Path(shards_root) if shards_root else DATA_ROOT
    df = pd.read_parquet(manifest)
    df = df[df.split == split].copy()
    if df.empty:
        raise ValueError(f"no rows with split == {split!r} in {manifest}")
    if n_images is not None:
        per_class = n_images // 2
        df = df[df.size_rank < per_class]
        counts = df.label.value_counts()
        if counts.get(0, 0) < per_class or counts.get(1, 0) < per_class:
            raise ValueError(f"{split}: asked for {per_class}/class, manifest has {counts.to_dict()}")
    parts = df.path.str.split("#", n=1, expand=True)
    df["tar"], df["key"] = parts[0].map(lambda p: str(root / p)), parts[1]
    idx = pd.concat([index_tar(Path(t), cache_dir).assign(tar=t) for t in sorted(df.tar.unique())])
    df = df.merge(idx, on=["tar", "key"], how="left", validate="one_to_one")
    if df.offset.isna().any():
        raise FileNotFoundError(f"{int(df.offset.isna().sum())} manifest rows not found in their tar shard")
    df["offset"], df["size"] = df.offset.astype(np.int64), df["size"].astype(np.int64)
    return df.sort_values("sample_id").reset_index(drop=True)


# ---------------------------------------------------------------- transforms


def aligned_offset(extent: int, size: int, rng=None) -> int:
    """Random (rng given) or centred crop offset, rounded down to the 8-pixel grid."""
    if extent < size:
        raise ValueError(f"image side {extent} < crop {size}")
    n = (extent - size) // GRID
    return GRID * (rng.randint(0, n) if rng is not None else n // 2)


def augment(img: Image.Image, rng, aug: dict) -> Image.Image:
    """Shared blur + JPEG augmentation (CNNDetection-style). Label-independent by construction."""
    if rng.random() < aug.get("blur_p", 0.0):
        lo, hi = aug.get("blur_sigma", [0.1, 3.0])
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(lo, hi)))
    if rng.random() < aug.get("jpeg_p", 0.0):
        lo, hi = aug.get("jpeg_q", [60, 100])
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=rng.randint(lo, hi), subsampling=2)
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return img


def to_input(img: Image.Image, size: int, geometry: str, rng=None) -> Image.Image:
    """crop (default): native-resolution 8-aligned crop. resize: ablation A2 only."""
    if geometry == "resize":
        return img.resize((size, size), Image.Resampling.BICUBIC)
    if geometry != "crop":
        raise ValueError(f"geometry must be crop|resize, got {geometry!r}")
    x, y = aligned_offset(img.width, size, rng), aligned_offset(img.height, size, rng)
    return img.crop((x, y, x + size, y + size))


class CropDataset(Dataset):
    """Items are (index, epoch) pairs from StreamSampler (training) or plain indices (eval)."""

    def __init__(self, rows: pd.DataFrame, input_size: int, geometry: str = "crop", train: bool = False,
                 aug: dict | None = None, seed: int = 0, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.tar = rows.tar.to_numpy()
        self.offset, self.size = rows.offset.to_numpy(), rows["size"].to_numpy()
        self.label = rows.label.to_numpy().astype(np.int64)
        self.sample_id = rows.sample_id.to_numpy()
        self.input_size, self.geometry, self.train = input_size, geometry, train
        self.aug, self.seed = aug or {}, seed
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self._fh: dict[str, object] = {}

    def __getstate__(self):                      # file handles are per worker process
        return {**self.__dict__, "_fh": {}}

    def __len__(self) -> int:
        return len(self.label)

    def read_bytes(self, i: int) -> bytes:
        fh = self._fh.get(self.tar[i])
        if fh is None:
            fh = self._fh[self.tar[i]] = open(self.tar[i], "rb")
        fh.seek(self.offset[i])
        return fh.read(self.size[i])

    def image(self, i: int, epoch: int = 0) -> Image.Image:
        img = Image.open(io.BytesIO(self.read_bytes(i))).convert("RGB")
        if not self.train:
            return to_input(img, self.input_size, self.geometry)
        rng = image_rng(self.seed, f"{epoch}:{self.sample_id[i]}")
        img = augment(img, rng, self.aug)
        img = to_input(img, self.input_size, self.geometry, rng)
        if self.aug.get("hflip", True) and rng.random() < 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return img

    def __getitem__(self, item):
        i, epoch = item if isinstance(item, tuple) else (item, 0)
        arr = np.asarray(self.image(i, epoch), dtype=np.uint8)
        x = torch.from_numpy(arr.copy()).permute(2, 0, 1).float().div_(255)
        return (x - self.mean) / self.std, self.label[i]


class StreamSampler(Sampler):
    """Endless stream of (index, epoch): a fresh seeded permutation per epoch.

    Position k of the stream depends only on (seed, n, k), so resuming at k = samples_seen
    reproduces exactly the data order of an uninterrupted run.
    """

    def __init__(self, n: int, seed: int = 0, start: int = 0):
        self.n, self.seed, self.start = n, seed, start

    def perm(self, epoch: int) -> np.ndarray:
        return np.random.default_rng([self.seed, epoch]).permutation(self.n)

    def __iter__(self):
        epoch, pos = divmod(self.start, self.n)
        while True:
            p = self.perm(epoch)
            for i in p[pos:]:
                yield int(i), epoch
            epoch, pos = epoch + 1, 0
