"""Re-encode every image in a folder tree as JPEG at one fixed quality (kills the JPEG-vs-PNG shortcut).

    python scripts/reencode_folder.py --src C:/data/genimage_raw --dst C:/data/genimage_q95 --quality 95

Run it on real AND fake folders so both classes go through exactly the same compression.
Keeps the folder structure; existing outputs are skipped so the script can be resumed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aidet import preprocess  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--no-skip", action="store_true", help="overwrite outputs that already exist")
    args = ap.parse_args(argv)

    paths = preprocess.iter_images(args.src)
    if not paths:
        sys.exit(f"no images under {args.src}")
    print(f"{len(paths)} images: {args.src} -> {args.dst} (JPEG q={args.quality}, 4:4:4)")

    t0 = time.time()
    written = skipped = 0
    for i, p in enumerate(paths, 1):
        out = (args.dst / p.relative_to(args.src)).with_suffix(".jpg")
        if out.exists() and not args.no_skip:
            skipped += 1
        else:
            preprocess.jpeg_reencode(p, out, quality=args.quality)
            written += 1
        if i % 500 == 0 or i == len(paths):
            rate = i / max(time.time() - t0, 1e-6)
            print(f"  {i}/{len(paths)}  written={written} skipped={skipped}  {rate:.1f} img/s", flush=True)
    print("done")


if __name__ == "__main__":
    main()
