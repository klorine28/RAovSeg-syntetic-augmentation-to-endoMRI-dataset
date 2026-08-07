"""
Cap every PNG in figures/ to a maximum edge of 2000 px (in place, high quality).
Keeps files small and lets them attach to image-limited chat interfaces.

Run:  python -m src.RaovSeg_recreation.cap_figures --dir figures --max 2000
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

from PIL import Image


def cap(directory: Path, max_edge: int = 2000) -> None:
    n = 0
    for f in sorted(glob.glob(str(directory / "*.png"))):
        im = Image.open(f)
        w, h = im.size
        if max(w, h) <= max_edge:
            continue
        scale = max_edge / max(w, h)
        im = im.convert("RGB") if im.mode in ("P", "RGBA") else im
        im = im.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
        im.save(f, optimize=True)
        print(f"  {w}x{h} -> {im.size[0]}x{im.size[1]}  {Path(f).name}")
        n += 1
    print(f"capped {n} images to <= {max_edge}px")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("figures"))
    ap.add_argument("--max", type=int, default=2000)
    args = ap.parse_args()
    cap(args.dir, args.max)


if __name__ == "__main__":
    main()
