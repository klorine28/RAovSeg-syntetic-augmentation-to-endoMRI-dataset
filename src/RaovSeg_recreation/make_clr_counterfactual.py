"""
CLR counterfactual figure (Fig 4.3): stack the cached explainability
counterfactual-ablation rows for concat vs SPADE to show, at the pixel level,
that removing the uterus channel changes the whole frame for concat (locked
out) but only the organ region for SPADE.

Reads the cached explain reports (rendered earlier by src/Generator/explain.py):
    1c/concat/explain/sample_00.png  + sample_00_metrics.json
    1c/spade/explain/sample_00.png   + sample_00_metrics.json

Run:  python -m src.RaovSeg_recreation.make_clr_counterfactual --root . --out-dir figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# TEST-3 counterfactual-row crops (px) for the explain.py report layout.
# concat report is shorter than spade (no SPADE-gamma section), so Y differs.
CROP = {"concat": (10, 1028, 1620, 1320), "spade": (10, 1035, 1620, 1300)}


def _clr_uterus(p: Path) -> float:
    return json.loads(p.read_text())["CLR_per_channel"]["uterus"]


def make(root: Path, out_dir: Path) -> bool:
    cpng = root / "1c/concat/explain/sample_00.png"
    spng = root / "1c/spade/explain/sample_00.png"
    if not (cpng.exists() and spng.exists()):
        print("[skip] explain reports not found:", cpng, spng)
        return False
    clr_c = _clr_uterus(root / "1c/concat/explain/sample_00_metrics.json")
    clr_s = _clr_uterus(root / "1c/spade/explain/sample_00_metrics.json")
    row_c = Image.open(cpng).crop(CROP["concat"])   # keeps panel titles
    row_s = Image.open(spng).crop(CROP["spade"])

    fig = plt.figure(figsize=(13, 5.4))
    gs = fig.add_gridspec(2, 1, hspace=0.04, left=0.11, right=0.995, top=0.9, bottom=0.11)
    ax0 = fig.add_subplot(gs[0]); ax0.imshow(row_c); ax0.axis("off")
    ax1 = fig.add_subplot(gs[1]); ax1.imshow(row_s); ax1.axis("off")
    ax0.text(-0.012, 0.5, f"concat\nCLR(uterus) ≈ {clr_c:.2f}", transform=ax0.transAxes,
             ha="right", va="center", rotation=90, fontsize=11, fontweight="bold", color="#C44E52")
    ax1.text(-0.012, 0.5, f"SPADE\nCLR(uterus) ≈ {clr_s:.2f}", transform=ax1.transAxes,
             ha="right", va="center", rotation=90, fontsize=11, fontweight="bold", color="#4C72B0")
    fig.suptitle("Counterfactual label ablation — image change when the uterus channel is removed",
                 fontsize=13, fontweight="bold", y=0.975)
    fig.text(0.5, 0.045,
             "Panels: full label · each organ removed in turn · rightmost = difference map "
             "(remove uterus). Red = brighter, blue = darker after removal; white = unchanged.\n"
             "concat changes the whole frame (label used globally — locked out); "
             "SPADE changes mainly the organ region (genuine per-organ conditioning).",
             ha="center", fontsize=9, color="#333333")
    out = out_dir / "fig_clr_counterfactual.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out}  (CLR concat={clr_c:.3f}, spade={clr_s:.3f})")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    make(args.root, args.out_dir)


if __name__ == "__main__":
    main()
