"""
Phase 2 planning: audit D1 T2 vs D2 T2FS NIfTI headers.

Prints per-subject headers and cross-cohort summary stats for:
  spacing (mm), size (voxels), direction, origin.

Used to design the D1 → D2 resampling recipe so Phase 2 synth (D1-anatomy
+ D2-style) is drop-in compatible with the RAovSeg preprocessing pipeline.

Usage on HPC:
    python -m src.Generator.audit_d1_vs_d2_headers \
        --d1-dir UT-EndoMRI/D1_MHS \
        --d2-dir UT-EndoMRI/D2_TCPW \
        --out-json d1_vs_d2_headers.json
"""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import SimpleITK as sitk


def _read_header(nii_path: Path) -> dict:
    img = sitk.ReadImage(str(nii_path))
    return {
        "path": str(nii_path),
        "spacing": list(img.GetSpacing()),
        "size": list(img.GetSize()),
        "direction": list(img.GetDirection()),
        "origin": list(img.GetOrigin()),
        "pixel_type": img.GetPixelIDTypeAsString(),
    }


def _scan_cohort(cohort_dir: Path, suffix: str) -> list[dict]:
    headers = []
    for subj_dir in sorted(cohort_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        nii = subj_dir / f"{subj_dir.name}{suffix}.nii.gz"
        if not nii.exists():
            continue
        try:
            headers.append({"subject": subj_dir.name, **_read_header(nii)})
        except Exception as e:
            print(f"[warn] failed to read {nii}: {e}")
    return headers


def _summarise(headers: list[dict], label: str) -> dict:
    if not headers:
        return {"label": label, "n": 0}
    sx = [h["spacing"][0] for h in headers]
    sy = [h["spacing"][1] for h in headers]
    sz = [h["spacing"][2] for h in headers]
    nx = [h["size"][0] for h in headers]
    ny = [h["size"][1] for h in headers]
    nz = [h["size"][2] for h in headers]

    def stats(vs):
        return {
            "min": min(vs),
            "max": max(vs),
            "mean": mean(vs),
            "std": stdev(vs) if len(vs) > 1 else 0.0,
        }

    return {
        "label": label,
        "n": len(headers),
        "spacing_x_mm": stats(sx),
        "spacing_y_mm": stats(sy),
        "spacing_z_mm": stats(sz),
        "size_x_vox": stats(nx),
        "size_y_vox": stats(ny),
        "size_z_vox": stats(nz),
    }


def _print_summary(s: dict) -> None:
    print(f"\n=== {s['label']} (n={s['n']}) ===")
    if s["n"] == 0:
        print("  (no volumes found)")
        return
    for k in ("spacing_x_mm", "spacing_y_mm", "spacing_z_mm",
             "size_x_vox", "size_y_vox", "size_z_vox"):
        v = s[k]
        print(f"  {k:15s}  min={v['min']:.3f}  max={v['max']:.3f}  "
              f"mean={v['mean']:.3f}  std={v['std']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d1-dir", required=True, type=Path,
                    help="Path to UT-EndoMRI/D1_MHS")
    ap.add_argument("--d2-dir", required=True, type=Path,
                    help="Path to UT-EndoMRI/D2_TCPW")
    ap.add_argument("--d1-suffix", default="_T2",
                    help="D1 modality suffix (default: _T2)")
    ap.add_argument("--d2-suffix", default="_T2FS",
                    help="D2 modality suffix (default: _T2FS)")
    ap.add_argument("--out-json", type=Path, default=None,
                    help="If set, write full per-subject headers here")
    args = ap.parse_args()

    d1 = _scan_cohort(args.d1_dir, args.d1_suffix)
    d2 = _scan_cohort(args.d2_dir, args.d2_suffix)

    s1 = _summarise(d1, f"D1_MHS {args.d1_suffix}")
    s2 = _summarise(d2, f"D2_TCPW {args.d2_suffix}")

    _print_summary(s1)
    _print_summary(s2)

    print("\n=== resampling delta (D1 → D2 target) ===")
    if s1["n"] and s2["n"]:
        for axis in ("x", "y", "z"):
            sp_d1 = s1[f"spacing_{axis}_mm"]["mean"]
            sp_d2 = s2[f"spacing_{axis}_mm"]["mean"]
            print(f"  spacing_{axis}: D1={sp_d1:.3f}mm  D2={sp_d2:.3f}mm  "
                  f"ratio={sp_d1/sp_d2:.3f}")
        for axis in ("x", "y", "z"):
            sz_d1 = s1[f"size_{axis}_vox"]["mean"]
            sz_d2 = s2[f"size_{axis}_vox"]["mean"]
            print(f"  size_{axis}:    D1={sz_d1:.0f}vox   D2={sz_d2:.0f}vox   "
                  f"ratio={sz_d1/sz_d2:.3f}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({"d1": d1, "d2": d2, "summary_d1": s1, "summary_d2": s2},
                      f, indent=2)
        print(f"\n[saved] {args.out_json}")


if __name__ == "__main__":
    main()
