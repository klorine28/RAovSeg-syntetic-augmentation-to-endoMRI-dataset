"""DATA-2: ResClass logit / firing per test subject across the 5 real-only seeds.

Emits metrics/data2_resclass_firing.csv with one row per (seed, subject).
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "RaovSeg_recreation"))

from evaluate import load_resclass, RESCLASS_THRESHOLD, DEVICE  # noqa: E402


def main() -> None:
    runs = Path("/mnt/parscratch/users/ijp25lg/synth_mri/runs")
    rows = []

    for seed in range(5):
        rd = runs / f"raovseg_real_only_seed{seed}"
        ckpt = rd / "models" / "resclass_best_ov.pth"
        td = rd / "processed" / "test"
        if not ckpt.exists() or not td.exists():
            print(f"seed {seed}: missing ckpt ({ckpt.exists()}) or test dir ({td.exists()})")
            continue

        model = load_resclass(ckpt)
        n_this_seed = 0
        for subj_dir in sorted(td.iterdir()):
            if not subj_dir.is_dir():
                continue
            img_p = subj_dir / "image.npy"
            lab_p = subj_dir / "ov_label.npy"
            if not img_p.exists() or not lab_p.exists():
                continue

            img = np.load(img_p).astype(np.float32)
            lab = np.load(lab_p).astype(bool)

            n_slices = img.shape[0]
            per_slice_prob = np.zeros(n_slices, dtype=np.float32)
            per_slice_gt = np.zeros(n_slices, dtype=bool)

            with torch.no_grad():
                for s in range(n_slices):
                    x = torch.from_numpy(img[s][np.newaxis, np.newaxis]).to(DEVICE)
                    logit = model(x).squeeze()
                    per_slice_prob[s] = torch.sigmoid(logit).item()
                    per_slice_gt[s] = lab[s].any()

            fired = per_slice_prob > RESCLASS_THRESHOLD
            rows.append({
                "seed": seed,
                "subject_id": subj_dir.name,
                "gt_present_volume": int(lab.any()),
                "n_gt_pos_slices": int(per_slice_gt.sum()),
                "n_fired_slices": int(fired.sum()),
                "n_true_pos_slices": int((fired & per_slice_gt).sum()),
                "n_false_pos_slices": int((fired & ~per_slice_gt).sum()),
                "n_false_neg_slices": int((~fired & per_slice_gt).sum()),
                "max_prob": float(per_slice_prob.max()),
                "mean_prob_gt_slices": float(per_slice_prob[per_slice_gt].mean()) if per_slice_gt.any() else float("nan"),
            })
            n_this_seed += 1
        print(f"seed {seed}: {n_this_seed} subjects processed")

    if not rows:
        print("no rows to write")
        return

    out = Path("metrics/data2_resclass_firing.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[saved] {out} ({len(rows)} rows)")

    fired_any = sum(1 for r in rows if r["n_fired_slices"] > 0)
    gt_pos = sum(1 for r in rows if r["gt_present_volume"] == 1)
    tp = sum(1 for r in rows if r["gt_present_volume"] == 1 and r["n_fired_slices"] > 0)
    fp = sum(1 for r in rows if r["gt_present_volume"] == 0 and r["n_fired_slices"] > 0)
    print(f"\nSummary across all seeds/subjects:")
    print(f"  gt-positive volumes:            {gt_pos}/{len(rows)}")
    print(f"  fired ≥1 slice:                 {fired_any}/{len(rows)}")
    print(f"  true-pos (gt+, fired):          {tp}/{gt_pos}")
    print(f"  false-pos (gt-, fired):         {fp}")


if __name__ == "__main__":
    main()
