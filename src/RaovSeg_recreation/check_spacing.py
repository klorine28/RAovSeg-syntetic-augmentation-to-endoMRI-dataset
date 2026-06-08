"""Check native NIfTI spacing for D2 T2FS scans.

The paper says "voxel size of 5mm by 5mm" — but at 512x512 pixels that would
be a 2.56m field of view, which is physically impossible. Run this to see
what the source data actually is and what our resampling is doing.
"""
import sys
from pathlib import Path
import SimpleITK as sitk

base = Path(sys.argv[1] if len(sys.argv) > 1
            else "/mnt/parscratch/users/ijp25lg/synth_mri/EndometriosisDataset/UT-EndoMRI/D2_TCPW")

subjects = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith("D2-")])

print(f"{'subject':<10} {'shape':<20} {'spacing (mm)':<35} {'FOV (mm)':<25}")
print("-" * 90)

for subj in subjects:
    nii = subj / f"{subj.name}_T2FS.nii.gz"
    if not nii.exists():
        continue
    img = sitk.ReadImage(str(nii))
    size = img.GetSize()
    spacing = img.GetSpacing()
    fov = tuple(s * sp for s, sp in zip(size, spacing))
    print(f"{subj.name:<10} {str(size):<20} {str(tuple(round(s,3) for s in spacing)):<35} {str(tuple(round(f,1) for f in fov)):<25}")