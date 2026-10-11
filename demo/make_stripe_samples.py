"""
Makes the simulated-stripe versions of the sample scans, with the SAME
generator and seeds as the paper (v8/v11), so the demo shows the paper's cases.
Run once:   python make_stripe_samples.py
"""
import glob, os
import numpy as np, nibabel as nib

def add_stripe(vol, brain, amp, rng):
    if amp == 0.0:
        return vol.copy()
    R = float(vol[brain].max() - vol[brain].min())
    _, yy, xx = np.mgrid[0:vol.shape[0], 0:vol.shape[1], 0:vol.shape[2]]
    kx, ky = int(rng.integers(6, 20)), int(rng.integers(6, 20))
    rip = (np.sin(2*np.pi*(kx*xx + ky*yy)/vol.shape[1]) * amp * R).astype(np.float32)
    out = vol.copy(); out[brain] += rip[brain]
    return out

for vi in range(3):
    # the original scan only (never a stripe file made by an earlier run)
    path = [p for p in glob.glob(f"samples/vol{vi}_*_flair.nii.gz") if "stripe" not in p][0]
    img = nib.load(path); vol = img.get_fdata().astype(np.float32); bt = vol > 0
    rng = np.random.default_rng(42 + 1000 + vi)          # SEED + 1000 + vi, as v11
    for amp in [0.0, 0.2, 0.5, 1.0, 2.0]:                # same order as v11
        out = add_stripe(vol, bt, amp, rng)
        if amp in (0.5, 2.0):
            name = path.replace("_flair.nii.gz", f"_stripe{amp}_flair.nii.gz")
            nib.save(nib.Nifti1Image(out, img.affine), name)
            print("saved", os.path.basename(name))
