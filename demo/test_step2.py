"""
Step 2 check. Runs predict() on held-out volume 0 and compares with the
numbers from the paper's experiment (v11_full30.csv, computed on a GPU).
Run from the project folder:   python test_step2.py
"""
import glob, time
import numpy as np
from gtm_core import predict, dice, load_flair

# exactly the stripe generator of the experiment (v8/v11)
def add_stripe(vol, brain, amp, rng):
    if amp == 0.0:
        return vol.copy()
    R = float(vol[brain].max() - vol[brain].min())
    _, yy, xx = np.mgrid[0:vol.shape[0], 0:vol.shape[1], 0:vol.shape[2]]
    kx, ky = int(rng.integers(6, 20)), int(rng.integers(6, 20))
    rip = (np.sin(2*np.pi*(kx*xx + ky*yy)/vol.shape[1]) * amp * R).astype(np.float32)
    out = vol.copy(); out[brain] += rip[brain]
    return out

# reference: v11_full30.csv, vol 0  (score, Dice no correction, Dice ours)
REF = {0.0: (0.1658, 0.82246, 0.82246),
       1.0: (2.1359, 0.49416, 0.53842),
       2.0: (2.6070, 0.31411, 0.52999)}

flair = glob.glob("samples/vol0_*_flair.nii.gz")[0]
label = glob.glob("samples/vol0_*_label.nii.gz")[0]
vol, _ = load_flair(flair); bt = vol > 0
gt = load_flair(label)[0] > 0
rng = np.random.default_rng(42 + 1000 + 0)                 # SEED + 1000 + vi
inputs = {a: add_stripe(vol, bt, a, rng) for a in [0.0, 0.2, 0.5, 1.0, 2.0]}

ok = True
for amp in [0.0, 1.0, 2.0]:
    t0 = time.time()
    r = predict(inputs[amp])
    d0, d1 = dice(r["seg_before"], gt), dice(r["seg_after"], gt)
    s_ref, d0_ref, d1_ref = REF[amp]
    good = (abs(r["score"] - s_ref) < 0.02 and abs(d0 - d0_ref) < 0.01
            and abs(d1 - d1_ref) < 0.01 and r["artefact_detected"] == (amp > 0))
    ok &= good
    print(f"stripe {amp}: score {r['score']:.4f} (paper {s_ref})  "
          f"detected {r['artefact_detected']}  "
          f"Dice before {d0:.4f} (paper {d0_ref})  after {d1:.4f} (paper {d1_ref})  "
          f"{time.time()-t0:.0f}s  {'OK' if good else 'DIFFERENT'}")
# the same case read from the saved file (as the API does) must agree
import os
sf = flair.replace("_flair.nii.gz", "_stripe2.0_flair.nii.gz")
if os.path.exists(sf):
    r = predict(sf)
    d1 = dice(r["seg_after"], gt)
    good = abs(d1 - REF[2.0][2]) < 0.01
    ok &= good
    print(f"stripe 2.0 from file: Dice after {d1:.4f} (paper {REF[2.0][2]})  {'OK' if good else 'DIFFERENT'}")
print("STEP 2 PASSED" if ok else "STEP 2: numbers differ - send me this output")
