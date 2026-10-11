"""
precompute.py  -  run the method once on every sample case and save the result,
so the demo opens each case instantly (also online, on a slow free CPU).
Run once, after make_stripe_samples.py:   python precompute.py
Output: the folder 'precomputed' (one small .npz file per case).
"""
import os, glob, re, time
import api

os.makedirs(api.PRECOMP, exist_ok=True)
files = sorted(glob.glob(os.path.join("samples", "vol*_flair.nii.gz")))
cases = []
for f in files:
    m = re.match(r"vol(\d)_(BRATS_\d+)(?:_stripe([\d.]+))?_flair\.nii\.gz$", os.path.basename(f))
    if not m:
        continue
    vi, case, amp = m.groups()
    order = int(vi) * 10 + (0 if amp is None else int(float(amp) * 4) + 1)
    name = f"{case} — " + ("clean scan" if amp is None else f"simulated stripe, amplitude {amp}")
    lab = os.path.join("samples", f"vol{vi}_{case}_label.nii.gz")
    cases.append((order, name, f, lab if os.path.exists(lab) else None))

for order, name, f, lab in sorted(cases):
    t0 = time.time()
    data = open(f, "rb").read()
    ldata = open(lab, "rb").read() if lab else b""
    rid = api.result_id(data, ldata)
    res = api.process(data, ldata)
    res["summary"]["case_name"] = name
    res["summary"]["order"] = order
    out = os.path.join(api.PRECOMP, rid + ".npz")
    api.save_result(res, out)
    s = res["summary"]
    print(f"{name:45s} score {s['score']:.3f}  detected {s['artefact_detected']}  "
          f"Dice {s.get('dice_before', float('nan')):.3f} -> {s.get('dice_after', float('nan')):.3f}  "
          f"{os.path.getsize(out)/1e6:.1f} MB  {time.time()-t0:.0f}s")
print(f"done: {len(cases)} cases saved in '{api.PRECOMP}'")
