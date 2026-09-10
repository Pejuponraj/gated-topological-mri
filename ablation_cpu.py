"""
ABLATION on CPU  --  no GPU, no network needed
==============================================

The full ablation needs a GPU because it runs the segmentation network
750 times. Colab is not giving us one right now.

But two of the three questions do not need the network at all. They
can be answered on your laptop, on the real BraTS volumes you already
have, using image error and gate separation instead of Dice.

  Q1  Does the detection contribute?
      ours (Xi-guided frequency subtraction) vs plain FFT subtraction
      with no Xi. Measured by RMSE against the clean volume.

  Q2  Does the Xi score separate clean from corrupted better than the
      alternatives? Against plain spectral concentration, an SNR-like
      ratio, and spectral kurtosis.

Only the Dice numbers need the GPU, and those can wait.

Runs on CPU in roughly 20 to 30 minutes for 10 volumes.

BEFORE RUNNING
  cd C:\\Users\\pejup\\Downloads
  .\\monai-env\\Scripts\\Activate.ps1
  cd C:\\Users\\pejup
  python ablation_cpu.py

The (monai-env) prefix must be visible in the prompt.
"""

import os, glob, time, csv, warnings
warnings.filterwarnings("ignore")
import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.stats import wilcoxon

# ---------------------------------------------------------------
DATA_DIR   = r"brain_data\Task01_BrainTumour\imagesTr"
N_VOLUMES  = 10          # raise to 30 if you have the time
MODALITY   = 0
AMPS       = [0.2, 0.5, 1.0, 2.0]
REPAIR_FRAC = 0.10
NPEAK      = 8
SEED       = 42

# ---------------------------------------------------------------
# artefact
# ---------------------------------------------------------------
def add_stripe(vol, brain, amp, rng):
    if amp == 0.0:
        return vol.copy()
    R = float(vol[brain].max() - vol[brain].min())
    _, yy, xx = np.mgrid[0:vol.shape[0], 0:vol.shape[1], 0:vol.shape[2]]
    kx, ky = int(rng.integers(6, 20)), int(rng.integers(6, 20))
    rip = (np.sin(2*np.pi*(kx*xx + ky*yy)/vol.shape[1]) * amp * R).astype(np.float32)
    out = vol.copy(); out[brain] += rip[brain]
    return out

# ---------------------------------------------------------------
# WITH Xi
# ---------------------------------------------------------------
def crit_hyper(I):
    r1 = ndimage.rank_filter(I, 1, size=3)
    r2 = ndimage.rank_filter(I, -2, size=3)
    return np.maximum(I - r2, r1 - I)

def flagged_residual(I, brain, frac=REPAIR_FRAC):
    v = crit_hyper(I)[brain]
    K = int(frac * v.size)
    fb = np.zeros(v.size, bool); fb[np.argsort(-v)[:K]] = True
    med5 = ndimage.median_filter(I, 5)
    rr = (I - med5)[brain]
    t = np.zeros(v.size, np.float32); t[fb] = rr[fb]
    res = np.zeros_like(I); res[brain] = t
    return res

def score_xi(I, brain):
    Kf = np.abs(np.fft.fftn(flagged_residual(I, brain)))
    return float(np.sort(Kf.ravel())[-6:].sum() / (Kf.sum() + 1e-9) * 1e4)

def repair_xi(I, brain, npeak=NPEAK):
    res = flagged_residual(I, brain)
    Kr = np.abs(np.fft.fftn(res)).ravel().copy(); Kr[0] = -1
    idx = np.argsort(-Kr)[:npeak*2]
    KI = np.fft.fftn(np.where(brain, I, 0.0))
    c = np.zeros_like(KI); c.ravel()[idx] = KI.ravel()[idx]
    out = I.copy(); out[brain] = (I - np.real(np.fft.ifftn(c)))[brain]
    return out

# ---------------------------------------------------------------
# WITHOUT Xi  -- the controls that decide whether Xi earns its place
# ---------------------------------------------------------------
def repair_plain(I, brain, npeak=NPEAK):
    """Subtract the largest non-DC components. No Xi anywhere."""
    KI = np.fft.fftn(np.where(brain, I, 0.0))
    m = np.abs(KI).ravel().copy(); m[0] = -1
    idx = np.argsort(-m)[:npeak*2]
    c = np.zeros_like(KI); c.ravel()[idx] = KI.ravel()[idx]
    out = I.copy(); out[brain] = (I - np.real(np.fft.ifftn(c)))[brain]
    return out

def repair_median(I, brain):
    """Classical blanket median filter."""
    med = ndimage.median_filter(I, 3)
    out = I.copy(); out[brain] = med[brain]
    return out

def score_plain(I, brain):
    Kf = np.abs(np.fft.fftn(np.where(brain, I, 0.0))); Kf.ravel()[0] = 0
    return float(np.sort(Kf.ravel())[-6:].sum() / (Kf.sum() + 1e-9) * 1e4)

def score_snr(I, brain):
    return float(I[brain].std() /
                 (np.abs(I - ndimage.median_filter(I, 3))[brain].mean() + 1e-9))

def score_kurtosis(I, brain):
    Kf = np.abs(np.fft.fftn(np.where(brain, I, 0.0))).ravel().copy(); Kf[0] = 0
    x = Kf / (Kf.sum() + 1e-9)
    return float(((x - x.mean())**4).mean() / (((x - x.mean())**2).mean()**2 + 1e-30))

def rmse(a, b, m):
    return float(np.sqrt(np.mean((a[m] - b[m])**2)))

# ---------------------------------------------------------------
files = sorted(f for f in os.listdir(DATA_DIR)
               if f.startswith("BRATS") and f.endswith(".nii.gz"))[:N_VOLUMES]
if not files:
    raise SystemExit(f"No volumes in {DATA_DIR}")
print(f"volumes: {len(files)}\n")

# ---------------------------------------------------------------
# self-tests
# ---------------------------------------------------------------
print("SELF-TESTS")
ok = True
_r = np.random.default_rng(0)
_S = (20, 50, 50)
_zz, _yy, _xx = np.mgrid[0:_S[0], 0:_S[1], 0:_S[2]]
_b = ((_yy-25)**2/20**2 + (_xx-25)**2/20**2 + (_zz-10)**2/8**2) < 1
_g = ndimage.gaussian_filter(_r.normal(0,1,_S), 2); _g = (_g-_g.min())/(_g.max()-_g.min())
_v = np.zeros(_S, np.float32); _v[_b] = (50 + 1828*_g[_b]).astype(np.float32)

good = np.array_equal(add_stripe(_v, _b, 0.0, _r), _v); ok &= good
print(f"  [{'ok ' if good else 'FAIL'}] amp=0 leaves the volume unchanged")

_I = add_stripe(_v, _b, 1.0, np.random.default_rng(1))
for nm, fn in [("ours", repair_xi), ("plain_fft", repair_plain),
               ("median", repair_median)]:
    n = int(((fn(_I, _b) != _I) & ~_b).sum())
    g2 = (n == 0); ok &= g2
    print(f"  [{'ok ' if g2 else 'FAIL'}] {nm} touches nothing outside the mask")

sc = score_xi(_v, _b); sd = score_xi(_I, _b)
g2 = sd > sc; ok &= g2
print(f"  [{'ok ' if g2 else 'FAIL'}] Xi score rises with corruption "
      f"({sc:.2f} -> {sd:.2f})")
print("  ALL PASSED\n" if ok else "  FAILED\n")
if not ok:
    raise SystemExit("Self-tests failed. Nothing was run.")

# ---------------------------------------------------------------
# Q1  repair, RMSE
# ---------------------------------------------------------------
REPAIRS = {"ours": repair_xi, "plain_fft": repair_plain, "median": repair_median}
res = {(a, m): [] for a in AMPS for m in list(REPAIRS) + ["none"]}
SC = {"xi": ([], []), "plain": ([], []), "snr": ([], []), "kurtosis": ([], [])}
SCORERS = {"xi": score_xi, "plain": score_plain,
           "snr": score_snr, "kurtosis": score_kurtosis}

t0 = time.time()
for vi, fname in enumerate(files):
    rng = np.random.default_rng(SEED + vi)
    vol = nib.load(os.path.join(DATA_DIR, fname)).get_fdata()[..., MODALITY].astype(np.float32)
    brain = vol > 0
    if brain.sum() < 10000:
        print(f"  [{vi+1}] {fname} skipped, too little brain"); continue

    for nm, fn in SCORERS.items():
        SC[nm][0].append(fn(vol, brain))          # clean

    for amp in AMPS:
        I = add_stripe(vol, brain, amp, rng)
        res[(amp, "none")].append(rmse(I, vol, brain))
        for m, fn in REPAIRS.items():
            res[(amp, m)].append(rmse(fn(I, brain), vol, brain))
        for nm, fn in SCORERS.items():
            SC[nm][1].append(fn(I, brain))        # corrupted

    el = (time.time()-t0)/60
    print(f"  [{vi+1}/{len(files)}] {fname}  {el:.1f}m  "
          f"eta {el/(vi+1)*(len(files)-vi-1):.1f}m")

n = len(res[(AMPS[0], "none")])
print(f"\n{n} volumes processed\n")

# ---------------------------------------------------------------
print("="*72)
print("Q1  DOES THE DETECTION CONTRIBUTE?   RMSE, lower is better")
print("="*72)
print(f"{'amp':>6}{'corrupted':>12}{'ours':>12}{'plain_fft':>12}{'median':>12}")
for a in AMPS:
    print(f"{a:>6}" + "".join(f"{np.mean(res[(a,m)]):12.1f}"
                              for m in ["none","ours","plain_fft","median"]))

print("\n  paired comparisons, ours against each alternative:")
for a in AMPS:
    o = np.array(res[(a,"ours")])
    for m in ["none","plain_fft","median"]:
        x = np.array(res[(a,m)]); d = o - x       # lower is better, so negative wins
        try: _, p = wilcoxon(o, x); ps = f"p={p:.1e}"
        except ValueError: ps = "p=n/a"
        tag = "BETTER" if d.mean() < 0 else "worse "
        print(f"    amp {a}  vs {m:<10} {tag} {d.mean():+9.1f}  {ps}  "
              f"{int((d<0).sum())}/{len(d)}")

# ---------------------------------------------------------------
print("\n" + "="*72)
print("Q2  DOES THE Xi SCORE SEPARATE BETTER THAN THE ALTERNATIVES?")
print("="*72)
print(f"{'score':>12}{'clean range':>26}{'corrupted range':>26}   verdict")
for nm in SCORERS:
    cl = np.array(SC[nm][0]); dy = np.array(SC[nm][1])
    sep = (cl.max() < dy.min()) or (cl.min() > dy.max())
    if sep:
        gap = (dy.min()/cl.max()) if cl.max() < dy.min() else (cl.min()/dy.max())
        v = f"SEPARATED  gap x{gap:.1f}"
    else:
        v = "overlap"
    print(f"{nm:>12}  [{cl.min():10.2f},{cl.max():10.2f}]"
          f"  [{dy.min():10.2f},{dy.max():10.2f}]   {v}")

# ---------------------------------------------------------------
print("\n" + "="*72)
print("VERDICT")
print("="*72)
q1 = all(np.mean(res[(a,"ours")]) < np.mean(res[(a,"plain_fft")]) for a in AMPS)
cl = np.array(SC["xi"][0]); dy = np.array(SC["xi"][1])
q2 = cl.max() < dy.min()
others = [nm for nm in ["plain","snr","kurtosis"]
          if np.array(SC[nm][0]).max() < np.array(SC[nm][1]).min()]
print(f"  detection beats plain FFT at every amplitude : {'YES' if q1 else 'NO'}")
print(f"  Xi score separates clean from corrupted      : {'YES' if q2 else 'NO'}")
print(f"  alternatives that also separate              : "
      f"{', '.join(others) if others else 'none'}")

if q1 and q2 and not others:
    print("""
  Both components earn their place, and no simpler score does the same
  job. This is the ablation the paper was missing. Put it in as a main
  table.

  The Dice version still needs a GPU, but if RMSE and separation both
  favour the method, Dice is very unlikely to reverse it.""")
else:
    print("""
  At least one component does not earn its place, or a simpler score
  matches it. Report that plainly in the paper and drop or reframe the
  component concerned.""")

with open("ablation_cpu_results.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["quantity","amp","method","value"])
    for a in AMPS:
        for m in ["none","ours","plain_fft","median"]:
            for val in res[(a,m)]:
                w.writerow(["rmse", a, m, val])
    for nm in SCORERS:
        for val in SC[nm][0]:
            w.writerow(["score_clean", 0.0, nm, val])
        for val in SC[nm][1]:
            w.writerow(["score_dirty", "", nm, val])
print("\nwrote ablation_cpu_results.csv")
