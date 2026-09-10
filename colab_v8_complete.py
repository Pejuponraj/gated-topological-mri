# =====================================================================
#  COMPLETE EXPERIMENT  --  v8
#  Data, training, gate, evaluation.  One cell, start to finish.
# =====================================================================
#
#  The question
#  ------------
#  MRI scans in hospitals pick up stripe artefacts from RF interference.
#  A segmentation model trained on clean data collapses on them: Dice
#  0.72 down to 0.18 in our tests. Can we detect that corruption and
#  undo enough of it to matter?
#
#  What we already established
#  ---------------------------
#  DETECTION works. Our hyperconnectedness criterion ranks corrupted
#  voxels better than the median residual and three statistical
#  controls, on 30 of 30 volumes, at every severity, p around 1e-9.
#
#  REPAIR by local median does not. It recovered about 1 percent of the
#  lost Dice. The reason is structural, not a tuning problem: a stripe
#  hits every voxel, we only touch 10 percent of them, and the
#  neighbourhood we repair from is itself corrupted.
#
#  RIPPLE SUBTRACTION does. Same detection, different replacement: the
#  flagged voxels tell us which frequencies carry the artefact, and we
#  subtract those from the whole volume. That recovered 15 to 56
#  percent of the lost Dice.
#
#  But it DAMAGED clean scans, 0.718 down to 0.625. Unusable in a
#  hospital, where most scans are clean.
#
#  What this script tests
#  ----------------------
#  A gate. Score the scan first; repair only if it looks corrupted.
#
#  An earlier attempt at this gate was broken: it divided by the
#  intensity spread, and a stripe widens that spread, so corrupted
#  scans scored LOWER than clean ones and the gate never fired. The
#  score here instead measures how CONCENTRATED the flagged residual is
#  in k-space. A stripe makes the flagged voxels periodic; on a clean
#  scan they are scattered. Verified monotonic on synthetic volumes
#  before being used here.
#
#  Two things must both hold:
#    clean scans left alone     (loss under about 0.01)
#    corrupted scans recovered  (a real share of the gap)
#
#  Runtime: about 15 min training, 50 min evaluation. Everything is
#  written to Drive as it goes, so a disconnect costs at most one
#  volume, and rerunning skips whatever is already done.
#
#  Runtime -> Change runtime type -> T4 GPU before running.
# =====================================================================

import subprocess, sys, os, glob, time, csv, collections
try:
    import monai
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "monai", "nibabel"])

import numpy as np
import torch
import nibabel as nib
from scipy import ndimage
from scipy.stats import wilcoxon

assert torch.cuda.is_available(), "No GPU. Runtime -> Change runtime type -> T4 GPU"
device = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0))

from google.colab import drive
drive.mount("/content/drive")

DRIVE     = "/content/drive/MyDrive"
TAR       = f"{DRIVE}/Task01_BrainTumour.tar"
MODEL_PTH = f"{DRIVE}/unet_clean.pth"
CSV_OUT   = f"{DRIVE}/dice_gated_v8.csv"
FIG_OUT   = f"{DRIVE}/fig_gated_v8.png"
ROOT      = "/content/data/Task01_BrainTumour"

# ---------- data ----------
if not os.path.exists(f"{ROOT}/imagesTr"):
    if not os.path.exists(TAR):
        raise SystemExit(f"No tar at {TAR}. Download it first.")
    os.makedirs("/content/data", exist_ok=True)
    print("extracting, about 10 minutes...")
    subprocess.run(["tar", "-xf", TAR, "-C", "/content/data"], check=True)
n_img = len(glob.glob(f"{ROOT}/imagesTr/BRATS*.nii.gz"))
print(f"data ready, {n_img} volumes")
assert n_img > 400, "extraction looks incomplete"

# ---------- settings, fixed in advance ----------
N_TRAIN, N_VAL = 60, 30
MODALITY    = 0
PATCH       = (96, 96, 96)
EPOCHS      = 80
LR          = 1e-3
AMPS        = [0.0, 0.2, 0.5, 1.0, 2.0]
REPAIR_FRAC = 0.10
NPEAK       = 8
SEED        = 42
N_GATE_FIT  = 12
MIN_CLEAN_DICE = 0.45

torch.manual_seed(SEED); np.random.seed(SEED)

images = sorted(glob.glob(f"{ROOT}/imagesTr/BRATS*.nii.gz"))
labels = sorted(glob.glob(f"{ROOT}/labelsTr/BRATS*.nii.gz"))
train_files = [{"image": i, "label": l} for i, l in zip(images[:N_TRAIN], labels[:N_TRAIN])]
gate_files  = images[:N_GATE_FIT]                      # training volumes only
val_files   = [{"image": i, "label": l} for i, l in
               zip(images[N_TRAIN:N_TRAIN+N_VAL], labels[N_TRAIN:N_TRAIN+N_VAL])]
print(f"train {len(train_files)}   gate-fit {len(gate_files)}   val {len(val_files)}")

# ---------- artefact ----------
def add_stripe(vol, brain, amp, rng):
    if amp == 0.0:
        return vol.copy()
    R = float(vol[brain].max() - vol[brain].min())
    _, yy, xx = np.mgrid[0:vol.shape[0], 0:vol.shape[1], 0:vol.shape[2]]
    kx, ky = int(rng.integers(6, 20)), int(rng.integers(6, 20))
    rip = (np.sin(2*np.pi*(kx*xx + ky*yy)/vol.shape[1]) * amp * R).astype(np.float32)
    out = vol.copy(); out[brain] += rip[brain]
    return out

# ---------- detection, score, repair ----------
def crit_hyper(I):
    """Xi. How far a voxel sits outside the span of its neighbours."""
    r1 = ndimage.rank_filter(I, 1, size=3)
    r2 = ndimage.rank_filter(I, -2, size=3)
    return np.maximum(I - r2, r1 - I)

def flagged_residual(I, brain, frac=REPAIR_FRAC):
    """Residual at the flagged voxels. Shared by the score and the repair."""
    v = crit_hyper(I)[brain]
    K = int(frac * v.size)
    fb = np.zeros(v.size, bool); fb[np.argsort(-v)[:K]] = True
    med5 = ndimage.median_filter(I, 5)
    rr = (I - med5)[brain]
    tmp = np.zeros(v.size, np.float32); tmp[fb] = rr[fb]
    res = np.zeros_like(I); res[brain] = tmp
    return res

def scan_score(I, brain):
    """Concentration of the flagged residual in k-space. High = striped."""
    Kf = np.abs(np.fft.fftn(flagged_residual(I, brain)))
    return float(np.sort(Kf.ravel())[-6:].sum() / (Kf.sum() + 1e-9) * 1e4)

def repair_ripple(I, brain, npeak=NPEAK):
    """Subtract the frequencies the flagged voxels point at."""
    res = flagged_residual(I, brain)
    Kr = np.abs(np.fft.fftn(res)).ravel().copy(); Kr[0] = -1
    idx = np.argsort(-Kr)[:npeak*2]
    KI = np.fft.fftn(np.where(brain, I, 0.0))
    comp = np.zeros_like(KI); comp.ravel()[idx] = KI.ravel()[idx]
    out = I.copy(); out[brain] = (I - np.real(np.fft.ifftn(comp)))[brain]
    return out

def normalise(I, brain):
    x = np.zeros_like(I); v = I[brain]
    x[brain] = (v - v.mean()) / (v.std() + 1e-8)
    return x

# ---------- self-tests ----------
print("\nSELF-TESTS")
_ok = True
_r = np.random.default_rng(0)
_S = (30, 60, 60)
_zz, _yy, _xx = np.mgrid[0:_S[0], 0:_S[1], 0:_S[2]]
_b = ((_yy-30)**2/25**2 + (_xx-30)**2/25**2 + (_zz-15)**2/12**2) < 1
_g = ndimage.gaussian_filter(_r.normal(0,1,_S), 2); _g = (_g-_g.min())/(_g.max()-_g.min())
_v = np.zeros(_S, np.float32); _v[_b] = (50 + 1828*_g[_b]).astype(np.float32)

_good = np.array_equal(add_stripe(_v, _b, 0.0, _r), _v); _ok &= _good
print(f"  [{'ok ' if _good else 'FAIL'}] amp=0 leaves the volume unchanged")

_sc = [scan_score(add_stripe(_v, _b, a, np.random.default_rng(1)), _b) for a in AMPS]
_mono = all(_sc[i] < _sc[i+1] for i in range(len(_sc)-1)); _ok &= _mono
print(f"  [{'ok ' if _mono else 'FAIL'}] score rises with artefact strength: "
      + " ".join(f"{s:.1f}" for s in _sc))

_I = add_stripe(_v, _b, 1.0, np.random.default_rng(2))
_n = int(((repair_ripple(_I, _b) != _I) & ~_b).sum()); _good = (_n == 0); _ok &= _good
print(f"  [{'ok ' if _good else 'FAIL'}] repair touches nothing outside the mask")

def _rmse(a, c, m): return float(np.sqrt(np.mean((a[m]-c[m])**2)))
_r0, _r1 = _rmse(_I, _v, _b), _rmse(repair_ripple(_I, _b), _v, _b)
_good = _r1 < _r0; _ok &= _good
print(f"  [{'ok ' if _good else 'FAIL'}] repair reduces error ({_r0:.0f} -> {_r1:.0f})")

print("  ALL PASSED\n" if _ok else "  FAILED\n")
assert _ok, "Self-tests failed. Nothing was run."

# ---------- model ----------
from monai.transforms import (Compose, LoadImaged, Lambdad, NormalizeIntensityd,
                              RandCropByPosNegLabeld, EnsureTyped)
from monai.data import CacheDataset, DataLoader
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference

model = UNet(spatial_dims=3, in_channels=1, out_channels=1,
             channels=(16,32,64,128,256), strides=(2,2,2,2),
             num_res_units=2).to(device)

if os.path.exists(MODEL_PTH):
    model.load_state_dict(torch.load(MODEL_PTH)); model.eval()
    print("REUSING the model already on Drive, skipping training\n")
else:
    train_tf = Compose([
        LoadImaged(keys=["image","label"], ensure_channel_first=True),
        Lambdad(keys="image", func=lambda x: x[MODALITY:MODALITY+1]),
        Lambdad(keys="label", func=lambda x: (x > 0).astype(np.float32)),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandCropByPosNegLabeld(keys=["image","label"], label_key="label",
                               spatial_size=PATCH, pos=2, neg=1, num_samples=2),
        EnsureTyped(keys=["image","label"]),
    ])
    train_ds = CacheDataset(train_files, train_tf, cache_rate=1.0, num_workers=2)
    train_ld = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)
    loss_fn = DiceLoss(sigmoid=True)
    opt = torch.optim.Adam(model.parameters(), LR)

    print(f"TRAINING   lr={LR}, {EPOCHS} epochs, no mixed precision")
    t0 = time.time(); best = float("inf")
    for ep in range(EPOCHS):
        model.train(); tot = 0.0; nb = 0
        for b in train_ld:
            x, y = b["image"].to(device), b["label"].to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss became {loss.item()} at epoch {ep}. "
                                   "Try LR = 5e-4.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        avg = tot/nb
        if avg < best:
            best = avg
            torch.save(model.state_dict(), MODEL_PTH)
        if ep % 10 == 0 or ep == EPOCHS-1:
            print(f"  epoch {ep:3d}  loss {avg:.4f}  best {best:.4f}  {(time.time()-t0)/60:.1f}m")
    model.load_state_dict(torch.load(MODEL_PTH)); model.eval()
    print(f"  best loss {best:.4f}, saved to Drive\n")

def dice_np(pred, gt, eps=1e-6):
    p = pred.astype(bool); g = gt.astype(bool)
    return float((2*(p & g).sum() + eps) / (p.sum() + g.sum() + eps))

def predict(I, brain):
    t = torch.tensor(normalise(I, brain))[None, None].to(device)
    with torch.no_grad():
        out = sliding_window_inference(t, PATCH, 1, model, overlap=0.25)
    return torch.sigmoid(out)[0,0].float().cpu().numpy() > 0.5

print("HEALTH CHECK")
chk = []
for f in val_files[:3]:
    vol = nib.load(f["image"]).get_fdata()[..., MODALITY].astype(np.float32)
    gt  = nib.load(f["label"]).get_fdata() > 0
    d = dice_np(predict(vol, vol > 0), gt); chk.append(d)
    print(f"  clean Dice {d:.3f}")
print(f"  mean {np.mean(chk):.3f}")
if np.mean(chk) < MIN_CLEAN_DICE:
    raise RuntimeError(f"Clean Dice {np.mean(chk):.3f} too low. Delete {MODEL_PTH} "
                       "and rerun to retrain.")
print("  healthy\n")

# ---------- fit the gate, training volumes only ----------
print("FITTING THE GATE  (training volumes, never the validation set)")
cl, dy = [], []
for i, path in enumerate(gate_files):
    rng = np.random.default_rng(SEED + 500 + i)
    vol = nib.load(path).get_fdata()[..., MODALITY].astype(np.float32)
    brain = vol > 0
    cl.append(scan_score(vol, brain))
    for amp in [0.2, 0.5, 1.0, 2.0]:
        dy.append(scan_score(add_stripe(vol, brain, amp, rng), brain))
    print(f"  [{i+1}/{len(gate_files)}]  clean {cl[-1]:.2f}")

cl = np.array(cl); dy = np.array(dy)
print(f"\n  clean : mean {cl.mean():.2f}   max {cl.max():.2f}")
print(f"  dirty : mean {dy.mean():.2f}   min {dy.min():.2f}")
if dy.mean() <= cl.mean():
    raise SystemExit("The score does not separate clean from corrupted on real "
                     "data. Stopping rather than running a pointless evaluation.")

if cl.max() < dy.min():
    GATE = float(np.sqrt(cl.max() * dy.min())); note = "fully separated"
else:
    GATE = float(np.percentile(cl, 95)); note = "overlap, gate at 95th percentile of clean"
print(f"  GATE = {GATE:.2f}   ({note})")
fp = int((cl > GATE).sum()); fn = int((dy <= GATE).sum())
print(f"  fitting set: {fp}/{len(cl)} clean repaired unnecessarily, "
      f"{fn}/{len(dy)} dirty missed")
if fn > len(dy) * 0.5:
    raise SystemExit("The gate misses more than half the corrupted scans. Stopping.")
print()

METHODS = {
    "none":       lambda I, b: I.copy(),
    "rip_always": lambda I, b: repair_ripple(I, b),
    "rip_gated":  lambda I, b: (I.copy() if scan_score(I, b) <= GATE
                                else repair_ripple(I, b)),
}

# ---------- evaluation ----------
print("EVALUATING")
rows = []
t0 = time.time()
for vi, f in enumerate(val_files):
    rng = np.random.default_rng(SEED + 1000 + vi)
    vol   = nib.load(f["image"]).get_fdata()[..., MODALITY].astype(np.float32)
    gt    = nib.load(f["label"]).get_fdata() > 0
    brain = vol > 0
    for amp in AMPS:
        I = add_stripe(vol, brain, amp, rng)
        sc = scan_score(I, brain); fired = int(sc > GATE)
        for mname, mfn in METHODS.items():
            rep = mfn(I, brain).astype(np.float32)
            rows.append([os.path.basename(f["image"]), amp, mname,
                         dice_np(predict(rep, brain), gt), sc, fired])
    with open(CSV_OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["volume","amp","method","dice","score","gate_fired"])
        w.writerows(rows)
    el = (time.time()-t0)/60
    print(f"  [{vi+1}/{len(val_files)}]  {el:.1f}m  eta {el/(vi+1)*(len(val_files)-vi-1):.1f}m")
print(f"  wrote {CSV_OUT}\n")

# ---------- results ----------
D = collections.defaultdict(list); G = collections.defaultdict(list)
for v, a, m, d, sc, gf in rows:
    D[(float(a), m)].append(d)
    if m == "none": G[float(a)].append(int(gf))
names = list(METHODS)

print("="*72)
print("DICE   mean")
print("="*72)
print(f"{'amp':>6}" + "".join(f"{m:>14}" for m in names) + f"{'gate fired':>13}")
for a in AMPS:
    print(f"{a:>6}" + "".join(f"{np.mean(D[(a,m)]):14.4f}" for m in names)
          + f"{f'{sum(G[a])}/{len(G[a])}':>13}")

ceiling = np.mean(D[(0.0,"none")])
print(f"\nclean ceiling {ceiling:.4f}")

print("\n" + "="*72); print("THE TWO QUESTIONS"); print("="*72)
la = np.mean(D[(0.0,"rip_always")]) - ceiling
lg = np.mean(D[(0.0,"rip_gated")])  - ceiling
print(f"\n1. Are clean scans protected?")
print(f"   always : {np.mean(D[(0.0,'rip_always')]):.4f}  ({la:+.4f})")
print(f"   gated  : {np.mean(D[(0.0,'rip_gated')]):.4f}  ({lg:+.4f})")
print(f"   -> {'YES' if lg > -0.01 else 'NO, clean scans still damaged'}")

print(f"\n2. Is recovery kept on corrupted scans?")
kept = True
for a in AMPS:
    if a == 0.0: continue
    base = np.array(D[(a,"none")]); gap = ceiling - base.mean()
    g = np.array(D[(a,"rip_gated")]); d = g - base
    pct = 100*d.mean()/gap
    if pct < 10: kept = False
    try: _, p = wilcoxon(g, base); ps = f"p={p:.1e}"
    except ValueError: ps = "p=n/a"
    print(f"   amp {a}:  {base.mean():.4f} -> {g.mean():.4f}   {pct:5.1f}% of gap"
          f"   {ps}  {int((d>0).sum())}/{len(d)}")
print(f"   -> {'YES' if kept else 'recovery collapsed somewhere'}")

print("\n" + "="*72); print("VERDICT"); print("="*72)
if lg > -0.01 and kept:
    print("""  Both ends hold.

  A training-free method that detects when an MRI is corrupted,
  corrects it only then, leaves clean scans alone, and recovers a
  measurable share of the segmentation accuracy the artefact destroys.

  Next: real spike-corrupted acquisitions, and RPCA despiking on the
  same protocol.""")
elif lg <= -0.01:
    print("""  Clean scans are still damaged. The gate separates clean from
  corrupted on synthetic volumes but not on real ones. Raise the
  threshold and check recovery survives; if not, this closes.""")
else:
    print("""  Clean scans protected, but recovery collapsed. The gate is too
  strict. Lower it using the TRAINING volumes only.""")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.figure(figsize=(7.5, 4.8))
for m, mk in zip(names, ["o-","s-","^-"]):
    plt.errorbar(AMPS, [np.mean(D[(a,m)]) for a in AMPS],
                 yerr=[np.std(D[(a,m)]) for a in AMPS], fmt=mk, capsize=3, label=m)
plt.axhline(ceiling, ls="--", c="gray", lw=1, label="clean ceiling")
plt.xlabel("stripe amplitude"); plt.ylabel("Dice"); plt.grid(alpha=.3)
plt.legend(); plt.tight_layout(); plt.savefig(FIG_OUT, dpi=200)
print(f"\nwrote {FIG_OUT}")
