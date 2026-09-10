# =====================================================================
#  ABLATION  --  v10   DOES Xi ACTUALLY CONTRIBUTE?
#
#  Why this exists
#  ---------------
#  A reviewer put it plainly: nowhere do we test plain frequency
#  subtraction with no Xi at all, or a plain spectral concentration
#  score with no flagged residual. For a single global sinusoid, which
#  is exactly what we simulate, both might work just as well. If they
#  do, Xi contributes nothing and the paper has no method.
#
#  That is the right question and it should have been run first.
#
#  A quick synthetic version came out in our favour:
#
#    repair, RMSE   amp 0.2:  ours 65 vs plain FFT 613
#                   amp 1.0:  ours 381 vs plain FFT 605
#                   amp 2.0:  ours 831 vs plain FFT 728   (we lose here)
#
#    gate           ours: clean max 5.37, dirty min 11.21   SEPARATED
#                   plain spectrum: overlap
#                   SNR ratio: overlap
#                   spectral kurtosis: overlap
#
#  Plain FFT subtraction actively destroys the image at low amplitude,
#  because without Xi to point at the artefact it removes the
#  low-frequency anatomy instead. This script checks whether that holds
#  on real BraTS volumes, measured by Dice.
#
#  ARMS
#    none          no correction
#    ours          Xi-guided frequency subtraction, gated by Xi
#    plain_fft     same subtraction, NO Xi. Ungated.
#    plain_gated   same subtraction, gated by plain spectral concentration
#    ours_ungated  Xi-guided subtraction with no gate (isolates the gate)
#
#  Comparing ours against plain_fft isolates the DETECTION.
#  Comparing ours against ours_ungated isolates the GATE.
#  Comparing plain_gated against ours isolates the SCORE.
#
#  Reuses the model on Drive. About 60 minutes. Writes after every
#  volume, so a disconnect costs at most one volume.
#
#  Paste into ONE Colab cell. Runtime -> Change runtime type -> T4 GPU.
# =====================================================================

import subprocess, sys, os, glob, time, csv, collections, warnings
warnings.filterwarnings("ignore")
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
CSV_OUT   = f"{DRIVE}/ablation_v10.csv"
FIG_OUT   = f"{DRIVE}/fig_ablation_v10.png"
ROOT      = "/content/data/Task01_BrainTumour"

if not os.path.exists(f"{ROOT}/imagesTr"):
    os.makedirs("/content/data", exist_ok=True)
    print("extracting...")
    subprocess.run(["tar", "-xf", TAR, "-C", "/content/data"], check=True)
print(f"data ready, {len(glob.glob(f'{ROOT}/imagesTr/BRATS*.nii.gz'))} volumes")
if not os.path.exists(MODEL_PTH):
    raise SystemExit(f"No model at {MODEL_PTH}. Run the training script first.")

# ---------- settings, identical to the main experiment ----------
N_TRAIN, N_VAL = 60, 30
MODALITY    = 0
PATCH       = (96, 96, 96)
AMPS        = [0.0, 0.2, 0.5, 1.0, 2.0]
REPAIR_FRAC = 0.10
NPEAK       = 8
SEED        = 42
N_GATE_FIT  = 12

torch.manual_seed(SEED); np.random.seed(SEED)
images = sorted(glob.glob(f"{ROOT}/imagesTr/BRATS*.nii.gz"))
labels = sorted(glob.glob(f"{ROOT}/labelsTr/BRATS*.nii.gz"))
gate_files = images[:N_GATE_FIT]
val_files  = [{"image": i, "label": l} for i, l in
              zip(images[N_TRAIN:N_TRAIN+N_VAL], labels[N_TRAIN:N_TRAIN+N_VAL])]
print(f"gate-fit {len(gate_files)} (training split)   val {len(val_files)}")

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

# ---------- with Xi ----------
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

# ---------- without Xi ----------
def score_plain(I, brain):
    """Spectral concentration of the whole volume. No Xi."""
    Kf = np.abs(np.fft.fftn(np.where(brain, I, 0.0))); Kf.ravel()[0] = 0
    return float(np.sort(Kf.ravel())[-6:].sum() / (Kf.sum() + 1e-9) * 1e4)

def repair_plain(I, brain, npeak=NPEAK):
    """Subtract the largest non-DC components. No Xi anywhere."""
    KI = np.fft.fftn(np.where(brain, I, 0.0))
    m = np.abs(KI).ravel().copy(); m[0] = -1
    idx = np.argsort(-m)[:npeak*2]
    c = np.zeros_like(KI); c.ravel()[idx] = KI.ravel()[idx]
    out = I.copy(); out[brain] = (I - np.real(np.fft.ifftn(c)))[brain]
    return out

def normalise(I, brain):
    x = np.zeros_like(I); v = I[brain]
    x[brain] = (v - v.mean()) / (v.std() + 1e-8)
    return x

# ---------- fit BOTH gates on the training split ----------
print("\nFITTING BOTH GATES  (training volumes only)")
cx, dx, cp, dp = [], [], [], []
for i, path in enumerate(gate_files):
    rng = np.random.default_rng(SEED + 500 + i)
    vol = nib.load(path).get_fdata()[..., MODALITY].astype(np.float32)
    b = vol > 0
    cx.append(score_xi(vol, b)); cp.append(score_plain(vol, b))
    for amp in [0.2, 0.5, 1.0, 2.0]:
        I = add_stripe(vol, b, amp, rng)
        dx.append(score_xi(I, b)); dp.append(score_plain(I, b))
    print(f"  [{i+1}/{len(gate_files)}]")

def fit_gate(cl, dy, name):
    cl = np.array(cl); dy = np.array(dy)
    if cl.max() < dy.min():
        g = float(np.sqrt(cl.max()*dy.min())); note = "separated"
    else:
        g = float(np.percentile(cl, 95)); note = "OVERLAP"
    fp = int((cl > g).sum()); fn = int((dy <= g).sum())
    print(f"  {name:<10} clean max {cl.max():9.2f}  dirty min {dy.min():9.2f}  "
          f"gate {g:9.2f}  {note}  fp {fp}/{len(cl)}  fn {fn}/{len(dy)}")
    return g

GATE_XI    = fit_gate(cx, dx, "Xi score")
GATE_PLAIN = fit_gate(cp, dp, "plain")
print()

METHODS = {
    "none":         lambda I, b: I.copy(),
    "ours":         lambda I, b: (I.copy() if score_xi(I, b) <= GATE_XI
                                  else repair_xi(I, b)),
    "ours_ungated": lambda I, b: repair_xi(I, b),
    "plain_fft":    lambda I, b: repair_plain(I, b),
    "plain_gated":  lambda I, b: (I.copy() if score_plain(I, b) <= GATE_PLAIN
                                  else repair_plain(I, b)),
}

# ---------- model ----------
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference

model = UNet(spatial_dims=3, in_channels=1, out_channels=1,
             channels=(16,32,64,128,256), strides=(2,2,2,2),
             num_res_units=2).to(device)
model.load_state_dict(torch.load(MODEL_PTH)); model.eval()
print("model loaded")

def dice_np(pred, gt, eps=1e-6):
    p = pred.astype(bool); g = gt.astype(bool)
    return float((2*(p & g).sum() + eps) / (p.sum() + g.sum() + eps))

def predict(I, brain):
    t = torch.tensor(normalise(I, brain))[None, None].to(device)
    with torch.no_grad():
        out = sliding_window_inference(t, PATCH, 1, model, overlap=0.25)
    return torch.sigmoid(out)[0,0].float().cpu().numpy() > 0.5

# ---------- evaluation ----------
print("\nEVALUATING")
rows = []
t0 = time.time()
for vi, f in enumerate(val_files):
    rng = np.random.default_rng(SEED + 1000 + vi)
    vol   = nib.load(f["image"]).get_fdata()[..., MODALITY].astype(np.float32)
    gt    = nib.load(f["label"]).get_fdata() > 0
    brain = vol > 0
    for amp in AMPS:
        I = add_stripe(vol, brain, amp, rng)
        sx, sp = score_xi(I, brain), score_plain(I, brain)
        for mname, mfn in METHODS.items():
            rep = mfn(I, brain).astype(np.float32)
            rows.append([os.path.basename(f["image"]), amp, mname,
                         dice_np(predict(rep, brain), gt), sx, sp])
    with open(CSV_OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["volume","amp","method","dice","score_xi","score_plain"])
        w.writerows(rows)
    e = (time.time()-t0)/60
    print(f"  [{vi+1}/{len(val_files)}]  {e:.1f}m  eta {e/(vi+1)*(len(val_files)-vi-1):.1f}m")
print(f"  wrote {CSV_OUT}\n")

# ---------- results ----------
D = collections.defaultdict(list)
for v, a, m, d, sx, sp in rows:
    D[(float(a), m)].append(d)
names = list(METHODS)

print("="*88)
print("DICE   mean")
print("="*88)
print(f"{'amp':>6}" + "".join(f"{m:>16}" for m in names))
for a in AMPS:
    print(f"{a:>6}" + "".join(f"{np.mean(D[(a,m)]):16.4f}" for m in names))

ceiling = np.mean(D[(0.0,"none")])
print(f"\nclean ceiling {ceiling:.4f}")

def compare(a, m1, m2):
    x = np.array(D[(a,m1)]); y = np.array(D[(a,m2)]); d = x - y
    try: _, p = wilcoxon(x, y); ps = f"p={p:.1e}"
    except ValueError: ps = "p=n/a"
    tag = "BETTER" if d.mean() > 0 else "worse "
    return f"{tag} {d.mean():+.4f}  {ps}  {int((d>0).sum())}/{len(d)}"

print("\n" + "="*88)
print("QUESTION 1  Does the DETECTION contribute?   ours vs plain_fft")
print("="*88)
for a in AMPS:
    print(f"  amp {a}:  ours {np.mean(D[(a,'ours')]):.4f}  "
          f"plain_fft {np.mean(D[(a,'plain_fft')]):.4f}   {compare(a,'ours','plain_fft')}")

print("\n" + "="*88)
print("QUESTION 2  Does the GATE contribute?   ours vs ours_ungated")
print("="*88)
for a in AMPS:
    print(f"  amp {a}:  ours {np.mean(D[(a,'ours')]):.4f}  "
          f"ungated {np.mean(D[(a,'ours_ungated')]):.4f}   {compare(a,'ours','ours_ungated')}")

print("\n" + "="*88)
print("QUESTION 3  Does the Xi SCORE beat a plain spectral score?")
print("="*88)
for a in AMPS:
    print(f"  amp {a}:  ours {np.mean(D[(a,'ours')]):.4f}  "
          f"plain_gated {np.mean(D[(a,'plain_gated')]):.4f}   {compare(a,'ours','plain_gated')}")

print("\n" + "="*88)
print("VERDICT")
print("="*88)
q1 = all(np.mean(D[(a,"ours")]) >= np.mean(D[(a,"plain_fft")]) for a in AMPS)
q2 = np.mean(D[(0.0,"ours")]) > np.mean(D[(0.0,"ours_ungated")])
q3 = all(np.mean(D[(a,"ours")]) >= np.mean(D[(a,"plain_gated")]) for a in AMPS)
print(f"  detection contributes : {'YES' if q1 else 'NO'}")
print(f"  gate contributes      : {'YES' if q2 else 'NO'}")
print(f"  Xi score beats plain  : {'YES' if q3 else 'NO'}")
if q1 and q2 and q3:
    print("""
  All three components earn their place. This is the ablation the
  paper was missing, and it should become a main table rather than a
  supplementary one.""")
else:
    print("""
  At least one component does not earn its place. Report that plainly
  and remove or reframe the component that does not. A paper with two
  working parts and an honest ablation is better than three claimed
  parts and no ablation.""")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.figure(figsize=(8.5, 5))
for m, mk in zip(names, ["o-","^-","v--","s:","D-."]):
    plt.errorbar(AMPS, [np.mean(D[(a,m)]) for a in AMPS],
                 yerr=[np.std(D[(a,m)]) for a in AMPS], fmt=mk, capsize=3, label=m)
plt.axhline(ceiling, ls="--", c="gray", lw=1, label="clean ceiling")
plt.xlabel("stripe amplitude"); plt.ylabel("Dice"); plt.grid(alpha=.3)
plt.legend(); plt.tight_layout(); plt.savefig(FIG_OUT, dpi=200)
print(f"\nwrote {FIG_OUT}")
