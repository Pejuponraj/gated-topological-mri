"""
gtm_core.py  -  the prediction function for the demo.

One call:   result = predict("scan.nii.gz")
Input : a brain MRI file (.nii.gz). Either the 4-channel Medical Segmentation
        Decathlon file (FLAIR is channel 0) or a single FLAIR volume.
Output: a dict with the scan, the corrected scan, the tumour mask before and
        after correction, the artefact score, the threshold and the flag.

Everything here is copied from the experiment code, unchanged:
  - network and checkpoint      : colab_v8_complete.py (lines 202-207)
  - normalise / segment         : colab_v11_all.py (as v8)
  - Xi, flagged residual, score : colab_v11_all.py (score x 1e4)
  - repair                      : colab_v11_all.py (repair_xi_from_res)
  - gate threshold 0.367        : colab_v11_all.py, fitted on 12 training volumes
No reorientation and no resampling: the training code did not use them.
"""
import os
import numpy as np
import nibabel as nib
import torch
from scipy import ndimage
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference

# ---------------- fixed settings (same as the paper) ----------------
MODALITY    = 0                 # FLAIR channel in the MSD file
PATCH       = (96, 96, 96)
OVERLAP     = 0.25
REPAIR_FRAC = 0.10              # top 10% of voxels by Xi are flagged
NPEAK       = 8                 # 8 conjugate pairs removed
GATE        = 0.367             # v11 threshold (training volumes only)
WEIGHTS     = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "models", "unet_clean.pth")

# ---------------- model (loaded once) ----------------
_MODEL = None
def get_model():
    global _MODEL
    if _MODEL is None:
        m = UNet(spatial_dims=3, in_channels=1, out_channels=1,
                 channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
                 num_res_units=2)
        m.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
        m.eval()
        _MODEL = m
    return _MODEL

# ---------------- loading and brain mask ----------------
def load_flair(path):
    img = nib.load(path)
    data = img.get_fdata()
    if data.ndim == 4:
        data = data[..., MODALITY]
    if data.ndim != 3:
        raise ValueError(f"expected a 3D or 4D volume, got shape {data.shape}")
    # C order matters: nibabel returns Fortran-ordered arrays, and the repair
    # writes into a flattened view, which silently fails on Fortran order.
    return np.ascontiguousarray(data, dtype=np.float32), img.affine

def _otsu(x):
    h, e = np.histogram(x, 256); c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h*c) / np.maximum(w0, 1)
    m1 = (np.sum(h*c) - np.cumsum(h*c)) / np.maximum(w1, 1)
    return c[np.argmax(w0*w1*(m0-m1)**2)]

def brain_mask(I):
    """Skull-stripped scans (as in the MSD data) have background exactly 0,
    and the mask is I != 0, as in the paper. If the background is not zero,
    the Otsu mask used in the paper for k-space artefacts is used instead."""
    m = I != 0
    if m.mean() < 0.9:
        return m, "nonzero"
    s = ndimage.gaussian_filter(I, 2)
    m = ndimage.binary_fill_holes(s > _otsu(s))
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    return m, "otsu"

# ---------------- Xi criterion, score, repair (unchanged) ----------------
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

def score_from_res(res):
    Kf = np.abs(np.fft.fftn(res))
    return float(np.sort(Kf.ravel())[-6:].sum() / (Kf.sum() + 1e-9) * 1e4)

def repair_from_res(I, brain, res, npeak=NPEAK):
    Kr = np.abs(np.fft.fftn(res)).ravel().copy(); Kr[0] = -1
    idx = np.argsort(-Kr)[:npeak * 2]
    KI = np.fft.fftn(np.where(brain, I, 0.0))
    pos = np.unravel_index(idx, KI.shape)          # safe for any memory order
    c = np.zeros_like(KI); c[pos] = KI[pos]
    out = I.copy(); out[brain] = (I - np.real(np.fft.ifftn(c)))[brain]
    return out

# ---------------- segmentation (unchanged) ----------------
def normalise(I, brain):
    x = np.zeros_like(I); v = I[brain]
    x[brain] = (v - v.mean()) / (v.std() + 1e-8)
    return x

@torch.no_grad()
def segment(I, brain):
    x = torch.from_numpy(normalise(I, brain))[None, None]
    p = sliding_window_inference(x, PATCH, 1, get_model(), overlap=OVERLAP)
    return torch.sigmoid(p)[0, 0].float().numpy() > 0.5

# ---------------- the one function the API and app use ----------------
def predict(source):
    """source: a path to .nii.gz, or a 3D numpy array (FLAIR)."""
    if isinstance(source, str):
        I, affine = load_flair(source)
    else:
        I, affine = np.ascontiguousarray(source, dtype=np.float32), np.eye(4)
    brain, mask_method = brain_mask(I)
    res = flagged_residual(I, brain)
    score = score_from_res(res)
    detected = score > GATE
    seg_before = segment(I, brain)
    if detected:
        corrected = repair_from_res(I, brain, res)
        seg_after = segment(corrected, brain)
    else:
        corrected, seg_after = I, seg_before          # clean scan: untouched
    return {
        "image": I, "corrected": corrected, "affine": affine, "brain": brain,
        "seg_before": seg_before, "seg_after": seg_after,
        "score": score, "threshold": GATE, "artefact_detected": bool(detected),
        "mask_method": mask_method,
    }

def dice(a, b, eps=1e-6):
    a = a.astype(bool); b = b.astype(bool)
    return float((2 * (a & b).sum() + eps) / (a.sum() + b.sum() + eps))
