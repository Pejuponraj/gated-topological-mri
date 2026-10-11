"""
api.py  -  FastAPI service for the demo.
Run:   uvicorn api:app --port 8000
Docs:  http://127.0.0.1:8000/docs   (try the endpoints in the browser)

Endpoints
  GET  /health                      -> is the service up, which threshold
  GET  /samples                     -> the pre-computed sample cases (instant)
  GET  /samples/{id}                -> summary of one sample case
  POST /predict  (upload .nii.gz)   -> run the method on any scan (20-60 s on CPU)
  GET  /result/{id}/slice/{z}       -> four PNG images for one axial slice
"""
import io, os, glob, json, hashlib, base64, tempfile
from collections import OrderedDict
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from gtm_core import predict, GATE, dice, load_flair

DISCLAIMER = "Research prototype - not for clinical use."
DATA_SOURCE = ("Medical Segmentation Decathlon, Task01 BrainTumour "
               "(BraTS-derived), CC-BY-SA 4.0. Artefacts are simulated.")
HERE = os.path.dirname(os.path.abspath(__file__))
PRECOMP = os.path.join(HERE, "precomputed")

app = FastAPI(title="Gated Topological Correction - demo API",
              description=DISCLAIMER + " " + DATA_SOURCE)

_CACHE = OrderedDict()            # uploaded scans: id -> result (keeps the last few)
_MAX = 6
_SAMPLES = {}                     # pre-computed sample cases: id -> result (always kept)

# ---------------- building a result ----------------
def result_id(data, ldata=b""):
    return hashlib.sha256(data + ldata).hexdigest()[:16]

def _to_u8(vol, lo, hi):
    """Display copy: the same grey-level window for before and after."""
    return (np.clip((vol - lo) / (hi - lo + 1e-9), 0, 1) * 255).astype(np.uint8)

def _summary(r):
    vb = r["seg_before"].sum(axis=(0, 1)); va = r["seg_after"].sum(axis=(0, 1))
    best = int(np.argmax(np.maximum(vb, va))) if (vb.max() + va.max()) > 0 \
        else r["image"].shape[2] // 2
    return {"score": round(r["score"], 4), "threshold": r["threshold"],
            "artefact_detected": r["artefact_detected"],
            "correction_applied": r["artefact_detected"],
            "n_slices": int(r["image"].shape[2]), "best_slice": best,
            "tumour_voxels_before": int(r["seg_before"].sum()),
            "tumour_voxels_after": int(r["seg_after"].sum()),
            "mask_method": r["mask_method"],
            "disclaimer": DISCLAIMER, "data_source": DATA_SOURCE}

def process(data, ldata=b"", suffix=".nii.gz"):
    """Run the method on the bytes of a scan (and optional expert label)."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data); path = f.name
    try:
        r = predict(path)
    finally:
        os.remove(path)
    b = r["brain"]
    lo, hi = np.percentile(r["image"][b], [1, 99]) if b.any() else (0.0, 1.0)
    res = {"image": _to_u8(r["image"], lo, hi),
           "corrected": _to_u8(r["corrected"], lo, hi),
           "seg_before": r["seg_before"], "seg_after": r["seg_after"],
           "summary": _summary(r)}
    if ldata:
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
            f.write(ldata); lpath = f.name
        try:
            gt = load_flair(lpath)[0] > 0
        finally:
            os.remove(lpath)
        if gt.shape != r["image"].shape:
            raise ValueError("label and scan have different shapes")
        res["summary"]["dice_before"] = round(dice(r["seg_before"], gt), 4)
        res["summary"]["dice_after"] = round(dice(r["seg_after"], gt), 4)
    return res

# ---------------- saving and loading pre-computed cases ----------------
def save_result(res, path):
    shape = res["image"].shape
    np.savez_compressed(path, image=res["image"], corrected=res["corrected"],
                        seg_before=np.packbits(res["seg_before"]),
                        seg_after=np.packbits(res["seg_after"]),
                        shape=np.array(shape),
                        summary=np.array(json.dumps(res["summary"])))

def load_result(path):
    z = np.load(path)
    shape = tuple(int(s) for s in z["shape"]); n = int(np.prod(shape))
    unpack = lambda k: np.unpackbits(z[k])[:n].reshape(shape).astype(bool)
    return {"image": z["image"], "corrected": z["corrected"],
            "seg_before": unpack("seg_before"), "seg_after": unpack("seg_after"),
            "summary": json.loads(str(z["summary"]))}

def _load_samples():
    for p in sorted(glob.glob(os.path.join(PRECOMP, "*.npz"))):
        try:
            _SAMPLES[os.path.basename(p)[:-4]] = load_result(p)
        except Exception as e:
            print(f"could not load {p}: {e}")
    print(f"loaded {len(_SAMPLES)} pre-computed sample cases")
_load_samples()

def _get(rid):
    if rid in _SAMPLES:
        return _SAMPLES[rid]
    if rid in _CACHE:
        return _CACHE[rid]
    raise HTTPException(404, "Unknown id.")

# ---------------- images ----------------
def _png(u8, mask=None):
    g = u8.astype(np.float32) / 255.0
    rgb = np.stack([g, g, g], -1)
    if mask is not None:
        m = mask.astype(bool)
        rgb[m] = 0.55 * rgb[m] + 0.45 * np.array([1.0, 0.85, 0.0])
    img = Image.fromarray((np.rot90(rgb) * 255).astype(np.uint8))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ---------------- endpoints ----------------
@app.get("/health")
def health():
    return {"status": "ok", "threshold": GATE, "samples": len(_SAMPLES),
            "disclaimer": DISCLAIMER}

@app.get("/samples")
def samples():
    items = [{"id": k, "name": v["summary"].get("case_name", k),
              "order": v["summary"].get("order", 0)} for k, v in _SAMPLES.items()]
    return sorted(items, key=lambda x: x["order"])

@app.get("/samples/{rid}")
def sample(rid: str):
    if rid not in _SAMPLES:
        raise HTTPException(404, "Unknown sample id.")
    return {"id": rid, **_SAMPLES[rid]["summary"]}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...),
                           label: UploadFile = File(None)):
    """file: the scan. label (optional): the expert tumour annotation; if given,
    Dice before and after correction is also returned."""
    if not file.filename.endswith((".nii.gz", ".nii")):
        raise HTTPException(400, "Please upload a .nii.gz file.")
    data = await file.read()
    ldata = await label.read() if label is not None else b""
    rid = result_id(data, ldata)
    if rid not in _SAMPLES and rid not in _CACHE:
        try:
            res = process(data, ldata, ".nii.gz" if file.filename.endswith(".gz") else ".nii")
        except Exception as e:
            raise HTTPException(400, f"Could not process this file: {e}")
        _CACHE[rid] = res; _CACHE.move_to_end(rid)
        while len(_CACHE) > _MAX:
            _CACHE.popitem(last=False)
    return {"id": rid, **_get(rid)["summary"]}

@app.get("/result/{rid}/slice/{z}")
def slice_endpoint(rid: str, z: int):
    c = _get(rid); n = c["image"].shape[2]
    if not 0 <= z < n:
        raise HTTPException(400, f"z must be between 0 and {n-1}")
    return {"z": z,
            "input":          _png(c["image"][:, :, z]),
            "corrected":      _png(c["corrected"][:, :, z]),
            "overlay_before": _png(c["image"][:, :, z], c["seg_before"][:, :, z]),
            "overlay_after":  _png(c["corrected"][:, :, z], c["seg_after"][:, :, z])}
