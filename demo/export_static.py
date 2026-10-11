"""
export_static.py  -  turn the pre-computed cases into a static website
(HTML + images), which Hugging Face hosts for free as a "Static Space".
Run after precompute.py:   python export_static.py
Output: the folder 'site' (index.html, data.json, img/...).
No model runs online: every image was computed on this laptop with the
same code and settings as the paper.
"""
import os, json, shutil, glob
import numpy as np
from PIL import Image
import api

OUT = "site"
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(os.path.join(OUT, "img"))

def save(u8, mask, path):
    g = u8.astype(np.float32) / 255.0
    rgb = np.stack([g, g, g], -1)
    if mask is not None and mask.any():
        rgb[mask] = 0.55 * rgb[mask] + 0.45 * np.array([1.0, 0.85, 0.0])
    Image.fromarray((np.rot90(rgb) * 255).astype(np.uint8)).save(path, "WEBP", quality=85)

cases = []
for rid, c in sorted(api._SAMPLES.items(), key=lambda kv: kv[1]["summary"].get("order", 0)):
    s = c["summary"]; n = c["image"].shape[2]
    d = os.path.join(OUT, "img", rid); os.makedirs(d)
    fixed = s["artefact_detected"]
    for z in range(n):
        save(c["image"][:, :, z], None, f"{d}/{z}_in.webp")
        save(c["image"][:, :, z], c["seg_before"][:, :, z], f"{d}/{z}_ob.webp")
        if fixed:                                  # clean scans: after == before
            save(c["corrected"][:, :, z], None, f"{d}/{z}_co.webp")
            save(c["corrected"][:, :, z], c["seg_after"][:, :, z], f"{d}/{z}_oa.webp")
    keep = ["case_name", "score", "threshold", "artefact_detected", "n_slices",
            "best_slice", "tumour_voxels_before", "tumour_voxels_after",
            "dice_before", "dice_after"]
    cases.append({"id": rid, **{k: s[k] for k in keep if k in s}})
    print(f"exported {s['case_name']}")

json.dump(cases, open(os.path.join(OUT, "data.json"), "w"), indent=1)
shutil.copy(os.path.join("static_tpl", "index.html"), OUT)
shutil.copy(os.path.join("static_tpl", "README.md"), OUT)
size = sum(os.path.getsize(p) for p in glob.glob(f"{OUT}/**/*", recursive=True) if os.path.isfile(p))
print(f"done: {len(cases)} cases, {size/1e6:.0f} MB in '{OUT}'")
