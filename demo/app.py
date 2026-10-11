"""
app.py  -  Streamlit demo. It calls the API (api.py); it does not run the model.
Run (second terminal, API already running):   streamlit run app.py
"""
import os, base64
import requests
import streamlit as st

API = os.environ.get("GTM_API", "http://127.0.0.1:8000")
IMG_W = 330          # image width in pixels; change to make images bigger/smaller

st.set_page_config(page_title="Gated Topological Correction - demo",
                   layout="wide", page_icon="🧠")

st.markdown("""
<style>
.block-container {padding-top:2.2rem;padding-bottom:1rem}
.banner {background:#fff4e5;border:1px solid #f0b44c;color:#5c3d00;
         padding:6px 12px;border-radius:8px;font-weight:600;margin-bottom:4px;font-size:.92rem}
.flag {border-radius:14px;padding:16px 22px;margin:4px 0 6px 0;color:white}
.flag h2 {margin:0 0 4px 0;font-size:1.6rem;color:white;padding:0}
.flag p {margin:0;font-size:1rem;opacity:.95}
.red {background:#c62828} .green {background:#2e7d32}
.small {font-size:.86rem;color:inherit;opacity:.8}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="banner">⚠️ Research prototype — not for clinical use. '
            'Not validated on real hospital data.</div>', unsafe_allow_html=True)
st.markdown("### Gated Topological Correction of Stripe Artefacts in Brain MRI")
st.markdown("**Dr Ponraj Asirwadam Paulraj · Prof. Muthu Ramachandran**")
st.caption("A 3D U-Net segments the tumour. A topological score decides whether "
           "the scan has a stripe artefact; only then is the correction applied.")

# ---------------- talk to the API ----------------
def api_get(path, timeout=60):
    r = requests.get(f"{API}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False, ttl=600)
def list_samples():
    return api_get("/samples")

@st.cache_data(show_spinner=False)
def get_summary(rid):
    return api_get(f"/samples/{rid}")

@st.cache_data(show_spinner=False, max_entries=400)
def get_slice(rid, z):
    return api_get(f"/result/{rid}/slice/{z}")

try:
    with st.spinner("Starting... (the first visit can take up to a minute)"):
        samples = list_samples()
except requests.exceptions.RequestException:
    st.error(f"Cannot reach the API at {API}. Start it with: uvicorn api:app --port 8000")
    st.stop()
if not samples:
    st.error("No pre-computed cases found. Run:  python precompute.py")
    st.stop()

# ---------------- sidebar: choose a case ----------------
with st.sidebar:
    st.header("1. Choose a scan")
    names = [c["name"] for c in samples]
    choice = st.radio("Pre-loaded cases", names, index=0)
    st.divider()
    st.markdown("**Data source**  \nMedical Segmentation Decathlon, Task01 "
                "BrainTumour (BraTS-derived), CC-BY-SA 4.0. FLAIR channel. "
                "Stripe artefacts are **simulated** on these scans.")
    st.markdown("<span class='small'>Paper: Gated Topological Correction of "
                "Stripe Artefacts in Brain MRI (Paulraj, Ramachandran, Panesar; "
                "under review).</span>", unsafe_allow_html=True)

res = get_summary(samples[names.index(choice)]["id"])

left_col, right_col = st.columns([1, 2], gap="large")

# ---------------- left: the flag and the numbers ----------------
with left_col:
    if res["artefact_detected"]:
        st.markdown(f"""<div class="flag red"><h2>🔴 Stripe artefact detected</h2>
        <p>Topological correction applied before segmentation.
        Artefact score {res['score']:.2f} &gt; threshold {res['threshold']:.3f}.</p></div>""",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="flag green"><h2>🟢 No stripe artefact detected</h2>
        <p>Scan used as acquired — no correction applied.
        Artefact score {res['score']:.2f} ≤ threshold {res['threshold']:.3f}.</p></div>""",
                    unsafe_allow_html=True)
    st.markdown("<span class='small'>The flag says whether a stripe artefact was found "
                "and corrected. It does not say whether the AI segmentation is correct. "
                "Threshold fitted on 12 training volumes; on 30 held-out volumes the flag "
                "fired on no clean scan and on every simulated stripe.</span>",
                unsafe_allow_html=True)

    vb, va = res["tumour_voxels_before"] / 1000, res["tumour_voxels_after"] / 1000
    change = f" ({va - vb:+.1f})" if res["artefact_detected"] else " (unchanged)"
    st.markdown(f"**Tumour volume**  \n<span style='font-size:1.5rem'>{vb:.1f} → "
                f"{va:.1f} mL</span><span class='small'>{change}</span>",
                unsafe_allow_html=True)
    if "dice_before" in res:
        d0, d1 = res["dice_before"], res["dice_after"]
        st.metric("Agreement with expert label (Dice)", f"{d1:.2f}",
                  None if not res["artefact_detected"]
                  else f"{d1 - d0:+.2f} vs before ({d0:.2f})")

# ---------------- right: the slice viewer ----------------
with right_col:
    cz, ct = st.columns([3, 1])
    z = cz.slider("2. Move through the scan (axial slice)",
                  0, res["n_slices"] - 1, res["best_slice"])
    show_seg = ct.toggle("Tumour", value=True, help="Show the tumour segmentation in yellow")
    s = get_slice(res["id"], z)
    img = lambda k: base64.b64decode(s[k])
    before, after = st.columns(2)
    with before:
        st.markdown("**Before correction** — input scan")
        st.image(img("overlay_before" if show_seg else "input"), width=IMG_W)
    with after:
        st.markdown("**After correction**" if res["artefact_detected"]
                    else "**After** — unchanged (no artefact found)")
        st.image(img("overlay_after" if show_seg else "corrected"), width=IMG_W)

with st.expander("Roadmap: from public data to hospital scans (not yet done)"):
    st.markdown(
        "This demo uses the public BraTS-derived data, already converted, "
        "skull-stripped and aligned. Hospital scans need the standard steps first:\n\n"
        "**DICOM → NIfTI** (dcm2niix) → **select FLAIR** → **skull stripping** "
        "(e.g. HD-BET or SynthStrip) → **align to 1 mm** (BraTS pre-processing) → "
        "**our gated topological correction** → **any segmentation model** "
        "(this U-Net, or a stronger one such as nnU-Net).\n\n"
        "The correction works on the image before the model, so it does not depend "
        "on this U-Net. On hospital data the artefact threshold would be re-fitted on "
        "clean local scans with the same rule. **None of this has been validated on "
        "hospital scans yet; that is the planned next step with clinical partners.**")

st.divider()
st.markdown("<span class='small'>Research prototype — not for clinical use. "
            "All artefacts shown are simulated; the method has not been validated "
            "on real hospital data. Model: MONAI 3D U-Net trained on 60 MSD volumes "
            "(FLAIR). Data: Medical Segmentation Decathlon Task01, CC-BY-SA 4.0.</span>",
            unsafe_allow_html=True)
