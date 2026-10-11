# Demo: prediction function, API and app

Research prototype, not for clinical use. Live version:
https://huggingface.co/spaces/Pejutopo/gated-topological-mri-demo

| File | Role |
|---|---|
| `gtm_core.py` | `predict(scan)`: brain mask, Xi score, gate (0.367), repair, U-Net. Same settings as the paper |
| `api.py` | FastAPI service: `/predict`, `/samples`, `/result/{id}/slice/{z}` |
| `app.py` | Streamlit app that calls the API |
| `colab_get_samples.py`, `make_stripe_samples.py` | Prepare three held-out scans and their simulated stripes (paper seeds) |
| `precompute.py` | Run the method once on every sample case |
| `export_static.py`, `deploy_static.py`, `static_tpl/` | Static website for free hosting |
| `check_weights.py`, `test_step2.py` | Checks: the checkpoint loads; results match the paper |

Run locally: `pip install torch monai nibabel numpy scipy fastapi uvicorn python-multipart pillow streamlit requests`,
put `unet_clean.pth` in `models/`, then `uvicorn api:app --port 8000` and `streamlit run app.py`.
The model checkpoint is not stored in this repository.
