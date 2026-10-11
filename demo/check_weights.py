"""
Step 1 check: load the trained U-Net weights on CPU and run one test.
Run from the project folder:   python check_weights.py
"""
import os, sys, hashlib
import torch
from monai.networks.nets import UNet

WEIGHTS = os.path.join("models", "unet_clean.pth")

# Same network as in training (colab_v8_complete.py, lines 202-204).
def build_model():
    return UNet(spatial_dims=3, in_channels=1, out_channels=1,
                channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
                num_res_units=2)

if not os.path.exists(WEIGHTS):
    sys.exit(f"Not found: {WEIGHTS}. Put unet_clean.pth in the 'models' folder.")

size_mb = os.path.getsize(WEIGHTS) / 1e6
sha = hashlib.sha256(open(WEIGHTS, "rb").read()).hexdigest()[:12]
print(f"file: {WEIGHTS}  ({size_mb:.1f} MB, sha256 {sha})")

model = build_model()
state = torch.load(WEIGHTS, map_location="cpu")   # saved on a GPU; load on CPU
model.load_state_dict(state)                        # fails loudly if anything differs
model.eval()
n = sum(p.numel() for p in model.parameters())
print(f"weights loaded into the U-Net: {n:,} parameters")

with torch.no_grad():
    out = model(torch.zeros(1, 1, 96, 96, 96))
print(f"test forward pass OK, output shape {tuple(out.shape)}")
print("STEP 1 PASSED")
