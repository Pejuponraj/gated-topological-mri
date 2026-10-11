# =====================================================================
#  Run ONCE in Colab (CPU is fine). Copies 3 held-out volumes to Drive
#  as small files, so you can download them to your laptop.
#  These are volumes 60, 61, 62 of the sorted list = "vol 0, 1, 2" in v11.
#  Only the FLAIR channel is kept (the model uses only FLAIR).
#  Needs on Drive: Task01_BrainTumour.tar
# =====================================================================
import os, glob, subprocess
import numpy as np, nibabel as nib
from google.colab import drive
drive.mount("/content/drive")
TAR  = "/content/drive/MyDrive/Task01_BrainTumour.tar"
ROOT = "/content/data/Task01_BrainTumour"
OUT  = "/content/drive/MyDrive/gtm_samples"
if not os.path.exists(f"{ROOT}/imagesTr"):
    os.makedirs("/content/data", exist_ok=True)
    print("extracting, about 10 minutes...")
    subprocess.run(["tar", "-xf", TAR, "-C", "/content/data"], check=True)
images = sorted(glob.glob(f"{ROOT}/imagesTr/BRATS*.nii.gz"))
labels = sorted(glob.glob(f"{ROOT}/labelsTr/BRATS*.nii.gz"))
assert len(images) == 484, len(images)
os.makedirs(OUT, exist_ok=True)
for vi in range(3):
    ip, lp = images[60 + vi], labels[60 + vi]
    name = os.path.basename(ip).replace(".nii.gz", "")
    img = nib.load(ip)
    flair = img.get_fdata()[..., 0].astype(np.float32)
    nib.save(nib.Nifti1Image(flair, img.affine), f"{OUT}/vol{vi}_{name}_flair.nii.gz")
    lab = nib.load(lp)
    nib.save(nib.Nifti1Image((lab.get_fdata() > 0).astype(np.uint8), lab.affine),
             f"{OUT}/vol{vi}_{name}_label.nii.gz")
    print(f"saved vol{vi}: {name}  shape {flair.shape}")
print("Done. Download the folder 'gtm_samples' from Drive.")
