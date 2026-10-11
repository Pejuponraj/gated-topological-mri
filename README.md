# Gated Topological Correction of Stripe Artefacts in Brain MRI

Code for the paper *Gated Topological Correction of Stripe Artefacts in Brain MRI*
(Ponraj Asirwadam Paulraj, Muthu Ramachandran, Kulvinder Panesar; under review).

**Live demo (research prototype, not for clinical use):**
https://huggingface.co/spaces/Pejutopo/gated-topological-mri-demo

---

## The problem

Radiofrequency spikes in k-space produce stripe and herringbone artefacts in MRI.
A segmentation network trained on clean data can lose much of its accuracy on such
scans. In our experiments, a MONAI 3D U-Net for brain tumour segmentation (FLAIR)
has a mean Dice of 0.71 on clean scans, which falls to between 0.16 and 0.57 at the
highest amplitude of each artefact tested.

## The method

A training-free correction that is applied only when a scan-level score says it
is needed.

1. **Detect.** A voxel-level outlier criterion, motivated by hereditary classes in
   generalized topology, flags voxels that lie outside the span of their
   neighbourhood (in implementation, a trimmed rank-filter top-hat).
2. **Score.** The flagged voxels are Fourier transformed. A stripe makes them
   periodic, so the spectrum is concentrated; on a clean scan it is diffuse.
3. **Gate.** If the score exceeds a threshold fitted on training volumes only
   (0.367), the scan is corrected; otherwise it is left untouched.
4. **Repair.** The frequency components identified by the flagged voxels are
   removed from the volume.

## Main results (30 held-out volumes, Medical Segmentation Decathlon Task01)

* **The gate protects clean scans.** It left all 30 clean scans untouched. Without
  the gate, a notch filter lowered clean-scan Dice on 25 of 30.
* **Image-domain stripes.** The gated correction raised Dice at every amplitude,
  by 0.047 to 0.075 (27 to 30 of 30 volumes; Holm-corrected p < 1e-6).
* **k-space spike repeated in every slice.** Dice rose by 0.097 and 0.137 at the
  two larger amplitudes (29 and 28 of 30 volumes).
* **No false alarms on other degradation.** The score fired on none of 180 scans
  with Rician noise, motion-like ghosting or a bias field. A conventional spike
  detector fired on up to 87% of them.
* **Per-slice score.** A per-slice version of the score also detects spikes at
  random positions in random slices (area under the ROC curve 0.999).

## Where it does not help

* For spikes at random positions in random slices, the image-domain correction is
  the wrong tool at high amplitude; robust-PCA despiking in k-space was better.
* A notch filter that only protects low frequencies repaired about as well; the
  contribution is the gating and the controlled comparison, not a new repair.
* The score detects artefacts. It does **not** predict segmentation failure, and a
  conventional spike detector was better for that task.
* All artefacts are **simulated**. The method has **not** been validated on real
  hospital scans; that is the next step.

## Data

Medical Segmentation Decathlon, Task01 BrainTumour (BraTS-derived), CC-BY-SA 4.0.
FLAIR channel only. 60 training volumes, 30 held-out volumes; the gate threshold is
fitted on 12 of the training volumes.

## Repository status

This repository currently holds the earlier experiment scripts. The full
experiment scripts used for the paper's final results and the demo code
(prediction function, FastAPI service, Streamlit app) are being added.

## Licence

Code: MIT. Data: CC-BY-SA 4.0 (Medical Segmentation Decathlon).

**Research prototype. Not for clinical use.**
