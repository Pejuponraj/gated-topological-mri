# Gated Topological Correction of Stripe Artefacts in Brain MRI

Code for the paper *Gated Topological Correction of Stripe Artefacts in Brain MRI*.

Radiofrequency interference and k-space spikes produce coherent stripe artefacts in
MRI. A segmentation model trained on clean data degrades sharply on them: in our
experiments the Dice coefficient of a 3D U-Net falls from 0.70 to 0.22.

This repository contains a training-free correction. A criterion from generalized
topology scores each voxel by how far it lies outside the span of its neighbourhood.
The flagged voxels identify which frequency components carry the artefact, and those
components are subtracted from the volume. A scan-level gate decides whether to
intervene at all, so clean scans are left untouched.

---

## What we found

**Detection works.** Across 30 volumes and four severities, the criterion ranks
corrupted voxels better than the classical median residual and three statistical
controls. It wins on every volume at every severity, p ≈ 1.9e-9.

**Local repair does not.** Replacing flagged voxels with the local median recovers
about 1% of the lost Dice. A stripe corrupts nearly every voxel, only a tenth are
repaired, and the neighbourhood supplying the replacement is itself rippled. We were
replacing bad values with bad ones.

**Frequency subtraction does.** Same detection, different replacement. Recovers 14 to
41% of the lost Dice.

**But it damages clean scans**, 0.712 → 0.573, so it cannot be applied
unconditionally. Most scans in a hospital are clean.

**The gate fixes that.** On 25 held-out volumes it fired on 0 of 25 clean scans and
100 of 100 corrupted ones. Clean-scan Dice is preserved exactly.

| Amplitude | No repair | Ungated | Gated | Gate fired |
|-----------|-----------|---------|-------|------------|
| 0.0 | 0.7039 | 0.5733 | **0.7039** | 0/25 |
| 0.2 | 0.5517 | 0.6114 | **0.6114** | 25/25 |
| 0.5 | 0.4475 | 0.4941 | **0.4941** | 25/25 |
| 1.0 | 0.3734 | 0.4487 | **0.4487** | 25/25 |
| 2.0 | 0.2160 | 0.2836 | **0.2836** | 25/25 |

**The ablation sets the boundary.** Removing the criterion and subtracting the
largest spectral components directly is *better* above an amplitude of roughly 0.5
and *worse* below it. At amplitude 0.2 the detection-free version raises the error
from 266 to 430, worse than leaving the artefact alone. The correction is therefore
scoped to subtle artefacts. That is also the regime that matters clinically: a strong
artefact is visible and prompts a repeat scan, a subtle one is not.

The gate carries no such caveat. None of three simpler scan-level scores separates
clean from corrupted volumes; the proposed score does, with a fivefold margin.

---

## Scripts

Each is self-contained and runs as a single Colab cell, or as a script on a machine
with the data. Each begins with self-tests and refuses to run if they fail.

| Script | Produces | Needs |
|--------|----------|-------|
| `final_stripe_experiment.py` | Detection results, 30 volumes | CPU |
| `colab_v8_complete.py` | Training, gate fitting, Dice, 25 volumes | GPU |
| `colab_v10_ablation.py` | Ablation by Dice | GPU |
| `ablation_cpu.py` | Ablation by image error and gate separation, 10 volumes | CPU |

The GPU scripts save the trained model and every result to Google Drive after each
volume, so a Colab disconnect costs at most one volume and rerunning skips whatever
is already done.

---

## Data

Medical Segmentation Decathlon, Task01_BrainTumour, derived from BraTS. Public and
de-identified, so no ethical approval is required.

```
https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar
```

7,608,266,240 bytes. Download it inside Colab rather than uploading from a laptop.

---

## The criterion

```python
def crit_hyper(I):
    r1 = ndimage.rank_filter(I,  1, size=3)   # 2nd smallest in the neighbourhood
    r2 = ndimage.rank_filter(I, -2, size=3)   # 2nd largest
    return np.maximum(I - r2, r1 - I)         # how far outside the span
```

At neighbourhood extrema this reduces exactly to a local hyperconnectedness
violation. Away from extrema it behaves as a rank-based deviation statistic, so it is
*motivated by* hyperconnectedness rather than a literal test of it. We say so in the
paper rather than overclaim.

It is not a morphological gradient. On a clean step edge the gradient fires and this
criterion is exactly zero, which is the point: it ignores anatomical boundaries and
responds to artefacts.

---

## Reproducing the tables

```
Table 2, 3      final_stripe_experiment.py
Table 4, 5      colab_v8_complete.py
Table 6, 7      ablation_cpu.py
```

All settings are fixed at the top of each script and were chosen before the runs. The
gate threshold is fitted on training volumes only and never on the held-out set.

---

## What this does not do

Artefacts are simulated, not acquired. A sinusoidal ripple is a reasonable model of a
single k-space point corruption, but real acquisitions vary in amplitude, in the
number of corrupted points, and in how they interact with reconstruction.
Confirmation on genuinely spike-corrupted scans is the main outstanding item.

No accuracy comparison against RPCA despiking, the established k-space method. We
attempted it and are not reporting it: our simulated ripple has k-space magnitude
below the low-frequency content of the anatomy, so the sparse component of the
decomposition does not isolate it, and the comparison would misrepresent the method.
Runtime is reported, since it does not depend on the artefact being a fair test.

Robustness through augmented training is not compared, nor are learned correction
networks. Both require retraining, which is the constraint this method is designed
around, but that is a reason to run the comparison rather than to skip it.

---

## Licence

MIT.
