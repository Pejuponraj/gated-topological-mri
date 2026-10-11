---
title: Gated Topological Correction - MRI demo
emoji: 🧠
colorFrom: gray
colorTo: red
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Stripe artefact detection and correction in brain MRI
---

# Gated Topological Correction of Stripe Artefacts in Brain MRI — demo

**Research prototype — not for clinical use. Not validated on real hospital data.**

Pre-computed results for three held-out volumes of the Medical Segmentation
Decathlon, Task01 BrainTumour (BraTS-derived, CC-BY-SA 4.0). Stripe artefacts are
**simulated**. Every image was produced with the same code and settings as the
paper (MONAI 3D U-Net; gate threshold 0.367, fitted on 12 training volumes).

Paper: *Gated Topological Correction of Stripe Artefacts in Brain MRI*
(Paulraj, Ramachandran, Panesar; under review).
Code: https://github.com/Pejuponraj/gated-topological-mri (MIT licence).
