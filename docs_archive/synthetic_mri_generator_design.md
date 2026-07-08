# Synthetic Pelvic MRI Generator Design
## For UT-EndoMRI Auto-Segmentation Augmentation
 
> **Context:** This document covers the generator architecture strategy for improving the RAovSeg pipeline (Liang et al., *Scientific Data* 2025, doi:10.1038/s41597-025-05623-3) by training a diffusion-based adversarial generator to produce synthetic paired image + label data, using Dataset 1 to train the generator and Dataset 2 as the discriminator's style anchor.

---

## 0. Implementation Status (updated June 2026)

The design below is preserved as originally written. Brief notes on what has changed in practice — see `architecture_dataflow_v2.md` Section 0 for the full experiment-by-experiment status and `TRAINING_OVERVIEW.md` for a plain-English summary.

- **Label is now 6-channel, not 5.** A 6th channel `body_other` (inside-body, non-target tissue) was added during Exp 1a to fix noisy grey image edges. The channel order in the implementation is `outside_body, uterus, ov_L, ov_R, em, body_other`. All references to "5-channel" or "5-ch" in Sections 3, 5, and 9 below should be read as 6-channel.
- **Added during Exp 1a, inherited into Exp 1b for parity:** Classifier-Free Guidance (10% dropout, guidance 3.0), EMA weights (decay 0.9999), self-attention reduced to deepest level only (64²) for memory budget, and a fixed-labels resampling fix for the periodic sample grids.
- **Exp 1b SPADE specifics:** hand-built `DiffusionUNetSPADE` (`src/Generator/unet_spade.py`) rather than subclassing MONAI — MONAI's ResBlock `forward()` doesn't expose a label argument. SPADE γ/β heads are zero-initialised so SPADE starts as identity-like; the first Exp 1b run was abandoned because this fix had not yet been applied.
- **Exp status:** 1a done at 80k steps; 1b v2 in progress; 1c (PatchGAN) and Med-DDPM (Exp 2) not started; Phase 2 (D1 generator + D2 discriminator) deferred until Phase 1 picks a winner.

---

## 1. The Core Dataset Problem
 
The RAovSeg pipeline achieves a best DSC of **0.290** for ovary segmentation, trained on only **30 subjects** from Dataset 2 (594 ovary-containing slices). This is far below the inter-rater DSC of **0.48 ± 0.24** measured on Dataset 1's multi-rater subset. The target is to close that gap via synthetic data augmentation.
 
The fundamental constraint is data: only 81 subjects in Dataset 2 total (30 used for training), and Dataset 1 adds 51 more but with different scanner/protocol characteristics. Any generator strategy must work within these numbers.
 
---
 
## 2. The Two-Dataset Training Strategy
 
### Why It Works
 
Using **Dataset 1 to train the generator** and **Dataset 2 to train the discriminator** is a principled form of adversarial domain adaptation integrated directly into the generation pipeline.
 
- **Dataset 1 (generator training):** 51 subjects across 15 sites, 9 scanner models, GE/Philips/Siemens, 1.5T and 3T, T1w + T1w FS + T2w sequences. Provides anatomical diversity — varying ovary shapes, endometrioma presence, uterine morphology — across a wide range of realistic pelvic MRI appearances.
- **Dataset 2 (discriminator anchor):** 81 subjects, single site, Philips Ingenia 1.5T, T2w Fat Suppression as the primary sequence, consistent protocol. The discriminator sees *only* real Dataset 2 T2w FS slices and learns to distinguish them from generator outputs. The adversarial signal then pressures the generator to produce images that are indistinguishable from the target domain, regardless of the domain diversity in its training data.
This is functionally equivalent to what CycleGAN achieves as a post-hoc step, but integrated natively into the generation training loop — a more efficient and principled design.
 
### Key Implementation Detail
 
Train the discriminator **exclusively on T2w FS slices from Dataset 2**, not the other sequences. Fat suppression suppresses the bright peritoneal and subcutaneous fat signal, making the background distinctly darker and ovaries/endometriomas more conspicuous. If T1w or non-fat-suppressed T2w slices contaminate the discriminator training data, the style signal becomes muddled and the generator will not converge to the correct appearance.
 
---
 
## 3. Generator Architecture: Conditional DDPM + SPADE + PatchGAN
 
### Why Conditional, Not Unconditional
 
The downstream task (segmentation training) requires **paired synthetic image + segmentation label** data — every generated image needs a perfect corresponding label. This rules out purely unconditional generation. The generator must be conditioned on segmentation label maps (uterus, left ovary, right ovary, endometrioma, background as separate binary channels) and produce a realistic T2w FS slice that anatomically matches those boundaries.
 
### Recommended Architecture
 
**Generator:** A 2D conditional DDPM with a U-Net denoising backbone. The segmentation label map is encoded and injected at each denoising timestep via **SPADE (Spatially Adaptive Denormalization)** modules in the decoder layers. SPADE is significantly better than simple channel concatenation or cross-attention for preserving fine anatomical boundary correspondence — particularly important for the small, irregular ovary contours in this dataset.
 
**Discriminator:** A **PatchGAN** operating at 70×70 patches. It takes as input the concatenation of the generated (or real) image with the corresponding label map, so it learns to penalise both poor local texture realism and label-image boundary mismatch simultaneously. The discriminator loss feeds into the DDPM training as an auxiliary signal rather than replacing the diffusion objective — this maintains training stability while adding the style-matching pressure from Dataset 2.
 
### Why SPADE Matters Here
 
Standard batch normalisation in generator decoders destroys spatial semantic information because it normalises over the entire feature map. SPADE modulates the normalised activations using learned affine parameters derived from the input label map at each spatial location. The result is that the generated image respects the label boundaries at a pixel level — critical for small structures like ovaries (mean volume 12.2 cc, compared to 220.3 cc for the uterus). A recent cardiac MRI augmentation study (2025) confirmed that SPADE outperforms cross-attention conditioning for multi-label semantic image synthesis tasks in exactly this setting.
 
### Label Map Generation at Inference
 
At inference time, you need label maps to condition the generator on. The practical options are:
1. Use the existing 30 training case label maps with heavy augmentation (elastic deformations, rotations, flips, random scaling of ovary size within physiological range).
2. Train a **separate unconditional label diffusion model** (a simpler DDPM on binary mask stacks, conditioned on slice index) to generate novel anatomical configurations. This is the approach used in **BrainSPADE** and the cardiac SPADE-diffusion paper above.
Option 2 is more powerful and is recommended for maximising diversity, but Option 1 is a sensible starting point.
 
---
 
## 4. The 2D → 3D Problem
 
### The Fundamental Issue
 
Training on 2D slices is computationally necessary given dataset size, but a model trained on independent slices has no mechanism for inter-slice consistency. Generated slices stacked into a volume will exhibit structural discontinuities along the z-axis — a structure may disappear and reappear, edges won't align smoothly, and tissue contrast may shift between adjacent slices.
 
Recent work has identified the root cause precisely: **uncoordinated stochasticity** in the diffusion sampling process. Each slice is denoised from an independently sampled Gaussian noise vector, so the random sampling trajectories diverge across slices, producing incoherent volumes even when each individual slice looks realistic.
 
### 2025 Solution: Inter-Slice Consistent Stochasticity (ISCS)
 
A plug-and-play method (ICLR 2025 submission) that addresses the root cause directly by **synchronising the noise components across adjacent slices** using smooth interpolation during the reverse diffusion process. It aligns sampling trajectories without constraining the generated content too heavily. Key properties:
 
- Works as a post-hoc modification to any pre-trained 2D diffusion model — no retraining required
- Smooth interpolation of noise vectors: `z_i = α * z_shared + (1-α) * z_independent`, where α controls the coherence-diversity tradeoff
- Removes inter-slice intensity mismatches and structural discontinuities
- Compatible with both DDPM and DDIM samplers
This is the most practical path to 3D coherence given your data constraints — train your 2D generator normally, then apply ISCS at inference time when assembling volumes.
 
### Alternative: Slice-Based Latent Diffusion with Positional Encoding (SBLDM)
 
**Kebaili et al. (ISBI 2024, arXiv:2406.05421)** — Directly relevant architecture. Decomposes 3D volumes into 2D slice pairs (image + mask), encodes each with a 2D VAE augmented with a **positional embedder** (slice index embedding), runs the diffusion process in the latent space, and assembles the per-slice latent vectors into a 3D latent volume before decoding. Key contributions:
 
- Simultaneous generation of image + segmentation mask — exactly what you need
- Positional embedder teaches the model that slice 20 follows from slice 19 anatomically
- Works in latent space, not pixel space — much faster sampling than pixel-space DDPM
- Demonstrated improved tumour segmentation in data-scarce regimes (the exact scenario here)
- Lightweight: excludes attention modules, single residual block per resolution — trainable on a single A6000 48GB GPU
This is the closest existing architecture to what you need and should be your primary reference for the 3D coherence component.
 
### Alternative: SLaM-DiMM Coherence Enhancement Module
 
**arXiv:2509.16019 (2025)** — Introduces a **Coherence Enhancement Network (CEn)** as a post-processing module after 2D slice generation. The CEn is a lightweight 3D network trained explicitly to fix inter-slice inconsistencies in stacked synthetic volumes. It takes the full generated volume as input and outputs a coherence-corrected version. The key advantage is that it separates the generation problem from the coherence problem — the main generator doesn't need to change. Demonstrated on BraTS 2025 challenge data.
 
### Alternative: X-Diffusion (2D → Full 3D Reconstruction)
 
**arXiv:2404.19604 (ICLR 2025 submission)** — Reconstructs a detailed 3D MRI volume from as few as a single 2D slice using cross-sectional diffusion. Models MRI as a holistic 3D volume during training rather than a stack of independent 2D slices. Generalises across anatomy (demonstrated on brain tumour + full-body + knee MRI despite training on brain data only). This is the most aggressive approach — useful if you want to generate complete 3D volumes from a small number of generated seed slices — but the most complex to implement and may require more data than available.
 
### Staged Recommendation
 
Given your data constraints and the fact that RAovSeg is a fundamentally 2D pipeline (ResClass + AttUSeg both operate on individual slices), 3D coherence is more a validity argument than a functional requirement for the primary segmentation augmentation goal. The recommended approach is:
 
**Stage 1:** Train the 2D conditional DDPM + SPADE + PatchGAN on 2D slices. Validate that synthetic slices improve RAovSeg DSC. This de-risks the project.
 
**Stage 2:** Apply ISCS at inference time to add inter-slice consistency when assembling volumes, with no retraining cost.
 
**Stage 3 (optional):** Implement the SBLDM positional encoding approach for fully coherent 3D volume generation, useful for demonstrating anatomical plausibility and for any future 3D segmentation methods.
 
---
 
## 5. Dataset and Preprocessing Match
 
All synthetic data must match the preprocessing pipeline used in the paper to be usable for RAovSeg training.
 
### Required Preprocessing
1. Intensity clip: 1st–99th percentile → normalise to [0, 1]
2. Resample: 512 × 512 pixels at 5 mm × 5 mm voxel size
3. Apply the custom ovary intensity enhancement:
   - Highlight voxels in the range [0.22, 0.3] → set to 1
   - Voxels < 0.5 and outside [0.22, 0.3] → unchanged
   - Voxels ≥ 0.5 → invert to (1 − I₀)
4. Target sequence: **T2w Fat Suppression** (Philips Ingenia 1.5T protocol)
### Label Channels (Multi-class mask input to generator)
- Channel 0: Background
- Channel 1: Uterus
- Channel 2: Left ovary
- Channel 3: Right ovary
- Channel 4: Endometrioma (where present)
---
 
## 6. Experimental Evaluation Design
 
### Primary Comparison
| Model | DSC | Notes |
|---|---|---|
| Inter-rater agreement | 0.48 ± 0.24 | Upper bound (Dataset 1, 7-subject subset) |
| RAovSeg (real data only) | 0.290 | Baseline to beat (Dataset 2 test set) |
| nnU-Net (real data only) | 0.272 | Secondary baseline |
| **RAovSeg + synthetic** | TBD | Your target |
 
### Test Sets
- **Primary:** Dataset 2 held-out test cases (same distribution as training — direct improvement measurement)
- **Secondary:** Dataset 1 inter-rater subset (multi-center generalisability — compare against human agreement)
### Important Caveat on Multi-Center Generalisation
Your synthetic data is trained on single-site Dataset 2 appearance. Dataset 1 is multi-center (15 sites, 9 scanner models, 1.5T and 3T). Performance drop on Dataset 1 test subjects is expected and should be framed as a domain generalisation limitation rather than a failure. Optionally, CycleGAN-based domain adaptation from Dataset 2 to Dataset 1 scanner characteristics could be added as a separate component.
 
---
 
## 7. Key Papers Reference Table
 
| Paper | Relevance | Link |
|---|---|---|
| **SBLDM** — Kebaili et al., ISBI 2024 | Slice-based LDM with positional encoding, simultaneous image + mask generation | arXiv:2406.05421 |
| **ISCS** — Kwon & Ye, ICLR 2025 | Plug-and-play inter-slice noise synchronisation for 3D coherence | openreview.net/pdf/a384e5 |
| **SLaM-DiMM** — 2025 | Coherence Enhancement Network as post-processing module | arXiv:2509.16019 |
| **X-Diffusion** — Hamdi et al., ICLR 2025 | Full 3D volume reconstruction from single 2D slice | arXiv:2404.19604 |
| **Med-DDPM** — Dorjsembe et al., IEEE JBHI 2024 | 3D semantic MRI synthesis with segmentation-conditioned DDPM | doi:10.1109/JBHI.2024.3385504 |
| **MRGen** — Wu et al., ICCV 2025 | Diffusion data engine conditioned on text + segmentation masks for underrepresented MRI modalities | arXiv:2412.04106 |
| **Multi-modal LDM** — Kebaili et al., Med. Image Anal. 2025 | Slice-by-slice LDM with Latent Aggregation module for multi-modal image + mask generation | doi:10.1016/j.media.2025... |
| **Cardiac SPADE-Diffusion** — MDPI Bioengineering 2025 | Two-stage SPADE + diffusion for multi-label conditioned cardiac MRI augmentation | doi:10.3390/bioeng12080812 |
| **BrainSPADE** — LDM label generator + SPADE image generator | Synthetic brain MRI with optional pathology, any scanner style | MICCAI DGM4MICCAI workshop |
| **SPADE** — Park et al., CVPR 2019 | Spatially adaptive denormalisation for label-to-image synthesis | arXiv:1903.07291 |
| **SynSeg-Net** — Huo et al., IEEE TMI 2019 | Joint synthesis + segmentation via adversarial training | doi:10.1109/TMI.2018.2870574 |
| **PelvicDiff** — ScienceDirect 2025 | Diffusion model specifically for pelvic MRI synthesis (Mamba + AKGB) | doi:10.1016/j.eswa.2025... |
| **3D Medical Diffusion** — Friedrich et al. | 3D DDPM for medical volumes with breast/knee MRI including anisotropic resolution | doi:10.1038/s41598-023-34341-2 |
| **Prostate Zone-Conditioned Diffusion** — Bashkanov et al., DGM4MICCAI 2024 | Zone-conditioned 3D prostate MRI generation | Springer LNCS 15224 |
 
---
 
## 8. Tooling & Implementation Resources
 
| Tool | Purpose | URL |
|---|---|---|
| **MONAI Generative** | Off-the-shelf DDPM, LDM, VQVAE for medical imaging in PyTorch | monai.io/research/generative-models |
| **RAovSeg codebase** | Baseline pipeline to extend | github.com/xlianguth/RAovSeg |
| **UT-EndoMRI Dataset** | Both datasets (NIfTI, Zenodo) | doi:10.5281/zenodo.15750762 |
| **Medical Diffusion** | DDPM adapted for 3D medical volumes | github.com/FirasGit/medicaldiffusion |
| **SBLDM** | Slice-based LDM with positional encoder + mask generation | (code linked in arXiv:2406.05421) |
| **dcm2niix** | DICOM → NIfTI conversion | github.com/rordenlab/dcm2niix |
| **SimpleITK** | N4 bias correction, registration, NIfTI I/O | simpleitk.org |
| **nnU-Net** | Downstream segmentation baseline | github.com/MIC-DKFZ/nnUNet |
 
---
 
## 9. Training Loop Summary
 
```
Dataset 1 (all sequences, multi-site)
    │
    ├─ Preprocessing: clip, normalise, resample to 512×512 @ 5mm
    ├─ Extract 2D axial slices + corresponding label maps
    └──► Generator training input
 
Dataset 2 (T2w FS only, single site)
    │
    ├─ Same preprocessing pipeline
    ├─ Extract 2D axial slices
    └──► Discriminator real-sample input
 
Generator (DDPM-UNet + SPADE decoder)
    ├─ Input: label map (5-channel binary) + noise
    ├─ Conditioning: SPADE at each decoder resolution
    ├─ Loss: DDPM diffusion objective + λ * PatchGAN adversarial loss
    └──► Output: synthetic T2w FS axial slice
 
Discriminator (PatchGAN)
    ├─ Input: [image | label map] concatenated (real from D2 or fake from G)
    └──► Loss: binary cross-entropy on 70×70 patches
 
At inference:
    1. Generate label maps (augmented real or label diffusion model)
    2. Run generator to produce synthetic slices
    3. Apply ISCS noise synchronisation across slice stack
    4. Apply ovary intensity enhancement preprocessing
    5. Stack into volumes → add to RAovSeg training set
```
 
---
 
*Last updated: February 2026. Document covers architecture design, dataset strategy, 2D→3D coherence solutions, and relevant literature through early 2026.*