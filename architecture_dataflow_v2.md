# Architecture Dataflow & Ablation Experiment Plan (v2)
## Synthetic Pelvic MRI Generator — Option A: Two-Phase Design
## Every data decision documented and justified

---

## 0. IMPLEMENTATION STATUS (updated June 2026)

The plan below is preserved as originally written. The notes here capture
deltas between the plan and what has actually been built. Companion docs:
`TRAINING_OVERVIEW.md` (plain-English summary) and
`src/Generator/EXP1A_NOTES.md` / `EXP1B_NOTES.md` (detailed bug log).

**Experiment status:**

| Exp | Status | Notes |
|---|---|---|
| 0a RAovSeg baseline | Done | Reproduced separately under `RAovSeg/` |
| 0b nnU-Net baseline | Not started | Deferred until Phase 1 ablation complete |
| 1a DDPM + concat   | **Done @ 80k steps** | `1a/current/`, see EXP1A_NOTES + RESULTS_2x2 |
| 1b DDPM + SPADE    | **Done @ 80k steps** | `1b/current/`, see EXP1B_SUMMARY + RESULTS_2x2 |
| **1c-concat** DDPM + concat + PatchGAN | **Done @ 100k steps** | `1c/concat/`, see EXP1C_SUMMARY |
| **1c-spade** DDPM + SPADE + PatchGAN | **Done @ 100k steps** | `1c/spade/`, see EXP1C_SUMMARY |
| 2  Med-DDPM (3D)   | Not started | Deprioritised — 1c results show 2D + adversarial gives publication-worthy story |
| 3  Phase 2 cross-domain | Not started | Contingent on Exp 4 outcome (see NEXT_STEPS.md) |
| **5  Image quality metrics** | **Done for all 4 variants** | FID, hist_KL, LPIPS in each variant's `quality.json`; aggregated in `master_metrics.csv` |
| **5b Interpretability metrics** | **Done for all 4 variants** | CLR, OSI, sparsity per sample in `explain/` JSONs |
| **5c Radiologist review set** | **Done for all 4 variants** | 50 matched-anatomy samples per variant in `radiologist_review/` |
| 4  Titration (RAovSeg DSC) | **Next major step** | Train RAovSeg with each variant as augmentation source; the actual "did this work" measurement |
| Outside-body hallucination fix | **In planning** | Preprocessing-side issue identified in diagnostic; fix via post-process masking at inference (cheap) or tighter body silhouette (re-preprocess + retrain) |

**Architectural changes applied to both 1a and 1b (kept locked for ablation parity):**

1. **6-channel label** instead of the 5-channel design in the plan below. The 6th channel `body_other` (inside-body, non-target tissue: fat, muscle, bowel, bladder wall…) was added during 1a to fix noisy grey edges. Channel order is now: `outside_body, uterus, ov_L, ov_R, em, body_other`. ⚠ All references to "5-channel label" / "5ch" in Sections 2–5 below should be read as **6-channel**. Concat input for Exp 1a is therefore **7 channels** (1 image + 6 label), not 6.
2. **Classifier-Free Guidance (CFG)** — 10% label-dropout in training, guidance scale 3.0 at inference. Substantially improved label-to-image spatial alignment.
3. **EMA of model weights** — decay 0.9999, used for all inference and validation samples.
4. **Self-attention budget** — removed at the 128² level, kept only at 64² (deepest). Without this, 1a OOMs at batch 4 on A100 80 GB.
5. **Fixed-labels resampling** — periodic in-training sample grids resample up to 20 batches to ensure foreground content is present (the original single random draw produced background-only grids ~24% of the time).

**Exp 1b-specific:**
- Hand-built `DiffusionUNetSPADE` (`src/Generator/unet_spade.py`, `src/Generator/spade.py`) instead of subclassing MONAI's UNet — MONAI buries GroupNorm inside ResBlocks whose `forward()` doesn't accept a label argument.
- SPADE γ/β heads are **zero-initialised** so SPADE starts as identity-like (matches DiT/Imagen/SDM practice). This was the fix from the broken first run.
- Pure SPADE: `in_channels: 1` (image only); label routes via SPADE in bottleneck + decoder ResBlocks. Encoder stays on standard GroupNorm.

These changes were inherited unchanged into 1c.

**Exp 1c-specific (June 2026):**
- **Two-arm design**: same conditional PatchGAN discriminator applied to both the 1a (concat) backbone and the 1b (SPADE) backbone. Tests adversarial loss's differential contribution per architecture.
- **Conditional PatchGAN**: discriminator input is `concat(image, label)` (7-ch) so D judges both texture realism AND label-image consistency. 5 conv blocks, 70×70 receptive field, spectral norm.
- **λ-warmup schedule**: pure DDPM for steps 0-10k, ramp λ=0→0.01 over 10k-30k, constant 0.01 thereafter. Prevents adversarial signal from disrupting early diffusion training.
- **D learning rate**: 2.5e-5 (¼ of G's 1e-4).
- **Total steps 100k** (vs 80k for 1a/1b) — gives 90k effective adversarial training steps.
- **Per-variant guidance preserved**: 1c-concat samples at g=3.0, 1c-spade at g=2.0 (inherited from 1a/1b optima after Tier 1 sweep).

**Per-variant inference defaults (June 2026 finding):**
- 1a, 1c-concat → `guidance_scale: 3.0`, `num_inference_steps: 100`
- 1b, 1c-spade → `guidance_scale: 2.0`, `num_inference_steps: 100`
- Rationale: Tier 1 sweep + explainability run showed concat needs higher guidance for visible organs; SPADE prefers lower guidance to avoid grain. See `src/Generator/TIER1_TUNING_AND_EXPLAINABILITY.md`.

**Key quantitative results (June 2026):**
| Metric | 1a | 1b | 1c-concat | 1c-spade |
|---|---|---|---|---|
| FID ↓ | 188 | 200 | **166** | 188 |
| hist_KL ↓ | 8.15 | 6.89 | **5.79** | 7.20 |
| LPIPS_mean ↓ | 0.82 | 0.75 | 0.77 | **0.70** |
| CLR_uterus ↑ | 0.013 | 0.41 | 0.07 | 0.41 |

- No single winner across all metrics; clean architectural map: concat→realism, SPADE→localisation, PatchGAN adds differentially.
- See `RESULTS_2x2.md` (project root) for the full analysis.

---

## 1. COMPLETE DATA INVENTORY
 
Before anything else, here is exactly what exists.
 
```
╔═══════════════════════════════════════════════════════════════════════════╗
║                        DATASET 1 (D1)                                    ║
║          Memorial Hermann Hospital System — /D1_MHS/                     ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  Subjects:        51 patients (suspected endometriosis)                  ║
║  Sites:           15 different clinical sites                            ║
║  Scanners:        9 scanner models (GE, Philips, Siemens)               ║
║  Field strength:  1.5T and 3T mixed                                      ║
║  8 patients were NOT diagnosed with endometriosis after scanning         ║
║                                                                          ║
║  SEQUENCES AVAILABLE:                                                    ║
║  ┌──────────┬──────────┬────────────────────────────────────────────┐    ║
║  │ Sequence │ Scans    │ Slices                                     │    ║
║  ├──────────┼──────────┼────────────────────────────────────────────┤    ║
║  │ T1w      │ 42       │ 3,846                                      │    ║
║  │ T1w FS   │ 42       │ 3,846                                      │    ║
║  │ T2w      │ 45       │ 1,943                                      │    ║
║  │ T2w FS   │ ——       │ ⚠ DOES NOT EXIST IN D1 ⚠                  │    ║
║  └──────────┴──────────┴────────────────────────────────────────────┘    ║
║                                                                          ║
║  LABELS:                                                                 ║
║  ┌──────────────┬──────────┬────────────────────────────────────────┐    ║
║  │ Structure    │ n        │ Notes                                   │    ║
║  ├──────────────┼──────────┼────────────────────────────────────────┤    ║
║  │ Uterus       │ 49       │ Contoured from T2w primarily           │    ║
║  │ Ovary        │ 43       │ Contoured from T2w primarily           │    ║
║  │ Endometrioma │ 40       │ Contoured from T1w FS primarily        │    ║
║  └──────────────┴──────────┴────────────────────────────────────────┘    ║
║                                                                          ║
║  RATER INFO:                                                             ║
║    11 subjects (22%) — 3 raters                                          ║
║    22 subjects (43%) — 2 raters                                          ║
║    18 subjects (35%) — 1 rater                                           ║
║    All reviewed and corrected by experienced abdominal radiologist        ║
║                                                                          ║
║  INTER-RATER SUBSET (7 subjects):                                        ║
║    Selected for: T2w available + suspected endo + 3 raters               ║
║    Excluded if: <3 raters or ovary seg covers endometriomas/cysts        ║
║    Ovary DSC: 0.48 ± 0.24 (this is our upper-bound target)              ║
║    Uterus DSC: 0.73 ± 0.18                                              ║
╚═══════════════════════════════════════════════════════════════════════════╝
 
╔═══════════════════════════════════════════════════════════════════════════╗
║                        DATASET 2 (D2)                                    ║
║    Texas Children's Hospital Pavilion for Women — /D2_TCPW/              ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  Subjects:        81 endometriosis patients (all confirmed diagnosed)    ║
║  Sites:           1 single site                                          ║
║  Scanner:         Philips Ingenia 1.5T (1 scanner model)                 ║
║  Field strength:  1.5T only                                              ║
║                                                                          ║
║  SEQUENCES AVAILABLE:                                                    ║
║  ┌──────────┬──────────┬────────────────────────────────────────────┐    ║
║  │ Sequence │ Scans    │ Slices                                     │    ║
║  ├──────────┼──────────┼────────────────────────────────────────────┤    ║
║  │ T1w      │ 76       │ 6,675                                      │    ║
║  │ T1w FS   │ 68       │ 6,006                                      │    ║
║  │ T2w      │ 50       │ 2,439                                      │    ║
║  │ T2w FS   │ 77       │ 2,785  ← THIS IS THE TARGET SEQUENCE      │    ║
║  └──────────┴──────────┴────────────────────────────────────────────┘    ║
║                                                                          ║
║  LABELS:                                                                 ║
║  ┌──────────────┬──────────┬────────────────────────────────────────┐    ║
║  │ Structure    │ n        │ Notes                                   │    ║
║  ├──────────────┼──────────┼────────────────────────────────────────┤    ║
║  │ Uterus       │ 62       │ Single rater, T2w FS based             │    ║
║  │ Ovary        │ 58       │ Single rater, T2w FS based             │    ║
║  │ Endometrioma │ 11       │ Single rater                           │    ║
║  │ Cyst         │ 17       │ Single rater                           │    ║
║  └──────────────┴──────────┴────────────────────────────────────────┘    ║
║                                                                          ║
║  RAovSeg SPLIT (from paper):                                             ║
║  ┌────────────────┬──────┬──────────────────────────────────────────┐    ║
║  │ Partition      │ n    │ Criteria                                  │    ║
║  ├────────────────┼──────┼──────────────────────────────────────────┤    ║
║  │ Train/Val      │ 30   │ Endo + T2w FS + ovary label available    │    ║
║  │ Test           │  8   │ Same criteria, held out                   │    ║
║  │ Excluded       │ 43   │ Obvious endometriomas OR cysts present   │    ║
║  │                │      │ OR missing T2w FS / ovary labels          │    ║
║  └────────────────┴──────┴──────────────────────────────────────────┘    ║
║                                                                          ║
║  KEY NUMBERS FROM RAovSeg TRAINING:                                      ║
║    ResClass trained on: 3,252 slices (train) + 2,168 slices (val)        ║
║    AttUSeg trained on:  594 ovary-containing slices (train)              ║
║                         + 136 ovary slices (val)                         ║
╚═══════════════════════════════════════════════════════════════════════════╝
```
 
---
 
## 2. DATA ALLOCATION — WHO GETS WHAT AND WHY
 
```
╔═══════════════════════════════════════════════════════════════════════════╗
║                    PHASE 1: CLEAN ABLATION                               ║
║           All generators trained on D2 only — same data, fair test       ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  GENERATOR TRAINING DATA (Experiments 1a, 1b, 1c, and Med-DDPM)         ║
║  ─────────────────────────────────────────────────────────────────       ║
║  Source:    D2 train/val, 30 subjects                                    ║
║  Sequence:  T2w FS only                                                  ║
║  Why T2w FS: This is what RAovSeg was trained on. Generator must         ║
║              produce images in this domain. Using other sequences         ║
║              would teach the generator wrong contrast/appearance.         ║
║  Content:   2D axial slices + corresponding 5-channel label maps         ║
║  Approx:    ~730 slices total (594 with ovary + ~136 without)            ║
║                                                                          ║
║  ⚠ WHY NOT USE THE 43 EXCLUDED D2 SUBJECTS?                             ║
║    The 43 excluded subjects have endometriomas/cysts. Their ovary        ║
║    labels include pathology regions. The test set (8 subjects)           ║
║    explicitly EXCLUDES endometriomas/cysts. Training the generator       ║
║    on pathology cases and testing on non-pathology cases introduces      ║
║    a distribution mismatch. Keep it clean: match training to test.       ║
║                                                                          ║
║  ⚠ LABEL MAP FOR GENERATOR: Use single-rater labels (D2 only has 1      ║
║    rater). 5-channel binary: Background, Uterus, L-Ovary, R-Ovary,     ║
║    Endometrioma. For the 30 RAovSeg subjects, endometrioma channel      ║
║    will be empty (they were selected for having NO endometriomas).       ║
║                                                                          ║
║                                                                          ║
║  DISCRIMINATOR REAL POOL (Experiment 1c PatchGAN only)                   ║
║  ─────────────────────────────────────────────────────────────────       ║
║  Source:    D2 T2w FS slices                                             ║
║  Subjects: The SAME 30 train/val subjects                                ║
║  Why same: In Phase 1, generator and discriminator see the same          ║
║            domain. The discriminator's job here is to enforce LOCAL       ║
║            texture realism and label-image boundary consistency,          ║
║            NOT domain adaptation (that's Phase 2's job).                 ║
║                                                                          ║
║  ⚠ EXCLUDE the 8 test subjects from discriminator pool.                  ║
║    Even though the discriminator doesn't use labels and there's no       ║
║    direct label leakage, an examiner could argue the discriminator       ║
║    learned to encode test-subject-specific texture patterns into         ║
║    the generator. Play it safe.                                          ║
║                                                                          ║
║  ⚠ COULD WE use the 43 excluded D2 subjects in the discriminator        ║
║    pool? The discriminator only needs T2w FS images (no labels).         ║
║    ~39 of these 43 likely have T2w FS (77 total D2 T2w FS scans         ║
║    minus 38 included = 39). This would give ~69 subjects in the         ║
║    discriminator pool vs 30. DECISION: Use the 30 only for Phase 1      ║
║    to keep the comparison with Phase 2 clean. Note this as a             ║
║    possible future experiment.                                           ║
║                                                                          ║
║                                                                          ║
║  MED-DDPM DATA (Experiment 2)                                            ║
║  ─────────────────────────────────────────────────────────────────       ║
║  Source:    D2 train/val, same 30 subjects                               ║
║  Format:   3D NIfTI volumes (NOT 2D slices)                              ║
║  Sequence:  T2w FS only                                                  ║
║  Why same data: Fair comparison. Same subjects, same sequence.           ║
║                 Only difference is architecture (3D vs 2D) and           ║
║                 conditioning method (concat inherent to Med-DDPM).       ║
║                                                                          ║
║                                                                          ║
║  TEST SET (FIXED across ALL experiments in BOTH phases)                   ║
║  ─────────────────────────────────────────────────────────────────       ║
║  Source:    D2, 8 held-out subjects                                      ║
║  NEVER used for generator training, discriminator training,              ║
║  or any form of validation during generator development.                 ║
║  Only touched when running final RAovSeg evaluation.                     ║
║                                                                          ║
╚═══════════════════════════════════════════════════════════════════════════╝
 
 
╔═══════════════════════════════════════════════════════════════════════════╗
║                    PHASE 2: CROSS-DOMAIN STRATEGY                        ║
║           Does D1 anatomical diversity + D2 domain adaptation help?      ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  GENERATOR TRAINING DATA                                                 ║
║  ─────────────────────────────────────────────────────────────────       ║
║  Source:    D1, 51 subjects                                              ║
║  Sequence:  T2w ONLY (45 scans, 1,943 slices)                           ║
║                                                                          ║
║  ⚠ CRITICAL: D1 HAS NO T2w FAT SUPPRESSION.                             ║
║    D1 only has T1w, T1w FS, and T2w. There is NO T2w FS in D1.          ║
║    T2w is the closest available sequence to T2w FS — same base           ║
║    contrast weighting (long TR/TE → fluid bright, muscle dark)           ║
║    but WITHOUT fat signal suppression. This means:                       ║
║      - Fat appears bright in D1 T2w (peritoneal, subcutaneous)          ║
║      - Fat appears dark in D2 T2w FS                                     ║
║      - The domain gap is BOTH scanner/site AND sequence type             ║
║    This is exactly why the PatchGAN discriminator is essential for       ║
║    Phase 2 — it must learn to force fat-suppressed appearance.           ║
║                                                                          ║
║  ⚠ WHY NOT USE T1w or T1w FS from D1?                                   ║
║    T1w has completely different tissue contrast (fat bright, fluid        ║
║    dark — opposite of T2w). Including it forces the generator to         ║
║    learn a much larger appearance space, with most of that space         ║
║    being irrelevant to T2w FS output. T2w at least shares the           ║
║    same fundamental contrast mechanism as T2w FS.                        ║
║                                                                          ║
║  ⚠ LABEL MAP SELECTION FOR D1:                                           ║
║    D1 has multi-rater labels. Use the radiologist-corrected final        ║
║    labels (the reviewed/corrected versions, not individual rater         ║
║    labels). These are the most accurate anatomical ground truth.         ║
║    D1 labels were contoured prioritising T2w for uterus/ovaries,        ║
║    which matches our T2w-only sequence choice.                           ║
║                                                                          ║
║  ⚠ D1 HAS 43 SUBJECTS WITH OVARY LABELS (not all 51).                   ║
║    8 subjects may lack ovary labels entirely. Filter during              ║
║    preprocessing: only include slices where at least one label           ║
║    channel is non-empty (uterus OR ovary). Slices with only             ║
║    background are still useful for teaching background anatomy           ║
║    but should be capped (e.g., max 50% of batch) to avoid               ║
║    overwhelming the model with empty-label examples.                     ║
║                                                                          ║
║                                                                          ║
║  DISCRIMINATOR REAL POOL                                                 ║
║  ─────────────────────────────────────────────────────────────────       ║
║  Source:    D2 T2w FS slices                                             ║
║  Subjects: 30 train/val subjects (same as Phase 1)                       ║
║  Why:      The discriminator's role here is DOMAIN ADAPTATION.           ║
║            It sees only D2 T2w FS images as "real" and forces the        ║
║            generator (trained on D1 T2w) to produce output that          ║
║            looks like D2 T2w FS. This is the core novelty.              ║
║                                                                          ║
║  ⚠ EXPANDED DISCRIMINATOR POOL (recommended for Phase 2):               ║
║    In Phase 2, the discriminator needs a strong style signal for         ║
║    a harder domain adaptation task (D1 T2w → D2 T2w FS).                ║
║    Consider expanding the discriminator pool to include T2w FS           ║
║    slices from ALL D2 subjects that have them (up to 69 subjects,       ║
║    excluding 8 test + 4 without T2w FS). The discriminator doesn't      ║
║    need paired labels — it just needs to know what "real D2 T2w FS"     ║
║    looks like. MORE style examples = stronger domain adaptation.         ║
║    Run this as a sub-experiment: Phase 2a (30 D-subjects) vs            ║
║    Phase 2b (69 D-subjects) to see if it matters.                        ║
║                                                                          ║
║                                                                          ║
║  ARCHITECTURE FOR PHASE 2                                                ║
║  ─────────────────────────────────────────────────────────────────       ║
║  Use the best-performing architecture from Phase 1                       ║
║  (expected: Exp 1c, DDPM + SPADE + PatchGAN).                           ║
║  Same hyperparameters, same training schedule.                           ║
║  Only change: swap generator training data from D2 to D1 T2w.           ║
║                                                                          ║
║  This isolates the cross-domain training strategy as a single            ║
║  variable: everything else is identical to Phase 1's Exp 1c.             ║
║                                                                          ║
║                                                                          ║
║  TEST SET                                                                ║
║  ─────────────────────────────────────────────────────────────────       ║
║  SAME 8 held-out D2 subjects as Phase 1. Never changes.                  ║
║                                                                          ║
╚═══════════════════════════════════════════════════════════════════════════╝
```
 
---
 
## 3. PREPROCESSING PIPELINE
 
```
╔═══════════════════════════════════════════════════════════════════════════╗
║              STEP 1: BASE PREPROCESSING (applied to everything)          ║
║              Identical for D1, D2, and later for synthetic data          ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  1. Load NIfTI from Zenodo download                                      ║
║     D1: /D1_MHS/{subject}/  → registered scans + multi-rater labels     ║
║     D2: /D2_TCPW/{subject}/ → registered scans + single-rater labels    ║
║                                                                          ║
║  2. Intensity clip: 1st–99th percentile                                  ║
║     Why: Removes extreme outlier voxels from scanner noise               ║
║                                                                          ║
║  3. Normalise to [0, 1]                                                  ║
║     Why: Standard range for neural network input                         ║
║                                                                          ║
║  4. Resample to 512 × 512 pixels @ 5mm × 5mm voxel size                 ║
║     Why: Matches RAovSeg training resolution exactly                     ║
║     ⚠ Use the same interpolation method as RAovSeg:                      ║
║       - Bilinear for images                                              ║
║       - Nearest-neighbour for label maps (preserves binary values)       ║
║                                                                          ║
║  5. Store NIfTI affine matrix and voxel spacing metadata                 ║
║     Why: Needed later to reassemble synthetic 2D slices into             ║
║     3D NIfTI volumes with correct spatial coordinates                    ║
║                                                                          ║
║  ⚠ DO NOT apply ovary intensity enhancement at this stage.               ║
║    The enhancement is a RAovSeg-specific preprocessing step.             ║
║    Generators should learn to produce images in the [0,1]                ║
║    normalised space. Enhancement is applied DOWNSTREAM,                  ║
║    identically to real and synthetic data, when training RAovSeg.        ║
║                                                                          ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║              STEP 2: 2D SLICE EXTRACTION (for all 2D models)             ║
║                                                                          ║
║  Extract axial slices from preprocessed 3D volumes:                      ║
║    - Each slice is a 512×512 grayscale image                             ║
║    - Each corresponding label map is 512×512 × 5 channels               ║
║    - Record slice index (z-position) for each slice                      ║
║      (needed for ISCS at inference and for positional encoding)          ║
║    - Record which subject and volume each slice came from                ║
║      (needed for train/test integrity checks)                            ║
║                                                                          ║
║  LABEL MAP CHANNELS (same order for D1 and D2):                          ║
║    Channel 0: Background (inverse of all other channels)                 ║
║    Channel 1: Uterus                                                     ║
║    Channel 2: Left ovary                                                 ║
║    Channel 3: Right ovary                                                ║
║    Channel 4: Endometrioma (empty for 30 RAovSeg train subjects)         ║
║                                                                          ║
║  ⚠ VERIFY after extraction: load a few slices, overlay label on          ║
║    image, confirm alignment is correct. Misaligned labels will           ║
║    silently break SPADE conditioning.                                    ║
║                                                                          ║
║  ⚠ SLICE FILTERING DECISIONS:                                            ║
║    Include ALL slices (with and without ovaries). The generator           ║
║    needs to learn full pelvic anatomy — background, uterus-only          ║
║    slices, and ovary-containing slices. The label map tells the          ║
║    generator what to put where.                                          ║
║    For training, consider a weighted sampler: oversample slices          ║
║    containing ovaries (2x–3x weight) so the model sees enough           ║
║    ovary examples despite them being a minority of slices.               ║
║                                                                          ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║         STEP 3: SEQUENCE FILTERING (depends on experiment phase)         ║
║                                                                          ║
║  PHASE 1 (all experiments): D2 T2w FS slices ONLY                        ║
║    Filter D2 volumes to T2w Fat Suppression sequence before              ║
║    extracting slices. Discard T1w, T1w FS, and non-FS T2w.              ║
║    Why: RAovSeg operates on T2w FS. All generators in Phase 1            ║
║    should learn this specific domain.                                    ║
║                                                                          ║
║  PHASE 2 (generator): D1 T2w slices ONLY                                ║
║    Filter D1 volumes to T2w sequence. Discard T1w and T1w FS.           ║
║    Why: T2w is the closest sequence to T2w FS available in D1.           ║
║    Same base contrast (long TR/TE) but without fat suppression.          ║
║    Using T1w would introduce completely different contrast that           ║
║    makes the domain adaptation task unnecessarily hard.                   ║
║                                                                          ║
║  PHASE 2 (discriminator): D2 T2w FS slices ONLY (unchanged)             ║
║    Why: Discriminator always defines the target domain = D2 T2w FS.      ║
║                                                                          ║
╚═══════════════════════════════════════════════════════════════════════════╝
```
 
### Data Count Summary After Preprocessing
 
```
┌──────────────────────────────────────────────────────────────┐
│              EXPECTED SLICE COUNTS                            │
├────────────────────────┬─────────────────────────────────────┤
│ D2 T2w FS (30 subj)   │ ~730 total slices                   │
│   of which ovary+      │ ~594 (from paper)                   │
│   of which ovary-      │ ~136 (from paper)                   │
├────────────────────────┼─────────────────────────────────────┤
│ D1 T2w (45 scans)      │ ~1,943 slices                      │
│   of which with labels │ ~1,600 (est, 43/51 have ovary)     │
├────────────────────────┼─────────────────────────────────────┤
│ D2 test (8 subj T2w FS)│ ~200 slices (NEVER TOUCH)          │
└────────────────────────┴─────────────────────────────────────┘
 
⚠ FIRST TASK AFTER DOWNLOAD: Count actual slices per sequence
  per subject. The numbers above are from the paper; your actual
  extraction may differ slightly due to edge slices or empty volumes.
  Log these counts — you'll need them for your methodology section.
```
 
---
 
## 4. ARCHITECTURE: DDPM + SPADE + PatchGAN
 
```
                    ┌──────────────────────┐
                    │   LABEL MAP INPUT    │
                    │  (5-ch binary mask)   │
                    │                      │
                    │  Source varies:       │
                    │  Phase 1 → D2 labels │
                    │  Phase 2 → D1 labels │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼──────────┐
                    │  SPADE ENCODER      │
                    │  (small ConvNet)     │
                    │                     │
                    │  For EACH decoder   │
                    │  resolution level:  │
                    │  label map →        │
                    │  AvgPool to match   │
                    │  spatial size →     │
                    │  Conv → ReLU →      │
                    │  Conv → γ, β        │
                    └──┬──┬──┬──┬──┬──┬──┘
                       │  │  │  │  │  │
            Per-pixel affine params to each decoder layer
                       │  │  │  │  │  │
                       ▼  ▼  ▼  ▼  ▼  ▼
┌──────────────────────────────────────────────────────────┐
│                                                          │
│              DDPM U-NET DENOISING BACKBONE                │
│                                                          │
│  ENCODER (unchanged, standard GroupNorm):                │
│  ┌─────────────────────────────────────────────┐        │
│  │ Input: noisy image x_t (1ch) + timestep t   │        │
│  │                                              │        │
│  │ ⚠ Exp 1a ONLY: input is x_t (1ch) CONCAT    │        │
│  │   label_map (5ch) = 6ch input total.         │        │
│  │   No SPADE, no separate label encoding.      │        │
│  │                                              │        │
│  │ Exp 1b, 1c, Phase 2: input is x_t (1ch)     │        │
│  │   only. Labels injected via SPADE in decoder.│        │
│  │                                              │        │
│  │ Down blocks: Conv → GroupNorm → SiLU → Down  │        │
│  │ Timestep t: sinusoidal encoding → added      │        │
│  └──────────────────────┬──────────────────────┘        │
│                         │ skip connections               │
│  MIDDLE BLOCK           │                                │
│  ┌──────────────────────┤                                │
│  │ Self-attention + Res │                                │
│  └──────────────────────┤                                │
│                         │                                │
│  DECODER (★ SPADE replaces GroupNorm here ★):            │
│  ┌──────────────────────▼──────────────────────┐        │
│  │ Up blocks:                                   │        │
│  │   feature_in = concat(upsampled, skip)       │        │
│  │   ★ SPADE(feature_in, label_map) ★           │        │
│  │     → normalise feature_in                   │        │
│  │     → multiply by γ(label_map)               │        │
│  │     → add β(label_map)                       │        │
│  │   Conv → SiLU → Up                           │        │
│  │                                              │        │
│  │ ⚠ Exp 1a: standard GroupNorm here instead    │        │
│  │   of SPADE. This is the concat baseline.     │        │
│  └──────────────────────┬──────────────────────┘        │
│                         │                                │
│  OUTPUT: predicted noise ε_θ(x_t, t, label_map)         │
│                                                          │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  DDPM REVERSE PROCESS │
              │  x_t → x_{t-1} → x_0 │
              │                       │
              │  Training: T=1000     │
              │  Inference: DDIM 50   │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  GENERATED IMAGE x_0  │
              │  Should look like:    │
              │  Phase 1 → D2 T2w FS  │
              │  Phase 2 → D2 T2w FS  │
              │  (both target same     │
              │   output domain)       │
              └───────────┬───────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
   ┌────────────┐  ┌───────────┐  ┌───────────────────┐
   │  L_diff    │  │  CONCAT   │  │  PatchGAN         │
   │ MSE(ε,ε̂)  │  │ [img |    │  │  DISCRIMINATOR    │
   │            │  │  label]   │  │                   │
   │ Primary    │  │ 6ch input │  │  ⚠ Exp 1a, 1b:   │
   │ loss,      │  │ to D      │  │    NOT PRESENT    │
   │ always on  │  └─────┬─────┘  │                   │
   └──────┬─────┘        │        │  ⚠ Exp 1c +       │
          │              ▼        │    Phase 2: ON     │
          │        ┌───────────┐  │                   │
          │        │ Real:     │  │  Real pool:       │
          │        │ D2 T2w FS │  │  Phase 1 → 30     │
          │        │ + its     │  │    D2 train subj  │
          │        │ label map │  │  Phase 2 → 30-69  │
          │        │           │  │    D2 subjects     │
          │        │ Fake:     │  │    (see Section 2)│
          │        │ gen image │  │                   │
          │        │ + input   │  │  70×70 patches    │
          │        │ label map │  │  Spectral norm    │
          │        └───────────┘  └─────────┬─────────┘
          │                                 │
          ▼                                 ▼
   ┌──────────────────────────────────────────────────┐
   │           COMBINED GENERATOR LOSS                 │
   │                                                   │
   │   Exp 1a:  L = L_diff                             │
   │   Exp 1b:  L = L_diff                             │
   │   Exp 1c:  L = L_diff + λ·L_adv                   │
   │   Phase 2: L = L_diff + λ·L_adv                   │
   │                                                   │
   │   λ SCHEDULE (Exp 1c and Phase 2 only):           │
   │     Steps 0–10k:    λ = 0 (pure DDPM warmup)      │
   │     Steps 10k–30k:  λ ramps 0 → 0.01 linearly     │
   │     Steps 30k+:     λ = 0.01                       │
   │                                                   │
   │   OPTIMISERS:                                     │
   │     Generator:     Adam, lr = 1e-4                 │
   │     Discriminator: Adam, lr = 2.5e-5 (¼ of G)     │
   │     Spectral norm on all discriminator weights     │
   │                                                   │
   │   ⚠ PHASE 2 NOTE: The discriminator has a HARDER  │
   │     job in Phase 2 (D1 T2w → D2 T2w FS gap is    │
   │     larger than D2 T2w FS → D2 T2w FS in Ph1).   │
   │     May need to lower λ further or extend warmup.  │
   │     Monitor D accuracy: if >90% after warmup,      │
   │     the generator is failing to adapt. Lower G lr.  │
   └──────────────────────────────────────────────────┘
```
 
---
 
## 5. TRAINING LOOP (annotated with data sources)
 
```
FOR each training step:
│
├── 1. SAMPLE batch of (image, label_map) pairs
│       Phase 1: from D2 T2w FS, 30 train subjects
│       Phase 2: from D1 T2w, 51 subjects
│       ⚠ Use weighted sampler: oversample ovary-containing slices 2-3x
│
├── 2. SAMPLE random timestep t ~ Uniform(1, 1000)
│
├── 3. SAMPLE noise ε ~ N(0, I), same shape as image (512×512)
│
├── 4. CREATE noisy image: x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε
│
├── 5. PREDICT noise:
│       Exp 1a: ε̂ = UNet(concat(x_t, label_map), t)     ← 6ch input
│       Exp 1b: ε̂ = UNet_SPADE(x_t, t, label_map)       ← 1ch input, SPADE
│       Exp 1c: ε̂ = UNet_SPADE(x_t, t, label_map)       ← same as 1b
│       Phase 2: ε̂ = UNet_SPADE(x_t, t, label_map)      ← same architecture
│
├── 6. L_diff = MSE(ε, ε̂)
│
├── 7. IF experiment has PatchGAN AND step > warmup_steps:
│   │
│   ├── 7a. Single-step x_0 estimate:
│   │       x̂_0 = (x_t - √(1-ᾱ_t)·ε̂) / √(ᾱ_t)
│   │       ⚠ This is noisy. Clamp to [0,1] before feeding to D.
│   │
│   ├── 7b. fake_input = concat(x̂_0, label_map)   → 6 channels
│   │
│   ├── 7c. SAMPLE real T2w FS slice from D2 discriminator pool
│   │       real_label = corresponding label map for that real slice
│   │       real_input = concat(real_slice, real_label)
│   │
│   │       ⚠ PHASE 1: real_label comes from D2's 30 train subjects
│   │       ⚠ PHASE 2: real_label comes from D2's 30 (or 69) subjects
│   │         The discriminator checks [image|label] consistency, so
│   │         it needs REAL paired image+label from D2.
│   │         For the 39 extra D2 subjects without RAovSeg labels,
│   │         you'd need their T2w FS + whatever labels they have.
│   │         If their labels include endometriomas/cysts, the D sees
│   │         a different label distribution than the generator.
│   │         SAFER: stick to 30 train subjects for discriminator
│   │         in BOTH phases for consistency.
│   │
│   ├── 7d. DISCRIMINATOR STEP (separate optimiser):
│   │       D_loss = BCE(D(real_input), 1) + BCE(D(fake_input.detach()), 0)
│   │       Optimiser_D.step()
│   │       ⚠ .detach() on fake_input — do NOT backprop D loss into G
│   │
│   ├── 7e. GENERATOR ADVERSARIAL LOSS:
│   │       L_adv = BCE(D(fake_input), 1)    ← G wants D to say "real"
│   │
│   └── 7f. L_total = L_diff + λ·L_adv
│
├── 8. ELSE: L_total = L_diff
│
├── 9. Optimiser_G.step()
│
└── 10. LOG to TensorBoard:
        - L_diff (should decrease steadily)
        - L_adv (should stabilise around 0.5-1.0 after warmup)
        - D accuracy on real batch (should be 50-70%, NOT 99%)
        - D accuracy on fake batch (should be 50-70%, NOT 99%)
        - Sample generated images every 5k steps
        - ⚠ SPADE CHECK: generate from 2 different label maps,
          confirm outputs look structurally different
```
 
---
 
## 6. INFERENCE & NIfTI ASSEMBLY
 
```
┌─────────────────────────────────────────────────────────────────┐
│                    LABEL MAP SOURCE                              │
│                                                                  │
│  Take the 30 D2 training subject label map volumes.              │
│  For each, apply augmentation to create novel variants:          │
│                                                                  │
│  Augmentations (applied to 3D label volume before slicing):      │
│    - Random elastic deformation (σ=10-20, grid=4-6)              │
│    - Random rotation ±15°                                        │
│    - Random flip (left-right → swaps L/R ovary channels too!)    │
│    - Random scaling of ovary region (0.7x – 1.5x area)           │
│      within physiological range (7-20cc volume)                  │
│    - Random in-plane translation ±20 pixels                      │
│                                                                  │
│  ⚠ When flipping left-right: ALSO swap channel 2 (L ovary)      │
│    and channel 3 (R ovary). Otherwise the label semantics are    │
│    wrong after the flip.                                         │
│                                                                  │
│  ⚠ After augmentation, verify labels are still valid binary      │
│    masks (no fractional values from interpolation). Round to     │
│    0/1 and verify no overlap between channels.                   │
│                                                                  │
│  Generate N augmented label map volumes (e.g., N=30 for 1:1).   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              SLICE-BY-SLICE GENERATION WITH ISCS                 │
│                                                                  │
│  For each augmented label volume (S slices along z):             │
│                                                                  │
│  1. Generate a SHARED noise base:                                │
│     ε_shared ~ N(0, I), shape (512, 512)                        │
│                                                                  │
│  2. For each slice z = 1..S:                                     │
│     a. Get label_map_z for this slice                            │
│     b. Generate independent noise:                               │
│        ε_z_ind ~ N(0, I)                                         │
│     c. ISCS blended noise:                                       │
│        ε_z = α·ε_shared + √(1-α²)·ε_z_ind                      │
│        ⚠ Note √(1-α²) not (1-α) — preserves unit variance      │
│        α = 0.8 (start here, tune if needed)                     │
│     d. Run DDIM reverse diffusion (50 steps):                    │
│        x_0_z = reverse_diffusion(ε_z, label_map_z)              │
│     e. Clamp output to [0, 1]                                    │
│                                                                  │
│  3. Stack slices: synthetic_volume = stack(x_0_1, ..., x_0_S)   │
│     Shape: (512, 512, S)                                         │
│                                                                  │
│  4. Stack label slices: label_volume = stack(label_1, ..., S)    │
│     This is the paired ground truth for RAovSeg training         │
│                                                                  │
│  ⚠ ISCS NOTE: Use same ε_shared across ALL slices in one        │
│    volume, but generate a NEW ε_shared for each volume.          │
│    Adjacent volumes should NOT share noise — only slices         │
│    within a single volume share the coherence signal.            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              NIfTI VOLUME ASSEMBLY                                │
│                                                                  │
│  For each synthetic volume:                                      │
│                                                                  │
│  1. Create NIfTI header matching D2 T2w FS volumes:              │
│     - Voxel size: same as D2 (in-plane from original, 5mm z)    │
│     - Affine matrix: copy from a representative D2 volume        │
│     - Data type: float32                                         │
│     - Orientation: match D2 (check with nibabel get_best_affine) │
│                                                                  │
│  2. Save image volume as: synth_{id}_image.nii.gz                │
│  3. Save label volume as: synth_{id}_label.nii.gz                │
│                                                                  │
│  ⚠ VERIFY by loading in 3D Slicer:                               │
│    - Axial view: does each slice look like pelvic MRI?            │
│    - Sagittal view: are structures continuous across slices?      │
│      (ISCS should prevent major discontinuities)                 │
│    - Overlay label on image: do boundaries match?                │
│                                                                  │
│  Deliverable: a folder of synthetic NIfTI pairs, ready to        │
│  drop into the RAovSeg preprocessing pipeline identically        │
│  to real D2 data.                                                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│       RAovSeg PREPROCESSING (applied to real AND synthetic)      │
│                                                                  │
│  1. Intensity clip: 1st–99th percentile                          │
│  2. Normalise to [0, 1]                                          │
│  3. Resample to 512×512 @ 5mm (should be no-op for synthetic)   │
│  4. Ovary intensity enhancement:                                 │
│     - Voxels in [0.22, 0.3] → set to 1                          │
│     - Voxels < 0.5 and outside [0.22, 0.3] → unchanged          │
│     - Voxels ≥ 0.5 → invert to (1 − I₀)                        │
│                                                                  │
│  ⚠ Steps 1-3 are mostly no-ops for synthetic data (already at   │
│    512×512, [0,1]). Run them anyway — the clip will catch any    │
│    diffusion sampling artefacts slightly outside [0,1].          │
│                                                                  │
│  ⚠ Step 4 is the CRITICAL one. This enhancement was tuned for   │
│    real D2 T2w FS intensity distributions. If your synthetic     │
│    images have a different intensity profile, the [0.22, 0.3]   │
│    window may not align with synthetic ovary intensities.        │
│    CHECK: histogram of synthetic ovary voxels vs real ovary      │
│    voxels. If misaligned, this is a problem.                     │
│                                                                  │
│  5. Extract 2D axial slices for ResClass + AttUSeg training      │
│  6. Augment: random translation ±25px, rotation ±25° (5x)       │
│     ⚠ Apply same augmentation to real and synthetic slices       │
└─────────────────────────────────────────────────────────────────┘
```
 
---
 
## 7. COMPLETE ABLATION EXPERIMENT DESIGN
 
```
═══════════════════════════════════════════════════════════════════
EXPERIMENT 0 — REPRODUCE BASELINES (no synthetic data)
═══════════════════════════════════════════════════════════════════
 
  0a. RAovSeg on D2 (30 train, 8 test)     → reproduce DSC 0.290
  0b. nnU-Net on D2 (30 train, 8 test)     → reproduce DSC 0.272
 
  Data: D2 T2w FS, original paper split
  ⚠ DO THIS FIRST. If you cannot reproduce 0.290, STOP.
    Recheck preprocessing, data split, hyperparameters.
    Every downstream result depends on this baseline.
 
 
═══════════════════════════════════════════════════════════════════
PHASE 1 — CLEAN ABLATION (isolate architectural components)
All generators trained on D2 T2w FS, 30 subjects
═══════════════════════════════════════════════════════════════════
 
  Exp 1a: 2D DDPM + CONCAT conditioning
  ─────────────────────────────────────
  Generator data:     D2 T2w FS, 30 subjects, ~730 slices
  Conditioning:       Label map concatenated as 5 extra input channels
  Discriminator:      NONE
  Loss:               L_diff only
  This tests:         "Does a basic conditional DDPM produce useful
                       synthetic data?"
 
  Exp 1b: 2D DDPM + SPADE conditioning
  ─────────────────────────────────────
  Generator data:     D2 T2w FS, 30 subjects, ~730 slices (SAME)
  Conditioning:       SPADE in decoder layers
  Discriminator:      NONE
  Loss:               L_diff only
  This tests:         "Does SPADE improve boundary fidelity vs concat?"
  Comparison:         1a vs 1b isolates SPADE's contribution
 
  Exp 1c: 2D DDPM + SPADE + PatchGAN
  ─────────────────────────────────────
  Generator data:     D2 T2w FS, 30 subjects, ~730 slices (SAME)
  Conditioning:       SPADE in decoder layers (SAME as 1b)
  Discriminator:      PatchGAN on D2 T2w FS, 30 subjects
  Loss:               L_diff + λ·L_adv
  This tests:         "Does adversarial training improve realism
                       even when generator and discriminator see
                       the same domain?"
  Comparison:         1b vs 1c isolates PatchGAN's contribution
 
  Exp 2: Med-DDPM (3D, original architecture)
  ─────────────────────────────────────
  Generator data:     D2 T2w FS, 30 subjects, 3D volumes
  Conditioning:       Concat (as published in Med-DDPM)
  Discriminator:      NONE (Med-DDPM is pure diffusion)
  Loss:               L_diff only (3D noise prediction)
  This tests:         "Can an established 3D method handle this
                       extreme low-data regime (30 volumes)?"
  Comparison:         Exp 2 vs Exp 1a (same conditioning, 3D vs 2D)
                      Exp 2 vs Exp 1c (established vs proposed)
 
  ⚠ KEY CONTROLS FOR PHASE 1:
    - ALL experiments use SAME 30 D2 subjects
    - ALL experiments use T2w FS sequence ONLY
    - ALL generate same N synthetic volumes (e.g., N=30)
    - ALL evaluate on SAME 8 test subjects
    - ALL use SAME RAovSeg hyperparameters for downstream eval
    - Report mean ± std over 3 runs with different random seeds
 
 
═══════════════════════════════════════════════════════════════════
PHASE 2 — CROSS-DOMAIN TRAINING STRATEGY
Take best architecture from Phase 1, test two-dataset approach
═══════════════════════════════════════════════════════════════════
 
  Exp 3: Best model + D1 generator / D2 discriminator
  ─────────────────────────────────────
  Architecture:       Same as best from Phase 1 (expected: 1c)
  Generator data:     D1 T2w, 45 scans, ~1,943 slices
                      ⚠ NOT T2w FS (doesn't exist in D1)
                      ⚠ NOT T1w (wrong contrast mechanism)
  Conditioning:       SPADE, using D1 radiologist-corrected labels
  Discriminator:      PatchGAN on D2 T2w FS, 30 subjects
  Loss:               L_diff + λ·L_adv (same schedule as 1c)
 
  This tests:         "Does training the generator on D1's diverse
                       anatomy (51 subjects, 15 sites) with D2 domain
                       adaptation produce better synthetic data than
                       training on D2 alone (30 subjects, 1 site)?"
  Comparison:         Exp 3 vs Exp 1c
                      (same architecture, only training data changes)
 
  ⚠ DOMAIN GAP IS LARGER IN PHASE 2:
    Phase 1 (Exp 1c): D2 T2w FS → D2 T2w FS (same domain)
    Phase 2 (Exp 3):  D1 T2w → D2 T2w FS (cross-site + cross-sequence)
    
    The discriminator must learn to push the generator from T2w
    appearance (bright fat) toward T2w FS appearance (dark fat).
    
    If this fails, possible mitigations:
    - Lower λ further (0.005 instead of 0.01)
    - Extend warmup (20k steps instead of 10k)
    - Use only the D1 subjects with confirmed endometriosis (43/51)
    - Pre-filter D1 slices to those most visually similar to T2w FS
      (e.g., by histogram matching before training)
 
 
═══════════════════════════════════════════════════════════════════
EXPERIMENT 4 — TITRATION (on best overall model)
═══════════════════════════════════════════════════════════════════
 
  Take the single best generator from Exp 1a-1c and Exp 3.
  Vary synthetic:real data ratio:
 
  ┌───────┬───────────────┬──────────────────────────────────────┐
  │ Ratio │ Real + Synth  │ Purpose                              │
  ├───────┼───────────────┼──────────────────────────────────────┤
  │ 0:1   │ 30 + 0        │ Baseline (Exp 0a)                    │
  │ 0.5:1 │ 30 + 15       │ Minimal augmentation                 │
  │ 1:1   │ 30 + 30       │ Equal                                │
  │ 2:1   │ 30 + 60       │ Heavy                                │
  │ 3:1   │ 30 + 90       │ Aggressive                           │
  │ ∞:1   │ 0  + 90       │ Synthetic ONLY — can it stand alone? │
  └───────┴───────────────┴──────────────────────────────────────┘
 
  ⚠ The ∞:1 condition reveals whether synthetic data has learned
    enough to replace real data or only works as a supplement.
 
 
═══════════════════════════════════════════════════════════════════
EXPERIMENT 5 — IMAGE QUALITY METRICS (generator evaluation only)
═══════════════════════════════════════════════════════════════════
 
  Run on ALL generator outputs from Exp 1a, 1b, 1c, 2, 3:
 
  ┌──────────────────┬─────────────────────────────────────────────┐
  │ Metric           │ What it tells you + how to compute          │
  ├──────────────────┼─────────────────────────────────────────────┤
  │ FID              │ Distribution distance: generated vs real D2 │
  │                  │ T2w FS. Lower = more realistic.             │
  │                  │ Use pytorch-fid library.                    │
  ├──────────────────┼─────────────────────────────────────────────┤
  │ MS-SSIM          │ Structural similarity to closest real image.│
  │                  │ Should be moderate (too high = memorisation).│
  ├──────────────────┼─────────────────────────────────────────────┤
  │ Boundary DSC     │ Segment generated image with pretrained     │
  │                  │ model → compare to input label map.         │
  │                  │ ⚠ KEY for SPADE argument: if 1b > 1a here, │
  │                  │ SPADE directly improves boundary fidelity.  │
  ├──────────────────┼─────────────────────────────────────────────┤
  │ NN Distance      │ LPIPS distance from each synthetic image to │
  │ (memorisation)   │ nearest real training image. If < threshold,│
  │                  │ model is memorising. Report min/mean/max.   │
  ├──────────────────┼─────────────────────────────────────────────┤
  │ Intensity hist.  │ Compare intensity histograms of synthetic   │
  │                  │ vs real D2 T2w FS. Especially check the     │
  │                  │ [0.22-0.3] ovary intensity window.          │
  │                  │ ⚠ If these don't match, the downstream      │
  │                  │ enhancement step will fail.                  │
  └──────────────────┴─────────────────────────────────────────────┘
```
 
---
 
## 8. RESULTS TABLE TEMPLATE
 
```
┌───────────────────────────────────┬──────────┬──────┬──────┬──────┐
│ Condition                         │ DSC↑     │ FID↓ │ BndDSC│NN↑  │
├───────────────────────────────────┼──────────┼──────┼──────┼──────┤
│ Upper bound (inter-rater)         │ 0.48±.24 │  —   │  —   │  —   │
│ 0a. RAovSeg (real only)           │ 0.290    │  —   │  —   │  —   │
│ 0b. nnU-Net (real only)           │ 0.272    │  —   │  —   │  —   │
├───────────────────────────────────┼──────────┼──────┼──────┼──────┤
│ PHASE 1: D2 training only                                         │
│ 1a. +DDPM concat synth            │  TBD     │ TBD  │ TBD  │ TBD  │
│ 1b. +DDPM SPADE synth             │  TBD     │ TBD  │ TBD  │ TBD  │
│ 1c. +DDPM SPADE+PG synth          │  TBD     │ TBD  │ TBD  │ TBD  │
│ 2.  +Med-DDPM 3D synth            │  TBD     │ TBD  │ TBD  │ TBD  │
├───────────────────────────────────┼──────────┼──────┼──────┼──────┤
│ PHASE 2: Cross-domain                                             │
│ 3.  +Best arch, D1-gen/D2-disc    │  TBD     │ TBD  │ TBD  │ TBD  │
├───────────────────────────────────┼──────────┼──────┼──────┼──────┤
│ TITRATION (best model):                                           │
│ 4a. 0.5:1 ratio                   │  TBD     │  —   │  —   │  —   │
│ 4b. 1:1 ratio                     │  TBD     │  —   │  —   │  —   │
│ 4c. 2:1 ratio                     │  TBD     │  —   │  —   │  —   │
│ 4d. 3:1 ratio                     │  TBD     │  —   │  —   │  —   │
│ 4e. synth only                    │  TBD     │  —   │  —   │  —   │
└───────────────────────────────────┴──────────┴──────┴──────┴──────┘
 
DSC = Dice Similarity Coefficient (ovary, 3D, on 8 test subjects)
FID = Fréchet Inception Distance (lower is better)
BndDSC = Boundary fidelity DSC (generated image segmented → vs input label)
NN = Nearest-neighbour LPIPS distance (higher = less memorisation)
```
 
---
 
## 9. CRITICAL WATCHPOINTS & VERIFICATION CHECKLIST
 
### Pre-Experiment Data Verification
- [ ] Download dataset from Zenodo, verify file count matches paper (Table 1)
- [ ] Count actual T2w FS scans in D2: expect 77 subjects, 2,785 slices
- [ ] Count actual T2w scans in D1: expect 45 scans, 1,943 slices
- [ ] Confirm D1 has NO T2w FS — search for any FS-tagged files
- [ ] Identify the 38 included + 8 test + 43 excluded D2 subjects by ID
- [ ] For D1 multi-rater labels: identify which file is the corrected final version
- [ ] Check label channel encoding: confirm binary (0/1), no overlap between structures
- [ ] Load 5 random D2 T2w FS slices + labels in matplotlib, verify visual alignment
- [ ] Load 5 random D1 T2w slices + labels, verify alignment
- [ ] Compare D1 T2w and D2 T2w FS visually: note the fat signal difference
### Pre-Training Verification
- [ ] Reproduce RAovSeg DSC of 0.290 on 8 test subjects (MUST PASS)
- [ ] MONAI Generative 2D DDPM runs on your GPU at 512×512 (check VRAM)
- [ ] If VRAM insufficient: test at 256×256, note in methodology as limitation
- [ ] Verify label maps are correctly loaded as 5-channel tensors
- [ ] Confirm weighted sampler oversamples ovary-containing slices
### During Training
- [ ] L_diff decreasing steadily for all experiments
- [ ] For Exp 1c: L_adv stabilises, D accuracy 50-70%
- [ ] Every 5k steps: generate 5 images, save to disk, visual check
- [ ] SPADE sanity check: same noise + different label → different output
- [ ] Concat sanity check: same noise + different label → different output
- [ ] If D accuracy hits 99%: generator is failing, halve λ
- [ ] If D accuracy stays <50%: discriminator too weak, increase D lr
### Pre-Evaluation Verification
- [ ] Generate 5 complete synthetic NIfTI volumes
- [ ] Load in 3D Slicer: check axial, sagittal, coronal views
- [ ] Overlay synthetic labels on synthetic images: boundaries match?
- [ ] Compare synthetic intensity histogram vs real D2 T2w FS histogram
- [ ] Check the [0.22, 0.3] ovary intensity window specifically
- [ ] Run nearest-neighbour LPIPS check: flag any synthetic < 0.1 from real
- [ ] Verify NIfTI headers (voxel size, affine, orientation) match D2
### During Ablation Evaluation
- [ ] SAME 8 test subjects across ALL experiments
- [ ] SAME RAovSeg hyperparameters (lr, epochs, loss params) across ALL
- [ ] SAME N synthetic volumes across Exp 1a, 1b, 1c
- [ ] 3 runs per condition, different random seeds → report mean ± std
- [ ] If any condition WORSENS DSC: report it, analyse why
- [ ] For Phase 2 vs Phase 1: same N synthetic volumes
---
 
## 10. BUILD SCHEDULE
 
```
⚠ STATUS NOTE (June 2026): The weekly schedule below is preserved as the
original plan from February 2026. The project is now past Week 4 in
calendar terms but is tracking the milestones non-linearly — Exp 1a is
complete with extra QoL additions (see Section 0), Exp 1b v2 is mid-run,
Exp 1c and Med-DDPM (Exp 2) have not started. Treat this section as
historical scope rather than current schedule.

WEEK 1:  Data preparation + baseline reproduction
         ├── Download dataset, run all data verification checks
         ├── Write preprocessing pipeline (clip, norm, resample)
         ├── Extract and count 2D slices per subject per sequence
         ├── Reproduce RAovSeg DSC 0.290 → GATE: must pass
         └── Get MONAI Generative 2D DDPM tutorial running
 
WEEK 2:  Exp 1a — 2D DDPM + concat conditioning
         ├── Modify MONAI DDPM: 6-channel input (image + 5ch label)
         ├── Train on D2 T2w FS, 30 subjects
         ├── Generate synthetic slices, visual inspection
         └── Compute FID, boundary DSC, NN distance
 
WEEK 3:  Exp 1b — 2D DDPM + SPADE conditioning
         ├── Fork Week 2 codebase
         ├── Replace decoder GroupNorm with SPADE (one layer at a time)
         ├── Reference: Park et al. SPADE repo for implementation
         ├── Verify: gradient check on SPADE params
         ├── Sanity check: different labels → different outputs
         └── Train, generate, compute quality metrics
 
WEEK 4:  Exp 1c — Add PatchGAN + Med-DDPM setup
         ├── Fork Week 3 codebase
         ├── Add PatchGAN discriminator (separate optimiser)
         ├── Implement λ warmup schedule
         ├── D2 T2w FS slices as discriminator real pool (30 subj)
         ├── Train, monitor D accuracy + losses carefully
         ├── In parallel: set up Med-DDPM on D2 3D volumes (Exp 2)
         └── Generate from all, compute quality metrics
 
WEEK 5:  Phase 1 downstream evaluation + Phase 2 training
         ├── Assemble synthetic NIfTIs for 1a, 1b, 1c, 2
         ├── Run RAovSeg augmented training for each (3 seeds each)
         ├── Record all DSC results → Phase 1 results table
         ├── Start Phase 2: retrain best model on D1 T2w + D2 disc
         └── ⚠ Watch for harder domain adaptation in Phase 2
 
WEEK 6:  Phase 2 evaluation + titration + write-up
         ├── Generate synthetic NIfTIs from Phase 2 model
         ├── Run RAovSeg evaluation for Phase 2 (3 seeds)
         ├── Run titration experiment (Exp 4) on best overall model
         ├── Compile full results table
         └── Begin writing methodology and results sections
```
 
---
 
*Document version: v2, February 2026. Option A two-phase design.
All data allocations explicit. D1 T2w FS absence documented.*
