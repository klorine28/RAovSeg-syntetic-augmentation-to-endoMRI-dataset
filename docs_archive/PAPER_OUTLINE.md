# Paper outline

*Working title*: "Synthetic augmentation for endometriosis ovary segmentation: a
two-phase negative result and its preprocessing / architectural lessons."

*Target*: ~2026-07-13 draft.

*Story in one sentence*: We tested whether conditional DDPM synthesis can
augment a data-starved ovary segmenter (RAovSeg, DSC baseline 0.290 on 8
sacred test subjects); across a 2×2 generator ablation and a cross-domain
extension, no configuration produces synth that improves downstream DSC —
and the diagnostic path yields a set of preprocessing / architectural
lessons for the field.

---

## Structure

### 1. Introduction (~1 page)

- Problem: pelvic MRI ovary segmentation for endometriosis; clinical value
  of automation; data scarcity (30 D2 training subjects → DSC ceiling).
- Idea: use conditional DDPM to synthesise additional training data with
  controlled anatomy (SPADE label conditioning).
- Contributions:
  1. A clean 2×2 generator ablation (concat vs SPADE × no-GAN vs PatchGAN)
     with per-organ CLR metrics.
  2. Empirical negative result — synth augmentation **hurts** RAovSeg DSC
     under every configuration tested.
  3. Diagnostic decomposition — why synth-for-segmentation-augmentation is
     harder than usually assumed; preprocessing pipeline alignment is
     necessary but not sufficient.

### 2. Related work (~0.5 page)

- Medical image synthesis for segmentation augmentation (Diakogiannis,
  MONAI Generative examples).
- Cross-domain MRI translation (Pix2Pix, CycleGAN).
- Diffusion models for medical imaging (Med-DDPM, RoentGen).
- Note: most positive results are on large-domain tasks (CT, chest X-ray);
  pelvic MRI at n=30 is an underexplored regime.

### 3. Datasets and downstream pipeline (~1 page)

- UT-EndoMRI overview (D1_MHS, D2_TCPW). Cite dataset paper.
- RAovSeg architecture: ResClass (slice classifier) + AttUSeg (Attention
  U-Net). Focal Tversky loss. Baseline DSC 0.290 ± ? on 8 test subjects.
- Preprocessing pipeline: percentile clip + minmax + ovary-specific
  intensity enhancement (voxels in [0.22, 0.30] → 1). This last step
  turned out to matter enormously for synth compatibility (§Discussion).

### 4. Generator (~1.5 pages)

- Backbone: MONAI Generative 2D UNet, 2D axial slices, 512×512, SPADE
  variant at bottleneck + every decoder ResBlock.
- Conditioning: 6-channel one-hot labels (bg, uterus, L-ov, R-ov, em,
  body_other).
- Body-centered resampling: body bbox → 512×512 with 5% margin. Per-subject
  in-plane spacing.
- Discriminator (1c only): PatchGAN, spectral norm, warmup 0→10k, ramp
  10k→30k, λ_peak = 0.01.
- Training: 100k steps, AdamW lr=1e-4, EMA=0.9999, CFG dropout 0.1,
  guidance scale 2.0, DDIM 100 steps.

Source refs: EXP1A_NOTES.md, EXP1B_NOTES.md, EXP1C_SUMMARY.md.

### 5. Phase 1 results — generator quality (~2 pages)

- Sample grids: 2×2 comparison at step 85k.
- FID, hist_KL, CLR / OSI metrics table (from RESULTS_2x2.md).
- SPADE γ maps show per-organ localisation improves with PatchGAN; concat's
  low CLR (0.03) predicts its downstream failure.
- Explainability panels (SHAP, counterfactual ablation, per-timestep
  snapshots) — cite explain.py output.

### 6. Phase 1 downstream — RAovSeg augmentation (~2 pages)

- Setup: mix 16-19 synthetic volumes into RAovSeg's 30 train_val subjects
  (ratio ~0.5:1), evaluate on 8 sacred test subjects.
- v1 (no preprocessing fixes): concat 0.150 ± 0.006; SPADE 0.138 ± 0.049.
- v2 (three preprocessing fixes — body silhouette mask, histogram match,
  resample to source): concat 0.044 ± 0.039; SPADE 0.169 ± 0.037.
- v3 (Path B label-aware ovary intensity rescale, t=0.26): SPADE 0.218 ± 0.057
  at n=3; **variance study at n=8 revises to 0.178 ± 0.054**.
- Per-subject analysis: D2-005 and D2-023 are DSC=0 across all 8 seeds —
  structural failures.

Source ref: RAOVSEG_AUGMENTATION_EXPERIMENT.md §8f, §8g.

### 7. Phase 2 — cross-domain extension (~1.5 pages)

- Motivation: leverage D1 (T2, 51 subjects) as generator training pool,
  discriminator anchored on D2 T2FS.
- Setup: dual dataloader in train.py; unconditional D to avoid
  label-distribution shortcut across cohorts; λ_peak=0.01 (later 0.05).
- **exp2 result**: synth quality plateaued at "gray blob" — no distinct
  T2FS style acquired. Downstream DSC = **0.020 ± 0.010 (n=3)** —
  catastrophic collapse, −93% vs baseline 0.290.
- **exp2_lam05 result** [PENDING — Track 2 SLURM still running].
- Diagnosis: DDPM MSE reconstruction loss on D1 T2 dominated adversarial
  signal from D2 discriminator; the two objectives are antagonistic and
  the balance found via λ_peak=0.01 (or 0.05) does not favour cross-domain
  style transfer.

### 8. Discussion (~1.5 pages)

Four headline claims:

1. **Bad synth is worse than no synth** — Phase 2's −93% DSC (0.020 vs
   baseline 0.290) demonstrates that a mediocre generator doesn't just
   waste synth capacity; it corrupts the real signal. This is the
   sharpest lesson from the two-phase study.
2. **Concat conditioning is architecturally unable to benefit downstream
   segmentation augmentation.** Low CLR (~0.03) means the label-aware
   preprocessing fixes have nothing anatomical to align. Confirmed across
   Phase 1 v1/v2/v3.
3. **SPADE conditioning enables preprocessing-aware augmentation but does
   NOT close the gap to real-only baseline at this data scale.** After
   four rounds of preprocessing fixes, best Phase 1 SPADE DSC is 0.178
   (n=8), 38% below baseline. Cross-domain (Phase 2) makes it worse.
4. **Preprocessing pipeline alignment matters more than raw synth
   quality.** FID and hist_KL don't predict downstream success. What
   matters: FOV match, body silhouette cleanup, intensity distribution
   match, and (especially) label-aware ovary intensity targeting.

Wider implication: synth-for-segmentation-augmentation at n<50 real
subjects is an extremely hard regime, and the diffusion + adversarial
paradigm — as commonly deployed — is not sufficient. Practitioners
should validate downstream utility before deploying synth augmentation
in data-scarce clinical tasks.

### 9. Limitations (~0.5 page)

- Small test set (8 sacred D2 subjects) → wide DSC confidence intervals.
- 2D axial slice synthesis limits 3D coherence (ISCS partly addresses).
- Single downstream architecture tested (RAovSeg); other segmenters may
  respond differently.
- Compute limited to ~10 GPU-days total.

### 10. Conclusion (~0.5 page)

- Systematic 2×2 generator ablation + variance study + cross-domain
  extension all point to the same conclusion: naive synth augmentation
  hurts in this data-scarce clinical regime.
- Preprocessing pipeline design and awareness of downstream consumer
  assumptions is the underappreciated lever.
- Future work: paired-cohort image-to-image translation (Pix2Pix on D2's
  T2 + T2FS pairs), semi-supervised RAovSeg pretraining, larger real-data
  collection.

---

## Figures needed

| # | Content | Source |
|---|---|---|
| 1 | System diagram: DDPM → assembly → RAovSeg augmentation flow | new |
| 2 | 2×2 sample grid at step 85k (concat vs SPADE × no-GAN vs PatchGAN) | 1a/1b/1c samples/ |
| 3 | Quality metrics table (FID, hist_KL, CLR, OSI) | RESULTS_2x2.md |
| 4 | SPADE γ maps: does adversarial unlock per-organ localisation? | explain.py |
| 5 | RAovSeg pipeline diagram | RAovSeg paper Fig X |
| 6 | v1→v2→v3 trajectory: preprocessing fix effect on downstream DSC | this work |
| 7 | v3 SPADE at n=8: distribution of DSC across seeds and per-subject | log parsing |
| 8 | exp2 samples at step 5k / 30k / 95k showing failure to acquire style | pulled |
| 9 | Final DSC comparison bars: baseline, Phase 1 (all versions), Phase 2 | this work |

## Tables needed

| # | Content | Status |
|---|---|---|
| 1 | Dataset characteristics (D1/D2 sizes, subjects, modalities) | done |
| 2 | Generator config comparison (concat vs SPADE, no-GAN vs PatchGAN) | done |
| 3 | Phase 1 v1/v2/v3 DSC × concat/SPADE (RAOVSEG_AUG_EXP §8g.3) | done |
| 4 | Phase 2 exp2 / exp2_lam05 DSC | PENDING |
| 5 | Per-subject DSC across n=8 seeds | done (§8g.2) |

## Source materials

Documents (read in order):

- `RESULTS_2x2.md` — Phase 1 generator quality
- `EXP1B_SUMMARY.md`, `EXP1C_SUMMARY.md` — per-experiment context
- `RAOVSEG_AUGMENTATION_EXPERIMENT.md` §§1–8g — Phase 1 downstream
- `RAOVSEG_AUGMENTATION_EXPERIMENT.md` §8h (to be added) — Phase 2 downstream
- `NEXT_STEPS.md` — status snapshot

Data:

- `runs/exp1a`, `runs/exp1b`, `runs/exp1c_concat`, `runs/exp1c_spade` on HPC
- `runs/exp2_d1_gen_d2_disc`, `runs/exp2_lam05` on HPC
- `runs/raovseg_aug_spade_seed{0..7}` for variance study logs

Metrics registries:

- `master_metrics.csv` — Phase 1 generator quality metrics
- SLURM `logs/raov_aug_*_s*.out` — RAovSeg DSC results (grep pattern in
  variance-study section)

---

## Next actions (writing side, parallel to Tracks 1/2)

1. Draft Sections 1–2 from scratch (highest cost, doesn't depend on more
   results).
2. Copy Sections 4–6 from existing MDs (mostly done material).
3. Wait for Track 1 DSC to fill Section 7 numbers.
4. Discussion + Conclusion last.

If pressed on time, submit with §7 as-is (exp2 samples + downstream DSC)
even if Track 2 tuning is inconclusive.
