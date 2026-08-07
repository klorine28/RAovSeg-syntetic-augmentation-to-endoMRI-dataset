# ROUGH WORKING DRAFT — reframed dissertation (info-dump, not polished)

> This is a brain-dump skeleton in the new narrative spine:
> **"in low-data medical augmentation, alignment to the downstream segmenter's
> pipeline assumptions matters more than conventional image realism."**
> Everything below is rough. Fragments, not sentences. Numbers are here so
> you don't have to look them up. FIG/TABLE tags: (HAVE) = already in the doc,
> (NEW) = need to make. Rewrite in your own voice.

---

## CH1 — INTRODUCTION (dump)

- endo = chronic, endometrial-like tissue outside uterus, ~10% reproductive-age women, pain + subfertility + QoL. (Zondervan 2020)
- diagnosis problem: gold standard = laparoscopy = surgery. 6–10 yr delay. (Goldstein & Cohen 2023)
- 3 non-invasive routes, we pick imaging: symptoms (ML on questionnaires, AUC 0.94, Goldstein&Cohen) / biomarkers (Nisenblat 2016 cochrane, CA-125 low spec; Vanhie 2024 miRNA promising not validated) / imaging = only one giving spatial, repeatable measurement.
- MRI > US/CT: soft-tissue contrast, no radiation. ovary segmentation = the quantitative anchor. BUT ovary tiny (<1% of frame), low contrast, data-scarce.
- RAovSeg (Liang 2025) = our downstream anchor. 2-stage (ResClass gate + AttUSeg). DSC 0.290 on 8 sacred D2 test. inter-rater ceiling 0.48 ± 0.24. trained on 30. gap 0.290→0.48 = headroom.
- lever in data-starved regime = synthetic augmentation. recipe works elsewhere: Med-DDPM (Dorjsembe 2024), RoentGen (Chambon 2022), Pix2Pix (Zhu 2017). BUT those cluster at n>100. our regime n=30. unknown.
- **REFRAME — state the thesis up front here, not "answer is no":** the question isn't only "does it work" but "what decides whether it works." finding = fidelity of synth to downstream pipeline assumptions > raw realism. promote this to the front.
- contributions (foreground novel ones): (1) CLR metric predicts downstream utility where FID/hist_KL don't; (2) Path B label-aware ovary-intensity rescale — single most impactful fix; (3) variance protocol n>=5 surfacing per-subject failures; (4) the empirical characterisation itself = a map of where/why the recipe breaks at n<50.
- [FIG 1.1 (NEW)] conceptual schematic: synth image → downstream preprocessing (framing, silhouette, [0.22,0.30] enhancement) → segmenter. show that a "realistic-looking" synth can still fail if it misses these gates. THIS is the whole thesis in one picture. worth making.
- report structure para at end.

## CH2 — LIT REVIEW (dump; mostly exists, just tilt toward the thesis)

- clinical context + diagnostic routes (2.2) — already reframed, keep.
- pelvic MRI + T2 vs T2FS (2.3). RAovSeg operates T2FS (ovary contrast higher). cross-cohort: D2 T2FS, D1 T2 only → sets up phase 2 style-transfer problem.
- segmentation task (2.3.2): U-Net (Ronneberger 2015), Attention U-Net (Oktay 2018), nnU-Net (Isensee 2021, DSC 0.272 ~ RAovSeg 0.290 → data not arch limits). RAovSeg detail: ResClass, AttUSeg, Focal Tversky (Abraham&Khan 2019). full 0.290 vs 0.013 w/o gate.
- **intensity non-standardness (2.3.3)** — LEAN INTO THIS for reframe. MR intensity relative (Nyúl 2000). normalization choice affects DL seg (Ghazvanchahi 2024). RAovSeg's [0.22,0.30] enhancement = hard-coded assumption. foreshadow: this is the "assumption" the thesis is about.
- data scarcity (2.4): n<50 sub-regime (Zhang D. 2022). consequences: DSC ceilings low, per-subject var dominates, universal-failure subjects. classical aug (RandAffine) + model-based aug (Yi 2019 review; positive at n>100, mixed at n<50).
- **NEW ANGLE for reframe:** add a short passage on the gap between generative-eval metrics (FID/LPIPS = "realism") and downstream task utility. the field evaluates generators on distributional realism; almost nobody checks pipeline-fit. this is the gap you fill.
- generative modelling + conditioning (2.5): GAN/VAE/DDPM. why DDPM (stable low-n, quality, conditioning). concat vs SPADE (Park 2019). adversarial (PatchGAN Isola 2017, spectral norm Miyato 2018, DDGAN Xiao 2022).
- cross-domain translation (2.6): paired (Pix2Pix) vs unpaired (CycleGAN Zhu, MUNIT Huang). Yang 2020 T1↔T2, Wolterink 2017 CT↔MRI. no paired T2/T2FS in UT-EndoMRI → phase 2 forced one-way.
- gap + positioning: DDPM aug at n<50 under-evaluated; concat vs SPADE not quantitatively compared; downstream-preprocessing-aware eval missing → CLR.
- [FIG 2.1 DDPM process (HAVE)]
- [TABLE 2.1 (NEW)] survey table: generative-augmentation studies × training-n × downstream outcome. columns: paper / modality / n_real / generator / downstream Δ. shows positive results cluster n>100, mixed/negative <50. directly supports the gap + positions your negative result as expected, not anomalous. HIGH VALUE, make this.

## CH3 — METHODOLOGY (dump; exists)

- dataset UT-EndoMRI: D1_MHS (Mem Hermann, T2, 51, bright fat), D2_TCPW (TCPW, T2FS, ~73, dark fat). per-organ masks. L/R ovary split by connected components. [TABLE 3.1 dataset (HAVE)]
- filters D2: missing T2FS (3), missing ut/ov mask (~9), sacred test (8) → 32 generator pool; RAovSeg stricter (no em/cyst) → 30 train + 8 test. [TABLE 3.2 filters (HAVE)]
- 6-channel one-hot label @512². why 6 not 5 (body_other channel fixed noisy grey edges). [FIG 3.3 label (HAVE)]
- generator: 2D UNet from MONAI Generative (Pinaya 2023). 2×2 ablation concat/SPADE × ±PatchGAN. [FIG 3.4 sys arch (HAVE), FIG 3.5 concat vs SPADE (HAVE), TABLE arch components (HAVE)]
- SPADE needs zero-init γ/β (else diverges). concat = 7-ch input, label propagates as feature data, CLR 0.03–0.07.
- PatchGAN 70×70, spectral norm. joint DDPM+adv loss, x̂0 estimate. λ schedule (peak 0.01). [FIG 3.6 ISCS (HAVE)]
- training: AdamW lr 1e-4 batch 4 100k steps ~10h A100. CFG (10% drop, g=3 concat / g=2 SPADE). EMA 0.9999. ISCS α=0.8. ablation parity principle.
- **preprocessing mismatch (3.6) — CENTRAL TO REFRAME.** generator body-centered (body ~90% frame) vs RAovSeg image-centered (body ~55–60%, black border). the [0.22,0.30] enhancement rule: sets those voxels to 1, ovary = brightest by construction. IF synth ovary intensity misses window → enhancement doesn't fire on synth → segmenter trains on synth where ovary invisible → aug defeated. [FIG 3.x (NEW, or reuse 4.7)] real ovary intensity histogram with [0.22,0.30] band drawn = shows why the window works on real. good to have early.
- assembly fixes flags (body mask / hist match / resample-to-source / ovary-target-intensity). [TABLE fixes (HAVE)]
- Path B: --ovary-target-intensity 0.26 = force ovary voxels into window via per-volume offset.
- Phase 2 config: gen on D1 T2, disc on D2 T2FS unconditional (label zeroed so D judges pure style). 32 synth resampled to D1 frames.
- metrics: FID (N=256, noise ±30), hist_KL, LPIPS-NN (>=0.34 = no memorisation), **CLR** (counterfactual: zero a label channel, regen same noise, fraction of change inside that channel's mask → 1 local, 0 global), **OSI** (SPADE γ per-organ correlation). downstream DSC on 8 sacred. variance protocol n=8 seeds.
- RAovSeg recreation validation: 0.220 ± 0.29 vs paper 0.290 (gap 0.07). no_pp reproduces within 0.05. irreproducibility sources: Zenodo 73 vs table 77, no split file (seed-42), 5mm spacing inconsistent. ResClass criticality + postproc didn't fully reproduce. report aug vs BOTH baselines 0.290 and 0.220. [FIG 3.7 RAovSeg pipeline (HAVE), FIG 3.8 baseline reproduction (HAVE)]

## CH4 — RESULTS (dump; exists)

### Phase 1 generator quality
- 2×2, no single winner. FID: 1c_concat best 166, rest 188–200. hist_KL: 1c_concat 5.79. LPIPS: 1c_spade 0.699. CLR: SPADE 0.30–0.53, concat 0.01–0.08. OSI: SPADE γ per-organ (organ_corr ~0.25, body ~0). [FIG 4.1 quality 2×2 (HAVE), 4.2 CLR (HAVE), 4.3 heat matrix (HAVE), 4.4 2×2 map (HAVE), 4.5 counterfactual ablation (HAVE), TABLE master metrics (HAVE)]
- PatchGAN asymmetric: delivers what arch lacks (concat←texture/FID, SPADE←perceptual/LPIPS). not a generic realism booster.
- **THESIS LINCHPIN FIGURE [FIG 4.NEW]:** two scatter panels — (a) FID vs downstream DSC = no/weak correlation; (b) CLR vs downstream DSC = clear correlation. one image proving "conventional realism doesn't predict utility, alignment does." IF you make one new figure, make this one.

### Phase 1 downstream v1→v2→v3
- setup: 30 synth (v1 only 16–19 assembled, SLURM timeout — FLAG as limitation) + 30 real, test 8, n=3 seeds.
- v1 (no fixes): SPADE 0.138 ± 0.049, concat 0.150 ± 0.006. both ~half baseline.
- diagnostic: (A) FOV mismatch (synth 90% vs real 55–60%), (B) outside-body hallucinations amplified by clip, (C) intensity-enhancement failure (synth ovary outside [0.22,0.30]). [FIG 4.6 real vs synth (HAVE), 4.7 ovary intensity (HAVE), 4.8 body intensity (HAVE)]
- v2 (3 fixes: body mask, hist match, resample): SPADE 0.169 (+22%), concat COLLAPSES 0.044 (seed2 = 0.000 everywhere). why: hist match is rank-based not semantic → for SPADE bright pixels near ovary (CLR high) so lands near window; for concat bright pixels random (CLR low) → segmenter trained on ovary-at-wrong-locations. [FIG 4.9 per-fix effect (HAVE)]
- **this is the cleanest reframe evidence**: SAME fix helps SPADE, breaks concat, purely because of localisation (CLR). alignment > realism.
- v3 (Path B t=0.26): SPADE 0.218 (+58% vs v1, seed0 hit 0.276 = within 0.014 of baseline!). concat stuck ~0.05. [FIG 4.10 trajectory (HAVE), TABLE v3 (HAVE)]
- Path B works only when generator can localise (SPADE yes, concat no).
- diagnostics: target sweep t=0.22→0.165, t=0.28→0.189, skip-enhancement (opt C) 0.170 < applying it. ceiling ~0.17–0.22 for SPADE at this quality (FID~188).

### Variance study n=8
- v3 SPADE n=3 0.218 → n=8 0.178 ± 0.054. the 0.218 was luck (extra 5 seeds avg 0.154). gap 0.11 = 2× cross-seed std. per-subject std ~0.24 >> cross-seed 0.054 (4×). universal failures D2-005, D2-023 (DSC=0 all 8 seeds). [FIG 4.11 variance (HAVE), 4.12 per-subject (HAVE), 4.13 why 005/023 fail (HAVE)]
- lesson: n=3 insufficient at n<50. **TIER-3 STATS GO HERE:** report 95% CI on 0.178 (~[0.133,0.223]), paired Wilcoxon vs baseline across 8 subjects, medians+IQR. underpowered = honest, supports negative result.

### Phase 2 cross-domain
- exp2: DSC 0.020 ± 0.010 (−93%). gray-blob collapse, no T2FS style acquired. mechanism: DDPM MSE on D1 dominated weak adv signal from unconditional D at λ=0.01. [FIG 4.14 forest (HAVE), 4.15 exp2 output over training (HAVE)]
- **strongest reframe evidence, not embarrassment:** when synth totally misaligned to target domain, downstream implodes. bad synth worse than none.
- exp2_lam05 (λ=0.05): 0.117 ± 0.112, driven by 1 seed (0.246 vs 0.056/0.049), unstable. tuning λ moves it but doesn't close gap.
- exp2_pathC (skip enhancement for D2-9 synth prefix): 0.152 ± 0.054 (+0.132, 7.6× vs exp2), within 0.5σ of Phase 1 SPADE. best seed 0.212. → confirms mechanism is PREPROCESSING mismatch not just gen quality.
- cross-experiment summary vs 0.290 and 0.220. [TABLE cross-exp (HAVE), FIG 4.16 forest (HAVE — ADD CIs)]

## CH5 — DISCUSSION (dump; use new opening)

- **NEW opening para** (the discovery, drafted separately) — lead with "alignment > realism," 4 claims build it.
- Claim 1: bad synth worse than no synth. exp2 0.020, −93%, stable (std 0.010), same failure mode, mechanism understood. field warning: at n<50 aug quality not optional.
- Claim 2: concat architecturally locked out. CLR 0.013–0.069. v1 0.150→v2 0.044→v3 0.053. not rescuable by preprocessing. architectural not preprocessing failure. → §5.2. (fix old "Chapter 6 claim 2" ref, done)
- Claim 3: SPADE approaches, doesn't close. CLR 0.30–0.53, trajectory up to 0.178 ceiling. necessary but not sufficient at n=30.
- Claim 4: **preprocessing pipeline alignment > raw synth quality** = the thesis stated as a claim. FID/hist_KL don't predict; CLR + intensity-window do. if we'd trusted FID we'd have picked 1c_concat (best FID) → −85%.
- meta-lessons: n>=5 seeds; FID doesn't predict utility; cross-domain adv at n<50 insufficient.
- limitations of interpretation: n=8 test wide CIs, D2-005/023 unresolved, single downstream arch.

## CH6 — CONCLUSION (dump; reframe summary)

- **NEW summary opening:** lead with what was learned (alignment is the lever), present "no config beat 0.290" as the evidence. not "we failed."
- contributions (tighten, once each, ranked): CLR; Path B; variance protocol; the characterisation.
- limitations (EXPAND per tier 3): single downstream arch (RAovSeg-specific enhancement rule); n=8 test underpowered (<0.05 DSC undetectable); single T2FS site D2; compute ceiling ~10 GPU-days capped ablation; **v1 synth-assembly shortfall 16–19/30 (SLURM timeout) confounds v1 comparison** — name it.
- future work (reordered): (1) paired T2/T2FS translation (Pix2Pix, 100× stronger signal); (2) semi-supervised pretraining on synth then fine-tune on real; (3) larger real data n=100–200; (4) alt segmenters (nnU-Net, TotalSegmentator) test generality; (5) **NEW — Optuna Bayesian HPO with CLR as cheap surrogate objective** (drafted separately). frame as TEST of whether 0.178 is a true ceiling, routes through CLR.
- closing: the path to positive result at n<50 = alignment + data scale, not more sampling.

---

## FIGURES / TABLES STILL TO MAKE (summary)

HIGH VALUE (make these):
- [FIG 4.NEW] scatter: FID vs downstream DSC (no corr) beside CLR vs downstream DSC (corr). = the thesis in one figure.
- [TABLE 2.1 NEW] survey of generative-aug studies × n_real × downstream outcome (positive cluster >100). positions negative result as expected.
- [FIG 1.1 NEW] conceptual pipeline schematic (synth → downstream gates → segmenter); the "alignment" thesis picture.

MEDIUM / NICE:
- [FIG 3.x NEW or reuse 4.7] real ovary intensity histogram with [0.22,0.30] band, shown early in methods.
- [FIG 4.16 ENHANCE] add 95% CIs to the forest plot.
- [TABLE 4.NEW] headline comparisons with mean, 95% CI, Wilcoxon p vs baseline (ties tier-3 stats).

OPTIONAL:
- [FIG 6.x NEW] Optuna + CLR-surrogate search-loop schematic for future work.

## STILL-NEED-FROM-YOU
- per-subject × per-seed DSC numbers (for the CIs + Wilcoxon) — point me at the metrics/ files and I can compute + write them in.
- confirm the two swapped refs (Yi 2019, Zhang D. 2022) match what you read.
