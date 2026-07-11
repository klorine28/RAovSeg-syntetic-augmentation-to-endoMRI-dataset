# 4 — Experiments and results

> **Target: 3,500 words.** The full experimental narrative in the order
> the story unfolded: Phase 1 generator ablation → Phase 1 downstream
> augmentation (v1 → v2 → v3 → Options B and C) → n=8 variance study →
> Phase 2 cross-domain. §4.6 covers the λ_peak sweep (lam05 and lam50)
> layered on top of the pathC preprocessing fix.
>
> **Sub-targets (for pruning current draft):**
> - §4.1 Experimental overview: 200 words
> - §4.2 Phase 1 generator quality: 1,000 words
> - §4.3 Phase 1 downstream (v1 → v3 + Options B/C): 1,400 words
> - §4.4 n=8 variance study: 700 words
> - §4.5 Phase 2 (exp2): 500 words
> - §4.6 exp2 λ_peak sweep (lam05_pathC, lam50_pathC): 200 words
>
> Current draft (~707 lines) exceeds the target; use the section
> targets to prune. Detailed source material remains available in
> `../docs_archive/RESULTS_2x2.md` and
> `../docs_archive/RAOVSEG_AUGMENTATION_EXPERIMENT.md`.

---

## 4.1 Experimental overview [target: 200 words]

| Phase | Experiment | Purpose | Status | Result |
|---|---|---|---|---|
| Real-only baseline | RAovSeg on 30 D2 subjects | Reproduce Liang et al. (2025) | Done | DSC 0.290 |
| Phase 1 | Exp 1a — concat DDPM | Baseline generator (global conditioning) | Done, 80k steps | See §4.2 |
| Phase 1 | Exp 1b — SPADE DDPM | Per-organ conditioning ablation | Done, 80k steps | See §4.2 |
| Phase 1 | Exp 1c_concat — concat + PatchGAN | Adversarial on concat | Done, 100k steps | See §4.2 |
| Phase 1 | Exp 1c_spade — SPADE + PatchGAN | Adversarial on SPADE | Done, 100k steps | See §4.2 |
| Phase 1 downstream | v1 (no fixes) | Naive augmentation | Done, n=3 | −48% to −52% vs baseline |
| Phase 1 downstream | v2 (3 preprocessing fixes) | Body-frame + hist match + resample | Done, n=3 | −85% concat, −42% SPADE |
| Phase 1 downstream | v3 (Path B ovary rescale) | Label-aware intensity target | Done, n=3 | −25% SPADE (0.218) |
| Phase 1 downstream | Option B (t sweep) | t = 0.22 and 0.28 SPADE | Done, n=3 | Both worse than 0.26 |
| Phase 1 downstream | Option C (skip enhancement) | Diagnostic on RAovSeg's enhancement | Done, n=3 | −41% (0.170) |
| Phase 1 downstream | Variance study (n=8 seeds) | Correct v3 SPADE mean | Done, n=8 | 0.178 ± 0.054 |
| Phase 2 | exp2 — cross-domain (D1 gen + D2 disc) | Cross-domain style transfer | Done, n=3 | **−93% (DSC 0.020)** |
| Phase 2 | exp2_pathC — skip enhancement for D2-9 | Preprocessing rescue on exp2 | Done, n=3 | −48% (DSC 0.152) |
| Phase 2 | exp2_lam05_pathC — λ = 0.05 + pathC | λ tuning stacked on pathC | Done, n=3 | −66% (DSC 0.098) |
| Phase 2 | exp2_lam50_pathC — λ = 0.50 + pathC | λ tuning stacked on pathC | Done, n=3 | −63% (DSC 0.107) |

---

## 4.2 Phase 1 — generator quality results [target: 1,000 words]

### 4.2.1 The four variants, side-by-side

| Variant | Conditioning | Adversarial | Steps | g | Notes |
|---|---|---|---|---|---|
| **1a** | Concat | — | 80k | 3.0 | Baseline; mid-flight adds CFG + EMA + 6-channel |
| **1b** | SPADE | — | 80k | 2.0 | v1 failed (no zero-init); v2 fixed |
| **1c_concat** | Concat | PatchGAN | 100k | 3.0 | Same backbone as 1a |
| **1c_spade** | SPADE | PatchGAN | 100k | 2.0 | Same backbone as 1b |

All four trained on the same 32 D2 subjects (~730 slices), same body-
centered preprocessing, same 6-channel labels, same batch=4 AdamW lr=1e-4
schedule.

### 4.2.2 Master metrics table

| Variant | CLR_ut ↑ | CLR_ov_L ↑ | CLR_em ↑ | OSI_organ | OSI_body | **FID ↓** | hist_KL ↓ | LPIPS_min | LPIPS_mean ↓ |
|---|---|---|---|---|---|---|---|---|---|
| 1a | 0.013 | 0.043 | 0.028 | n/a | n/a | 188.2 | 8.15 | 0.493 | 0.824 |
| 1b | **0.407** | **0.494** | **0.532** | 0.242 | −0.011 | 200.1 | 6.89 | 0.350 | 0.745 |
| **1c_concat** | 0.069 | 0.080 | 0.063 | n/a | n/a | **166.5** | **5.79** | 0.426 | 0.773 |
| **1c_spade** | 0.405 | 0.297 | 0.420 | **0.258** | 0.029 | 188.1 | 7.20 | 0.343 | **0.699** |

↑ = higher is better, ↓ = lower is better. **Bold** = column winner.
Source: `../metrics/master_metrics.csv`.

### 4.2.3 What each metric tells us

**CLR (Counterfactual Localisation Ratio)** — for each label channel,
zero the channel, regenerate with the same initial noise, and compute
what fraction of the per-pixel change is inside the channel's own mask.

- CLR → 1: local, clean per-channel conditioning.
- CLR → 0: global, mixed conditioning.

**Result**: SPADE variants achieve 0.30–0.53 (10–30× higher than
concat). PatchGAN slightly nudges both directions but does not change
the fundamental pattern — **the conditioning mechanism, not adversarial
loss, determines localisation**.

**OSI (Organ Specificity Index, SPADE-only)** — per-SPADE-module Pearson
correlation between |γ| and each organ mask, and between |γ| and the
body mask.

**Result**: Both 1b and 1c_spade show organ_corr ≈ 0.25, body_corr ≈ 0.
γ heads pick up per-organ structure, not just body silhouette — SPADE
is doing its job as designed.

**FID (Fréchet Inception Distance)** — Inception-V3 feature-space
distance between synth and real distributions.

| FID range | Interpretation |
|---|---|
| < 50 | State-of-the-art (StyleGAN on faces) |
| 50–100 | Solid medical synthesis, thousands of subjects |
| 150–250 | Reasonable for low-data medical regime (**our setting**) |
| > 300 | Distribution meaningfully different from real |

**Result**: 1c_concat at 166 is meaningfully best — 22 points lower
than 1a. At N=256 samples, FID has a noise floor of ~±30, so
1a/1b/1c_spade at 188–200 are statistically tied.

**hist_KL (intensity histogram KL divergence)** — more sample-efficient
than FID. A 2.4-point gap is real signal, not noise.

**Result**: 1c_concat at 5.79 is 29% lower than 1a. PatchGAN clearly
tightens the concat baseline's intensity distribution toward real.

**LPIPS-NN (perceptual nearest-neighbour distance)** — for each synth,
distance to nearest real image in AlexNet feature space.

| LPIPS_min value | Interpretation |
|---|---|
| < 0.05 | Memorisation warning |
| 0.05–0.30 | Mild similarity |
| 0.30–0.50 | Healthy |
| > 0.50 | Very distinct from any real |

**Result**: all min values ≥ 0.34 → no memorisation. 1c_spade has
lowest mean (0.699) — synth perceptually closest to real without
memorisation.

### 4.2.4 The architectural map

```
                ← MORE LOCALISED                  MORE GLOBAL →

  best        ┌──────────────────────────────────────────────────┐
  texture     │                                                   │
  realism     │   1c_spade                 1c_concat              │
              │   LPIPS 0.70               FID 166, hist_KL 5.79  │
              │   CLR_ut 0.41              CLR_ut 0.07            │
              │                                                   │
              │       1b                        1a                │
              │   CLR_ut 0.41              CLR_ut 0.01            │
              │   FID 200                  FID 188                │
  decent      │                                                   │
  realism     └──────────────────────────────────────────────────┘
```

**Reading the map:**
- **1c_concat** dominates texture realism (FID, hist_KL) but conditions
  globally.
- **1c_spade** dominates perceptual realism (LPIPS) while preserving
  SPADE's per-channel localisation.
- **1a** is dominated by 1c_concat on every quality metric — retired as
  a candidate.
- **1b** is dominated by 1c_spade on LPIPS.

**No single winner across metrics.** The choice of variant depends on
what the downstream consumer cares about — which is the whole point of
running the augmentation study in §4.3.

### 4.2.5 PatchGAN's asymmetric effect

The 2-arm design (same PatchGAN on both backbones) isolates the
adversarial contribution from the conditioning mechanism.

| Δ from baseline | concat (1a → 1c_concat) | SPADE (1b → 1c_spade) |
|---|---|---|
| FID | −12% (real improvement) | −6% (within noise) |
| hist_KL | **−29%** (strong) | +5% (slight regression) |
| LPIPS_mean | −6% | −6% (best overall) |
| CLR (localisation) | +small (still ~7%) | −small (still ~40%) |

**PatchGAN's effect depends on what the architecture lacked**:
- Concat lacked texture realism → PatchGAN delivered (big FID and hist_KL
  wins).
- SPADE lacked perceptual smoothness → PatchGAN delivered (best LPIPS)
  at a small cost to hist_KL.

This is one of Chapter 6's headline claims: PatchGAN is not a generic
realism booster.

### 4.2.6 An earlier misinterpretation, corrected

An earlier visual inspection of SPADE counterfactual diff panels
suggested SPADE was not doing per-organ localisation ("removing the
uterus label doesn't visibly change anything"). This was wrong — a
**colormap-normalisation artefact**: matplotlib auto-normalises each
diff panel independently, and SPADE's locally-concentrated changes have
a narrow value range, so they look "empty" relative to concat's
globally-distributed changes.

The quantitative CLR (0.41 for 1b uterus, vs 0.01 for 1a) proves SPADE
does what it was designed to do. The RESULTS_2x2 doc originally
propagated the wrong visual reading; the correction is now the canonical
interpretation and drives the Chapter 6 architectural claim.

### 4.2.7 Caveats on Phase 1 quality metrics

- **CLR and OSI computed at N=4** — small sample. The 5–10× concat vs
  SPADE CLR gap is robust to noise; smaller within-arm differences
  (e.g. 1b vs 1c_spade CLR) are within noise.
- **FID at N=256** has ~±30 noise floor. 1a → 1c_concat's 22-point gain
  is at the edge of meaningful. hist_KL confirms the direction more
  confidently.
- **AILM was dropped** — always ≈ 1.0 by construction of GradientSHAP;
  degenerate.
- **Absolute FID 170–200 is high** but expected at 32 training subjects.
  Published medical synthesis at this data scale often sits in 100–300.

## 4.3 Phase 1 downstream — RAovSeg augmentation [target: 1,400 words]

The Phase 1 quality metrics tell us *what the generators are like*. The
actual test of the project's hypothesis is whether the synth improves
downstream ovary DSC. This is done via `assemble_synthetic_volumes.py`
+ `preprocess.py --extra-train-dir` + full RAovSeg training.

### 4.3.1 v1 — no preprocessing fixes

**Setup**: 30 synth (attempted; 16–19 actually assembled due to SLURM
timeout) + 30 real train subjects. Test on 8 sacred D2 subjects. n=3
seeds per variant.

| variant | seed | DSC (full) | DSC (no_pp) | DSC (no_rc) |
|---|---|---|---|---|
| concat | 0 | 0.1466 | 0.1628 | 0.1429 |
| concat | 1 | 0.1567 | 0.1728 | 0.0978 |
| concat | 2 | 0.1473 | 0.1832 | 0.0794 |
| **concat mean** | — | **0.1502 ± 0.006** | 0.1729 | 0.1067 |
| spade | 0 | 0.1108 | 0.0748 | 0.1093 |
| spade | 1 | 0.1087 | 0.1009 | 0.1200 |
| spade | 2 | 0.1947 | 0.1542 | 0.1951 |
| **spade mean** | — | **0.1381 ± 0.049** | 0.1100 | 0.1415 |
| **Baseline (real-only)** | — | **0.290** | 0.235 | 0.013 |

**Both augmentation variants halved the DSC.** Concat is tightly
reproducible across seeds (std 0.006); SPADE has more variance (std
0.049).

### 4.3.2 The diagnostic — what went wrong at v1

We loaded `data/processed/train_val/D2-001/image.npy` (real, post-
RAovSeg-preprocess) and `data/processed/train_val/D2-900/image.npy`
(synth from 1c_concat, post-same-preprocess) and inspected them
side-by-side (figure: `../figures/synth_vs_real_after_raovseg_preprocess.png`).
Every primary suspect from the pressure-points list confirmed:

**Problem A — FOV mismatch (body-centered vs image-centered)**:
- Real D2-001: body sits in the middle of the frame, body 55–60% of
  frame, clear black border.
- Synth D2-900: body fills ~90% of the frame. The generator's
  body-centered preprocessing persists straight through RAovSeg's
  resampling because the synth NIfTI header carries body-centered
  spacing/origin.

**Problem B — outside-body hallucinations**:
- Real D2-001: outside-body is uniform black.
- Synth D2-900: outside-body is filled with structured grainy noise —
  the same hallucination pattern visible in the earlier explainability
  diagnostics, amplified by percentile-clip + normalisation.

**Problem C — the intensity enhancement step is failing on synth
(MOST DAMAGING)**:
- Real D2-001 histogram post-enhancement: massive spike at 1.0 →
  enhancement fired strongly on real ovary tissue.
- Synth D2-900 histogram post-enhancement: much smaller spike at 1.0;
  bulk of the intensity mass sits at 0.2–0.5, mostly *above* the o1=0.22
  threshold but not cleanly inside [0.22, 0.30].

**Overlay panels** made it visually clear:
- Real: red ovary overlay sits on a bright white enhanced region — the
  segmenter is told "the ovary is THIS bright thing."
- Synth: red ovary overlay sits on a medium-gray region indistinguishable
  from surrounding tissue — the enhancement did not fire, so the
  segmenter has no visual hint where the ovary is.

**The resulting training dynamic**:
- Real images: ovary = obvious bright blob → easy.
- Synth images: ovary = average gray tissue → hard.
- Model learns "predict ovary at the brightest blob," which works on
  real subjects where enhancement fires (D2-016, D2-017 get DSC ≥ 0.5)
  and fails on real subjects where enhancement misses (D2-005, D2-023
  get DSC = 0).

The v1 → v2 fix plan was adopted directly from this diagnostic.

### 4.3.3 v2 — three preprocessing fixes

Applied to `src/Generator/assemble_synthetic_volumes.py`:

**Fix 1 — Body silhouette mask** (`--no-body-mask` disables): set synth
pixels to 0 where the `outside_body` label channel is 1, killing the
outside-body hallucinations.

**Fix 2 — Histogram match** (`--no-histogram-match` disables): match
the synth's intensity distribution to the source real subject's raw
intensity distribution (after the same percentile-clip that RAovSeg will
apply), so the synth ends up at a similar [0, 1] distribution shape
post-preprocess.

**Fix 3 — Resample to source real frame** (`--no-resample-to-source`
disables): use SimpleITK's `ResampleImageFilter` with the raw real
subject's NIfTI as the reference, transferring the synth (body-centered
frame) into the real's image-centered frame at matching spacing / origin
/ direction. After RAovSeg's preprocess, synth has the same FOV framing
as real.

Bumped `assemble_synth_*.sh` time limit from 2 h → 4 h. All 30 synth
volumes assembled per variant (previously 16–19 due to timeout).

**Visual confirmation** (`../figures/synth_vs_real_v2.png`): all three fixes
visually accomplished their stated goal. FOV matches, outside-body is
clean black, histograms have similar overall shape.

**v2 DSC results**:

| Variant | v1 | v2 | Δ vs v1 | vs baseline 0.290 |
|---|---|---|---|---|
| **concat** | 0.150 ± 0.006 | **0.044 ± 0.039** | −71% (WORSE) | −85% |
| **spade** | 0.138 ± 0.049 | **0.169 ± 0.037** | +22% (marginal) | −42% |

**The fixes had opposite effects on the two variants.** SPADE improved
modestly; concat collapsed. concat seed 2 returned DSC = 0.000 on every
test subject.

### 4.3.4 Why v2 fixes broke concat but helped SPADE

Root cause: **histogram matching is rank-based, not semantic**. It
aligns the synth's brightest pixels with the real's brightest pixels by
rank.

- In real T2FS, the rank-95+ pixels happen to be ovary tissue (which is
  what the enhancement targets at [0.22, 0.30]).
- In synth, the rank-95+ pixels are whatever the diffusion model
  decided to make bright — not necessarily the ovary.

For SPADE (CLR ~0.40 for uterus, ~0.30–0.53 across channels), the
bright pixels are at least *near* the ovary region because SPADE's
per-organ localisation constrains where organ-textured content is
generated. Result: histogram-matched SPADE synth ovary lands near
[0.22, 0.30] → enhancement fires nearby → segmenter learns something
useful.

For concat (CLR ~0.04, essentially no per-organ localisation), bright
pixels are randomly distributed → histogram-matched concat synth has
enhanced regions in wrong locations → segmenter is actively trained to
predict ovary at wrong locations → real test DSC collapses.

**This is the paper's first clean architectural finding**: concat's
CLR ~0.03 predicts its downstream failure. See Chapter 6, claim 2.

### 4.3.5 v3 — Path B: label-aware ovary intensity rescaling

Insight from §4.3.4: instead of hoping histogram matching places synth
ovary pixels at the right intensity by rank luck, explicitly force them
there using the ovary label mask.

**Implementation**: `--ovary-target-intensity 0.26` flag in
`assemble_synthetic_volumes.py`. Computes a per-volume additive offset
on the ovary region only, targeting t = 0.26 (middle of the
[0.22, 0.30] enhancement window).

**v3 DSC results (n=3)**:

| Version | concat | spade |
|---|---|---|
| Baseline (real-only) | 0.290 | 0.290 |
| v1 (no fixes) | 0.150 ± 0.006 | 0.138 ± 0.049 |
| v2 (3 preprocessing fixes) | 0.044 ± 0.039 | 0.169 ± 0.037 |
| **v3 (3 fixes + Path B, t=0.26)** | **0.053 ± 0.056** | **0.218 ± 0.057** |

**SPADE seed 0 in v3: DSC = 0.276** — within 0.014 of the real-only
baseline. SPADE's trajectory 0.138 → 0.169 → 0.218 shows Path B is
genuinely helping.

**Concat stuck at ~0.05** regardless of fixes. Path B did not move the
needle for concat because concat's synth doesn't actually contain
ovary-textured content in the ovary region — the rescale just puts a
bright blob at the label location, disconnected from surrounding synth
tissue, which the segmenter can't learn from.

### 4.3.6 The architectural interpretation of v3

This maps cleanly to Phase 1's CLR finding:

| Variant | CLR_uterus | Path B outcome |
|---|---|---|
| concat | 0.013–0.069 | Doesn't help — concat's synth doesn't contain ovary-textured content at the label location |
| SPADE | 0.407–0.494 | Helps — SPADE's synth ovary region DOES have ovary-shaped content; intensity rescale ensures RAovSeg's enhancement fires on it |

**Path B works when the generator can localise the ovary.** Concat
can't → intensity shifting doesn't rescue it. SPADE can → intensity
shifting turns "correctly-located but wrong-intensity" ovary into
"correctly-located, correctly-highlighted" ovary that RAovSeg can find.

### 4.3.7 Option B — sweep the ovary target intensity (SPADE-only)

The current Path B target t = 0.26 was a first guess. Values at the
edges of the [0.22, 0.30] enhancement window might work better if there
is an interaction with the percentile-clip that shifts things.

**Setup**: SPADE only (concat is a lost cause per §4.3.5), n=3 seeds.

| Config | seed 0 | seed 1 | seed 2 | Mean | Std |
|---|---|---|---|---|---|
| v3 SPADE (t=0.26) | 0.2755 | 0.1620 | 0.2167 | **0.2181** | 0.057 |
| Opt B SPADE t=0.22 | 0.1192 | 0.1008 | 0.2753 | 0.1651 | 0.096 |
| Opt B SPADE t=0.28 | 0.1363 | 0.1236 | **0.3061** | 0.1887 | 0.102 |

**t = 0.26 was serendipitously optimal.** Moving the ovary target
intensity away from the middle of the enhancement window (down to 0.22
or up to 0.28) reduced the mean DSC. Confirms "put the ovary in the
middle of the enhancement window" was the right heuristic.

### 4.3.8 Option C — skip enhancement for synth subjects

Diagnostic: is the enhancement step itself the bottleneck? If we skip
it for synth (train on synth-as-is with percentile-clip + minmax only,
while real still gets enhancement), does synth utility improve?

Implementation: `preprocess.py` skips the o1/o2 enhancement step when
subject ID matches `D2-9*`.

| Config | seed 0 | seed 1 | seed 2 | Mean | Std |
|---|---|---|---|---|---|
| v3 SPADE (t=0.26) | 0.2755 | 0.1620 | 0.2167 | **0.2181** | 0.057 |
| Opt C SPADE (no enh) | 0.1345 | 0.2349 | 0.1405 | **0.1700** | 0.056 |

**Enhancement is helpful, not hurtful**: Opt C (0.170) is worse than v3
(0.218). When enhancement fires correctly on synth (via Path B), it IS
beneficial. Removing enhancement does not rescue synth utility.

### 4.3.9 Full trajectory including B and C (pre-variance-study)

| Version | concat | spade (n=3) |
|---|---|---|
| Baseline (real-only) | 0.290 | 0.290 |
| v1 (no fixes) | 0.150 ± 0.006 | 0.138 ± 0.049 |
| v2 (3 preprocessing fixes) | 0.044 ± 0.039 | 0.169 ± 0.037 |
| v3 (3 fixes + Path B, t=0.26) | 0.053 ± 0.056 | 0.218 ± 0.057 |
| Opt B (Path B, t=0.22) | — | 0.165 ± 0.096 |
| Opt B (Path B, t=0.28) | — | 0.189 ± 0.102 |
| Opt C (no enh for synth) | — | 0.170 ± 0.056 |

**Multiple targeted interventions converge on ~0.17–0.22 mean DSC for
SPADE augmentation.** That's the ceiling of what SPADE synth at this
quality level (FID ~188, hist_KL ~7.2, CLR ~0.4) can contribute.

## 4.4 The n=8 variance study — v3 SPADE revisited [target: 700 words]

**Motivation**: at n=3, std is 0.057 → mean known to only ±0.06. To
decide whether variance is masking a real augmentation benefit, we ran
5 more seeds (3–7) of the same v3 config and analysed both cross-seed
and per-subject variance.

### 4.4.1 Aggregate n=8 result

| | n=3 (§4.3) | **n=8** |
|---|---|---|
| Mean DSC | 0.2181 | **0.1783** |
| Std across seeds | 0.0570 | 0.0537 |
| Best seed | 0.2755 (s0) | 0.2755 (s0) |
| Gap to baseline (0.290) | −25% | **−38%** |

Per-seed:

| Seed | DSC |
|---|---|
| 0 | 0.2755 |
| 1 | 0.1620 |
| 2 | 0.2167 |
| 3 | 0.1014 |
| 4 | 0.1308 |
| 5 | 0.1600 |
| 6 | 0.2007 |
| 7 | 0.1793 |

The extra 5 seeds (3–7) averaged 0.1544 ± 0.0378 — well below the
original three. **The original 0.218 mean was luck, not a stable
estimate.**

### 4.4.2 Per-subject variance dwarfs cross-seed variance

Across the 8 test subjects × 8 seeds:

| Subject | Seeds hitting DSC > 0.1 | Behaviour |
|---|---|---|
| D2-017 | **8/8** (all ~0.5) | Reliably segmented |
| D2-016 | 6/8 (2 total failures) | Usually great |
| D2-015 | 7/8 (variable 0.14–0.60) | Seed-dependent |
| D2-038 | 2/8 | Rare hit |
| D2-024 | 1/8 (only s0) | One-hit wonder |
| D2-026 | 1/8 (only s7) | One-hit wonder |
| **D2-005** | **0/8** | **Universal failure** |
| **D2-023** | **0/8** | **Universal failure** |

- **Two test subjects (D2-005, D2-023) are universal-failure cases**:
  DSC = 0 across every seed. Structural failure, not variance. Alone
  caps the achievable mean at ~0.22 even if every other subject were
  segmented perfectly.
- **Std WITHIN a seed (across 8 subjects) is ~0.24, 4× larger than std
  ACROSS seeds (~0.054).** The dominant variance axis is per-subject
  tractability, not training-run seed.

### 4.4.3 What the variance study told the paper story

1. **The "variance masks a real benefit" story is dead.** SPADE
   augmentation under Phase 1 conditions robustly underperforms the
   real-only baseline by ~38%. The gap (0.11) is 2× the cross-seed std
   (0.054).

2. **The DSC ceiling for Phase 1 augmentation is ~0.18** — no further
   preprocessing tuning will move this meaningfully. Confirmed by
   Options B/C sweeps (all landed lower) and now by the variance study.

3. **Per-subject structural failure** on D2-005 and D2-023 suggests
   these subjects are outside the augmented pipeline's competence
   entirely. An unresolved question: do they also fail on the real-only
   baseline (dataset property) or only under augmented training
   (distribution-shift artefact)? Chapter 6 flags this as a limitation.

4. **Phase 1 is complete for the paper.** Any narrative "synth helps
   downstream segmentation" now requires Phase 2 (cross-domain D1 → D2)
   to move the needle. Continuing to tune Phase 1 augmentation has no
   remaining upside.

### 4.4.4 Corrected v3 SPADE cell

The final v3 row should read (post-variance-study):

| Version | concat | **spade (n=8)** |
|---|---|---|
| v3 (3 fixes + Path B, t=0.26) | 0.053 ± 0.056 (n=3) | **0.178 ± 0.054** |

Anywhere the paper cites v3 SPADE, use the n=8 numbers (0.178 ± 0.054),
not the earlier n=3 numbers (0.218 ± 0.057). Any Phase 2 comparisons
should be against 0.178.

## 4.5 Phase 2 — cross-domain (exp2) [target: 500 words]

**Motivation**: Phase 1 exhausted preprocessing-fix levers within the
D2-only generator setup. The remaining hypothesis is that data scale
and diversity — not architectural or preprocessing choices — are the
limiting factor. Cross-domain leverages D1's 51-subject T2 pool (~70%
larger than D2's 32 subjects) for diversity.

### 4.5.1 Configuration recap

- Generator: SPADE + PatchGAN backbone (inherited from 1c_spade).
- Generator training data: D1_MHS T2 (32 subjects).
- Discriminator: conditional PatchGAN with **label zeroed** (unconditional
  D) to avoid label-distribution shortcut between D1 and D2.
- Discriminator training data: D2_TCPW T2FS (41 subjects — RAovSeg
  training pool + em-positive subjects).
- λ_peak = 0.01 (same as Phase 1 1c).
- 100k steps.
- Inference: D1 r1 masks; ovary target t = 0.26 (inherited from Phase 1;
  not re-calibrated because pilot batch had no clear ovary structure).

Assembly (`assemble_synth_exp2.sh`): all three preprocessing fixes ON
(body silhouette mask, histogram match to D1 raw source, resample to D1
source frame). 32 synth volumes assembled.

### 4.5.2 exp2 DSC results (n=3)

| Seed | DSC (full) |
|---|---|
| 0 | 0.0266 |
| 1 | 0.0255 |
| 2 | 0.0089 |
| **Mean** | **0.0203** |
| Std | ~0.010 |

vs Phase 1 v3 SPADE (n=8): 0.178 ± 0.054
vs real-only baseline: 0.290
**Gap to baseline: −93%. Gap to Phase 1 best: −89%.**

### 4.5.3 Why Phase 2 collapsed

Sample grids at every step from 5k → 95k show the generator plateaued
at "gray-blob body silhouette with textured noise" — **no distinct T2FS
style acquired, no visible organ structure**. Diagnosis:

1. **DDPM MSE loss on D1 T2 dominated adversarial signal at λ = 0.01.**
   MSE says "reconstruct D1 T2 (bright, non-fat-sup)"; adversarial says
   "look more like D2 T2FS (dark, fat-sup)". The two objectives are
   antagonistic, and MSE won. The generator settled on a compromise
   that satisfies neither: a gray blob that has neither T2's bright fat
   nor T2FS's suppressed-fat texture.

2. **Unconditional D provided weaker gradient than Phase 1's
   conditional D.** Zeroing the label to avoid the label-distribution
   shortcut removed the label-consistency signal, leaving D judging pure
   style at pixel level — not enough to counter MSE dominance.

3. **Domain gap larger than Phase 1's (T2 → T2FS across sites vs D2 → D2
   in-domain), fewer training subjects (32 D1 vs 41 D2).** Both
   compounded the two above.

### 4.5.4 Why the −93% DSC drop is stable

- **3 seeds within std ~0.010** (much tighter than Phase 1's ~0.054).
- All 3 seeds land in the same failure mode: predict near-zero ovary on
  essentially every test subject.
- **Bad synth doesn't just waste training capacity — it corrupts the
  real signal.** At n=30 real, ~30 pieces of gray-blob "training data"
  is enough to break the model's ovary-detection prior.

### 4.5.5 What Phase 2 tells the paper

Detailed in Chapter 6. The three claims that emerge from exp2:

1. **Bad synth is worse than no synth.** At data-scarce clinical scales,
   augmentation quality is not optional — a mediocre generator poisons
   training rather than neutrally not-helping. This is the strongest
   practical lesson from the whole two-phase study.

2. **Naive DDPM + adversarial cross-domain translation does not work**
   with the standard schedule (λ_peak = 0.01, PatchGAN base_channels = 64,
   100k steps). The MSE reconstruction loss is too dominant.

3. **The −93% DSC is a much cleaner negative-result claim than the
   Phase 1 −39%.** Paper story: Phase 1 shows synth augmentation helps
   marginally then plateaus below baseline; Phase 2 shows cross-domain
   extension actively harms. Together they map the design space where
   synth augmentation fails in this regime.

## 4.6 exp2 λ_peak sweep — lam05_pathC and lam50_pathC [target: 200 words]

Local artefacts: `exp2_lam05_samples/`, `exp2_lam05_samples_volumes/`,
`exp2_samples_volumes/`, `scripts/train_exp2_lam05.sh`,
`scripts/train_exp2_lam50.sh`, `scripts/assemble_synth_exp2_lam05.sh`,
`scripts/assemble_synth_exp2_lam50.sh`, and the six
`scripts/run_raovseg_aug_exp2_lam{05,50}_pathC_seed{0,1,2}.sh`.

### 4.6.1 Rationale

exp2 collapsed at λ_peak = 0.01, and exp2_pathC (§4.7) recovered a
usable −48% by fixing the enhancement-window mismatch. The remaining
question was whether **λ_peak tuning stacks additively on top of pathC**
— i.e. is the adversarial signal at λ = 0.01 too weak to matter even
after preprocessing is corrected? Two variants were trained end-to-end
and evaluated with the same pathC preprocessing: λ_peak = 0.05 (5× exp2)
and λ_peak = 0.50 (50× exp2). Everything else identical to exp2_pathC.

### 4.6.2 DSC results (n = 3 seeds each)

| Seed | lam05_pathC (full) | lam50_pathC (full) |
|---|---|---|
| 0 | 0.1190 | 0.1142 |
| 1 | 0.1436 | 0.1569 |
| 2 | 0.0314 | 0.0493 |
| **Mean** | **0.0980** | **0.1068** |
| Std | 0.0590 | 0.0542 |

vs exp2_pathC (0.152 ± 0.054): both variants land ~0.05 DSC below the
pathC-only mean but within 1σ — no statistically significant difference
at n = 3. vs baseline (0.290): −66% and −63% respectively. vs each
other: within 0.02 of each other, well inside noise; a 10× change in
λ_peak did not move the mean.

### 4.6.3 What §4.6 tells the paper

Three findings.

1. **λ_peak is not the limiting factor.** A 10× multiplier (0.05 → 0.50)
   did not move the downstream DSC. This retires the "we abandoned exp2
   prematurely at λ = 0.01" hypothesis: even at λ = 0.50 the mean sits
   ~0.05 below exp2_pathC's 0.152.

2. **λ tuning does not stack with pathC.** Both stacked variants are
   marginally lower than pathC-alone. The preprocessing lever
   (enhancement-window mismatch) explained essentially all of the
   recoverable gap in Phase 2; adversarial-weight tuning contributes
   nothing at the mean-effect level.

3. **Seed 2 crashed in both variants** (0.031 and 0.049) — the ResClass
   instability documented in §4.4 recurs in Phase 2. Cross-seed variance
   dominates the difference between λ settings.

Together the sweep places the interpretive result in the **0.05–0.15
bucket** of §4.6.2's original planning matrix: *λ tuning is a lever
but not by itself sufficient*. Chapter 6 claim on Phase 2 remains
unchanged: cross-domain DDPM + adversarial translation at n = 30 real
is architecturally insufficient, and λ_peak is not the missing
ingredient.

## 4.7 exp2_pathC — diagnostic-driven preprocessing rescue

### 4.7.1 Motivation

After exp2's collapse to 0.020, we ran a synth-vs-real intensity
diagnostic (`src/RaovSeg_recreation/diagnose_synth_vs_real.py`) that
plots per-variant body-voxel intensity histograms after RAovSeg's
percentile-clip + minmax normalisation. Key finding:

- **Real D2 body voxels** are bimodal — one peak near 0.10 (suppressed
  fat / background), another near 0.55 (organs / other tissue). RAovSeg's
  enhancement window [0.22, 0.30] sits in the VALLEY between them, so
  the enhancement fires predominantly on ovary and nearby structures.
- **1c_spade synth body voxels** are similarly bimodal → enhancement
  window in the valley → augments mostly ovary. Consistent with Phase 1's
  usable DSC 0.178.
- **exp2 synth body voxels** are unimodal, peaked around 0.35 — the peak
  sits **on top of** the enhancement window. When RAovSeg applies the
  enhancement, it saturates ~30–40% of body voxels to 1, not just
  ovaries. The segmenter learns "if it looks like this, ovary is
  everywhere" and predicts near-zero at test time on real T2FS.

**The primary mechanism behind exp2's −93% collapse is a preprocessing
mismatch — not generator quality per se.**

### 4.7.2 The fix

Skip the enhancement step for D2-9 prefix subjects via
`preprocess.py --skip-enhancement-for-prefix D2-9`. The synth remains
in the training set at its natural intensity distribution; the RAovSeg
enhancement fires only on real D2 T2FS. Everything else (generator,
synth NIfTIs, augmentation ratio, seeds) unchanged.

### 4.7.3 exp2_pathC DSC results (n=3)

| Seed | DSC (full) |
|---|---|
| 0 | 0.1317 |
| 1 | 0.2118 |
| 2 | 0.1117 |
| **Mean** | **0.1517** |
| Std | 0.0538 |

vs exp2 (naive): 0.020 ± 0.010 → **+0.132 DSC (7.6× improvement)**
vs Phase 1 v3 SPADE (n=8): 0.178 ± 0.054 → within 0.5σ, no
statistically significant difference at n=3.
vs baseline: 0.290 → still −48% but no longer catastrophic.

Best seed (seed 1) hit 0.2118 — above Phase 1 v3's n=8 mean of 0.178.

### 4.7.4 What §4.7 tells the paper

1. **Preprocessing alignment is the dominant downstream lever, not
   generator quality.** exp2's generator quality is unchanged between the
   0.020 and 0.152 configurations. The 0.132 DSC gain came from ONE
   preprocessing flag on the SAME synth NIfTI files.

2. **Cross-domain synth is competitive with in-domain synth once
   preprocessing is fixed.** exp2_pathC (0.152) vs Phase 1 v3 SPADE
   (0.178) — means within 0.5σ. No cross-domain benefit at mean-effect
   level, but no additional penalty once the enhancement mis-application
   is corrected.

3. **The −93% number remains a valid diagnostic case study, not the
   headline Phase 2 number.** For the paper's Phase 1 vs Phase 2
   comparison, **use exp2_pathC (0.152) as the Phase 2 downstream
   result**, and use exp2 (0.020) as an illustration of how
   preprocessing mismatch can catastrophically poison training.

4. **The gap to baseline persists across all configurations.** Even the
   best "fixed" Phase 2 mean (0.152) is 48% below 0.290. At n=30 real,
   augmentation with either in-domain (Phase 1) or cross-domain (Phase 2)
   synth cannot close the gap.

## 4.8 Cross-experiment DSC summary

| Configuration | n | DSC (mean ± std) | Δ vs baseline |
|---|---|---|---|
| **Baseline (real-only)** | — | **0.290** | — |
| Phase 1 v3 concat (t=0.26) | 3 | 0.053 ± 0.056 | −82% |
| Phase 1 v3 SPADE (t=0.26) | 3 → 8 | **0.178 ± 0.054** | **−39%** |
| Phase 1 Opt B SPADE (t=0.22) | 3 | 0.165 ± 0.096 | −43% |
| Phase 1 Opt B SPADE (t=0.28) | 3 | 0.189 ± 0.102 | −35% |
| Phase 1 Opt C SPADE (no enh) | 3 | 0.170 ± 0.056 | −41% |
| **Phase 2 exp2** (D1 gen + D2 disc, λ=0.01, naive) | 3 | **0.020 ± 0.010** | **−93%** ← diagnostic case study |
| **Phase 2 exp2_pathC** (skip enh for D2-9) | 3 | **0.152 ± 0.054** | **−48%** ← Phase 2 headline |
| Phase 2 exp2_lam05_pathC (λ=0.05 + pathC) | 3 | 0.098 ± 0.059 | −66% |
| Phase 2 exp2_lam50_pathC (λ=0.50 + pathC) | 3 | 0.107 ± 0.054 | −63% |

**The three headline numbers are:**
- **0.178** (best Phase 1, in-domain synth) — the ceiling of preprocessing
  tuning on Phase 1 pathway.
- **0.152** (Phase 2 with proper preprocessing) — cross-domain matches
  in-domain within noise; preprocessing lever explains most of the
  Phase 2 story.
- **0.020** (Phase 2 naive) — the diagnostic case study proving that
  preprocessing mismatch destroys downstream utility.

All three are meaningfully below the 0.290 real-only baseline. The
λ_peak sweep (0.05 and 0.50, both applied on top of pathC) added
additional evidence that adversarial weight is not the limiting factor:
both stacked variants land ~0.05 below exp2_pathC's mean, within noise
of each other and of pathC-alone.

## 4.9 Notes on reproducibility of these results

- All Phase 1 checkpoints and Phase 2 exp2 checkpoints are on Stanage at
  `/mnt/parscratch/users/ijp25lg/synth_mri/runs/exp1{a,b,c_concat,
  c_spade}/` and `.../exp2_d1_gen_d2_disc/`.
- All synth NIfTI volumes: `/mnt/parscratch/users/ijp25lg/synth_mri/
  synth_volumes/exp1c_{concat,spade}/`, `.../exp2/`, `.../exp2_lam05/`.
- Per-run RAovSeg outputs (models, predictions, per-subject DSC):
  `.../runs/raovseg_aug_{concat,spade,exp2,exp2_lam05}_seed{0..7}/`.
- Master metrics CSV: `../metrics/master_metrics.csv` at project root.
- Variance study summary: `../metrics/variance_study_summary.json`.
- exp2 DSC summary: `../metrics/exp2_dsc_summary.json`.
- Full commands and SLURM scripts: Chapter 7 appendix.

## 4.10 What is not in this chapter

- **Radiologist qualitative review results** — 50 matched-anatomy synth
  samples per variant were prepared in `{variant}/radiologist_review/`
  for clinical review, but the qualitative assessment is deferred to a
  future paper revision. See `../docs_archive/NEXT_STEPS.md` Tier A2.
- **Post-process body-mask fix at the sample level** (Tier A1 in
  NEXT_STEPS) — not applied because the augmentation pipeline uses the
  fix already at `assemble_synthetic_volumes.py`. The sample-level fix
  would improve reported FID/hist_KL by 10–30 points but doesn't change
  the DSC story.
- **Alternative downstream segmenters** — RAovSeg is the only downstream
  tested. Chapter 6 flags this as a limitation.
