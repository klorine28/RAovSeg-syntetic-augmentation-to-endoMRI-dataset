# 5 — Discussion

> **Target: 1,300 words.** Four headline claims (§5.1–5.4),
> meta-lessons (§5.5), and interpretive limitations (§5.6). Project-level
> limitations and future work move to Chapter 6.

---

## 5.1 Claim 1 — bad synth is worse than no synth [target: 250 words]

Phase 2's exp2 downstream DSC of **0.020 ± 0.010** (n = 3) —  93%
below the real-only baseline of 0.290 — demonstrates that a mediocre
generator does not merely fail to help; it **actively corrupts the
downstream training signal**. This is the sharpest empirical lesson
from the two-phase study.

Three properties strengthen the claim:
- Standard deviation across seeds is ~0.010 (much tighter than Phase
  1's ~0.054), so the failure is stable, not variance-driven.
- All three seeds land in the same failure mode: predict near-zero
  ovary on essentially every test subject.
- The mechanism is understood (Chapter 4, §4.5.3): the DDPM MSE loss
  on D1 T2 dominated the adversarial signal from the unconditional
  D2-trained discriminator at λ = 0.01. The generator plateaued at a
  "gray blob" that satisfies neither reconstruction nor style-transfer
  objectives.

The implication for the field is direct: at n < 50 real subjects,
augmentation quality is not optional. A generator whose outputs do not
faithfully match the target distribution can degrade a working
baseline by 90%. This warning applies to any medical vision task at
similar data scale using generative augmentation. It also refines the
common assumption that "more data is always better" — mediocre
synthetic data violates that assumption in a specific and quantifiable
way, at least in the ovary segmentation regime this work occupies.

## 5.2 Claim 2 — concat conditioning is architecturally locked out [target: 275 words]

Concat-conditioned generators (Exp 1a, 1c_concat) achieve
Counterfactual Localisation Ratio (CLR) values of 0.013–0.069 across
all target channels. In interpretive terms: removing the uterus label
channel and regenerating with the same initial noise changes the
image nearly uniformly across the whole frame, rather than
preferentially inside the uterus region. The label is used globally,
not per-organ.

The downstream consequences are severe and consistent across every
preprocessing intervention:

| Fix level | Concat DSC | Δ vs baseline (0.290) |
|---|---|---|
| v1 (no fixes) | 0.150 ± 0.006 | −48% |
| v2 (framing + hist match + body silhouette) | 0.044 ± 0.039 | −85% |
| v3 (v2 + Path B label-aware rescale) | 0.053 ± 0.056 | −82% |

The mechanism is clear from Chapter 4's diagnostic (§4.3.4).
Label-aware preprocessing fixes (v2 histogram matching, v3 Path B
ovary rescale) rely on the generator having localised, ovary-textured
content in the correct spatial location. Concat's CLR ≈ 0.03 means it
does not. Rank-based histogram matching then places bright pixels
wherever the generator happened to make them bright — random
locations — and the segmenter trains to predict ovary at random
locations. Label-aware Path B rescale forces bright pixels *inside*
the ovary mask, but disconnected from surrounding synth tissue, so
the segmenter cannot learn from an intensity-forced blob with no
textured context.

Concat is not a rescuable augmentation source at Phase 1 preprocessing
sophistication. This is an architectural limitation, not a
preprocessing failure. For downstream label-aware tasks (segmentation,
detection), per-organ localisation at the generator matters more than
raw image realism.

## 5.3 Claim 3 — SPADE conditioning approaches but does not close the gap [target: 275 words]

SPADE-conditioned generators achieve CLR values of 0.30–0.53 across
target channels — genuine per-organ localisation. This gives the
preprocessing fixes something to align with, and the downstream
trajectory reflects that:

| Fix level | SPADE DSC | Note |
|---|---|---|
| v1 (no fixes) | 0.138 ± 0.049 | Baseline augmentation attempt |
| v2 (3 preprocessing fixes) | 0.169 ± 0.037 | +22% vs v1 |
| v3 (v2 + Path B t = 0.26, n = 3) | 0.218 ± 0.057 | +58% vs v1 |
| **v3 revised at n = 8** | **0.178 ± 0.054** | −38% vs baseline 0.290 |

Every preprocessing fix moved SPADE upward, up to a ceiling. The
sequence tells us three things:

- Preprocessing pipeline alignment is necessary (v2 → v3 mattered).
- It is not sufficient (0.178 remains 38% below baseline).
- The gap (0.11) is 2× the cross-seed standard deviation (0.054) —
  the underperformance is statistically robust.

The n = 8 variance study corrects an earlier optimistic reading. At
n = 3, v3 SPADE appeared to be approaching baseline (0.218 → −25%
gap). Five additional seeds averaged 0.154, dragging the mean to
0.178. The original narrative — "variance is masking a real benefit"
— did not survive the added seeds.

Options B (target intensity sweep) and C (skip enhancement for synth)
both landed below v3, confirming that further preprocessing tuning
does not move the needle. Phase 1 is exhausted at 0.178 for SPADE.

The interpretive implication: even with a generator that achieves
per-organ localisation and careful preprocessing alignment, DDPM
augmentation at n = 30 real subjects does not match the real-only
baseline. Per-organ localisation is necessary but not sufficient at
this data scale.

## 5.4 Claim 4 — preprocessing pipeline alignment matters more than raw synth quality [target: 300 words]

FID and hist_KL do not predict downstream success. The 2×2 Phase 1
quality map (Chapter 4, §4.2.4) shows no single winner across
quality metrics: 1c_concat wins FID (166) and hist_KL (5.79),
1c_spade wins LPIPS (0.699), the two SPADE variants win CLR.
Downstream DSC does not correspond to any of these standard metrics —
1c_concat has the worst downstream DSC (0.053 at v3), 1c_spade has
the best (0.178 at v3, n = 8).

What does predict downstream utility:

1. **Field-of-view match** — synth NIfTI must be saved at the source
   real subject's spacing/origin/direction so downstream preprocessing
   produces synth with the same body framing (~60% of frame) as real.
2. **Body silhouette cleanup** — outside-body hallucinations survive
   downstream percentile-clip + minmax and get amplified into
   structured noise; kill them at generation time.
3. **Intensity distribution match** — rank-based histogram matching
   aligns the post-clip distribution but does not by itself guarantee
   the ovary lands in the enhancement window.
4. **Label-aware ovary intensity targeting** (Path B) — the single
   most impactful fix for SPADE (0.169 → 0.218 at n = 3).

The meta-lesson is that synthetic generators for downstream
augmentation must be designed with awareness of the downstream
consumer's preprocessing assumptions. RAovSeg's ovary enhancement
rule at [0.22, 0.30] is a hidden pipeline assumption that turns out
to matter more than raw image realism. If we had reported only FID
and moved directly to downstream evaluation, we would have picked
1c_concat (best FID) and produced −85% DSC. The CLR + per-organ
localisation reasoning pointed us to SPADE; the downstream pipeline
analysis pointed us to the enhancement window.

## 5.5 Meta-lessons for the field [target: 100 words]

Three cross-cutting lessons:

- **n = 3 seeds are insufficient for downstream augmentation claims at
  n < 50 real.** The v3 SPADE mean went from 0.218 (n = 3) to 0.178
  (n = 8) — a 22% drop. Per-subject variance dominates cross-seed
  variance by ~4×, so aggregate means without per-subject reporting
  hide universal-failure patterns like D2-005 and D2-023.
- **FID does not predict downstream utility** for label-aware tasks.
  Task-relevant metrics like CLR are essential.
- **Cross-domain DDPM + adversarial translation at n < 50 is
  architecturally insufficient**, at least under the standard
  λ schedule.

## 5.6 Limitations of interpretation [target: 100 words]

Three limitations bear on interpretation of the results above:

- **Small sacred test set (n = 8)** produces wide DSC confidence
  intervals. Per-subject standard deviation across the 8 subjects
  (~0.24) exceeds cross-seed standard deviation (~0.054), so
  differences smaller than ~0.05 are within noise.
- **D2-005 and D2-023 as universal failures.** Whether they also fail
  under the real-only baseline (dataset property) or only under
  augmented training (distribution-shift artefact) is unresolved.
- **Single downstream architecture (RAovSeg).** Whether the
  conclusions transfer to nnU-Net or TotalSegmentator is unknown.

Project-level limitations and future work are addressed in Chapter 6.
