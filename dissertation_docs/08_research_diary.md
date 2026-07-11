# Research Diary and Reflection

> **Word count target: 1,000** (separate budget from the main
> dissertation per the IJC403/404 assessment brief). Structured to hit
> the rubric's four distinction-level requirements: all elements
> completed, substantial evidence of engagement with feedback, strong
> reflection on how changes improved the dissertation, and shown
> learning.

**Student**: Lorenzo Garduno Roqueni (250200038)
**Programme**: MSc Data Science, University of Sheffield IJC
**Supervisor**: Dr Neda Azarmehr
**Project start**: 1 April 2026 &nbsp; **Submission**: 26 August 2026
**Dissertation title**: Synthetic pelvic MRI data augmentation for
downstream segmentation tasks

---

## 1. Project origin and timeline

The project topic was student-initiated: I knew from the outset that I
wanted to combine medical imaging and diffusion models. Preliminary
discussions with Dr Azarmehr before the project formally started led
us to the UT-EndoMRI dataset (Liang et al., 2025) as the working
substrate. After exploring the raw data myself, the concrete framing
— *synthetic augmentation for ovary segmentation* — emerged from
noticing that Liang et al.'s baseline DSC of 0.290 was the direct
product of a 30-subject training pool. That framing became the
dissertation proposal, and the project began on 1 April 2026.

## 2. Supervision meetings and feedback log

Meetings were monthly in person, with email correspondence between. I
kept the notion of "feedback → action → outcome" explicit throughout.

**Early April 2026 — first official supervision (project kickoff).**
I arrived having attempted to reproduce the RAovSeg baseline and
failed to match its published DSC. We discussed the missing
pre-trained weights on the upstream repository and I agreed to move
forward with the recreation as a verified anchor rather than an exact
reproduction, then plan Exp 1a (concat-conditioned DDPM). **Feedback
impact**: adopting the "recreation-as-anchor" framing turned a blocker
into a clean baseline for later comparisons.

**Late May 2026 — ideas-exchange / mid-project scope discussion.**
With Exp 1a partly in flight, we revisited overall scope and the
mid-flight quality-of-life additions (CFG, EMA, and the 6-channel
label with the added `body_other` channel to fix the noisy grey edges
in the early sample grids). **Feedback impact**: agreeing to
retroactively inherit these changes across all Phase 1 variants
locked in the ablation-parity principle that governs Chapter 3 —
without that alignment the 2×2 comparison would have been silently
corrupted.

**June 2026 — Exp 1a and 1b review.** I presented visual samples from
both variants. Dr Azarmehr's response was positive, with two specific
pieces of feedback: (i) the samples still showed visible graininess
worth addressing at inference time, and (ii) I should move beyond
"visual eyeball checks" and introduce quantitative metrics.
**Feedback impact**: the metrics feedback prompted the addition of
LPIPS-NN, FID, hist_KL and a GradientSHAP-based interpretability
pass — the same metrics that appear in Chapter 4's master table.

**End of June 2026 — Exp 1c (both concat and SPADE) review.** I
presented the full 2×2 quality table. Dr Azarmehr agreed the metrics
told a coherent story, and pushed me to *validate the story
downstream* by supplementing RAovSeg's training data with the synth
and measuring DSC. She also mentioned that a radiologist review of
the samples would strengthen the work but was probably outside
project scope. **Feedback impact**: this triggered the entire Phase 1
downstream investigation (Chapter 4 §4.3), which turned out to be the
richest part of the dissertation. The radiologist idea I noted for
Chapter 6's future-work section.

**July 2026 — upcoming.** Planned agenda: the negative Phase 1
downstream results, the n = 8 variance study revising the SPADE mean
from 0.218 to 0.178, and the Phase 2 (cross-domain) collapse. My
expectation is that the discussion will be constructive despite the
negative direction; the analytical depth and methodological rigour
are what matter most.

## 3. How the project evolved from the original brief

The proposal outlined a 7-week schedule with a 3-configuration
ablation (concat, SPADE, SPADE + PatchGAN) plus a Med-DDPM baseline
comparison. Actual project shape after ~4 months of work:

- **Ablation expanded to 4 configurations** (added 1c_concat) so the
  PatchGAN contribution could be isolated on both backbones. This
  made the ablation a proper 2×2 grid and produced the cleanest
  finding of the whole project — PatchGAN's *asymmetric* effect on
  concat vs SPADE.
- **Med-DDPM comparison dropped.** The compute budget and the depth of
  the downstream investigation made a fair 3D comparison
  impractical. Reframed as a limitation and future-work item.
- **Variance study added.** The n = 3 → n = 8 seed extension was not
  planned; it emerged from the wide confidence intervals at n = 3
  and turned out to substantively revise the paper's headline SPADE
  number.
- **Novel CLR and OSI metrics** were not in the brief. They emerged
  from needing an interpretability metric that FID and hist_KL do
  not provide.
- **Phase 2 collapsed but produced the sharpest claim**. The original
  brief treated Phase 2 as a diversity-uplift experiment; in reality
  it became the "bad synth is worse than no synth" evidence that
  anchors the discussion.

Each of these deviations strengthened the dissertation rather than
weakened it, but they came from either supervisor feedback (metrics,
downstream) or from doing the work and letting the data lead
(variance study, CLR, Phase 2 reframing).

## 4. Reflection on learning

**Technical skills built**: medical image preprocessing (SimpleITK,
NIfTI I/O, body-centered vs image-centered framing), DDPM
implementation and training (MONAI Generative, SPADE, EMA, CFG,
DDIM), joint DDPM + adversarial training with spectral norm and
λ scheduling, statistical analysis (variance study, per-subject
distributions, cross-seed vs per-subject variance), SLURM/HPC
workflow, and scientific figure production. Beyond mechanics, my
understanding of *why* diffusion models are stable at low n, and *why*
per-organ localisation matters for segmentation-augmentation
utility, is much deeper than at the start.

**Soft skills built**: long-project planning, scope management,
critical reading of medical papers, problem diagnosis under
uncertainty (the ovary intensity enhancement discovery was a
multi-hour diagnostic session), and communicating negative results
constructively.

## 5. What I would do differently

**Run n = 5 seeds from the start.** The original n = 3 gave me an
optimistically biased SPADE mean of 0.218 that I initially believed;
n = 8 revealed the true mean was 0.178. If I had committed to
n = 5 seeds from Exp 1c onward, I would have caught the variance
problem months earlier.

**Diagnose synth-vs-real side-by-side after the first assembly.**
The body-centered vs image-centered framing mismatch was visible
immediately in a side-by-side plot, but I discovered it only weeks
into the v1 downstream investigation. A single diagnostic figure
after the first assembly would have collapsed the v1 → v2 iteration
loop.

**Involve a radiologist earlier if possible.** Even a single
15-minute review of ten synth samples would have given me confidence
that the images pass a clinical plausibility check. Without that, I
have been relying on quantitative metrics and my own judgement.

**Address the ovary generation problem directly.** The dissertation
identifies the discriminative ovary appearance as the pipeline
bottleneck. If I were starting again I would try localised
ovary-specific conditioning (e.g. per-organ latents, ovary-focused
patch discriminator, or paired T2/T2FS translation if per-subject
pairs exist) as the primary intervention rather than treating it as
downstream future work.

## 6. What I am proud of

The project planning: I designed the ablation so each variant
isolated a specific architectural axis, and set up the fair-
comparison protocol *before* running experiments. The negative
result stands up to scrutiny precisely because of that upfront
rigour. I am also proud of the analytical depth — the CLR metric,
the variance study, and the preprocessing-alignment thread came from
sustained methodical work rather than isolated experiments.
