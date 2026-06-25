# Tier 1 Inference Tuning + Explainability — Session Notes

A record of the inference-side tuning sweep on Exp 1b (CFG guidance + DDIM
steps), the resulting default change applied to both 1a and 1b for ablation
parity, and the design + implementation of the explainability module
(`src/Generator/explain.py`). Companion to `EXP1A_NOTES.md` and
`EXP1B_NOTES.md`.

---

## 1. Why we ran Tier 1

The body-centered v2 runs (both 1a and 1b at 80k steps) produced visually
acceptable bodies and label-aligned organ positions, but the SPADE-conditioned
1b samples had **noticeably noisier / grainier interior textures** than the
concat-conditioned 1a samples. The 1a interior looked anatomically clean
(visible muscle/fat striation, smooth organ regions). The 1b interior carried
a persistent high-frequency noise floor.

The instinct was to attribute this to the conditioning mechanism — pure SPADE
forces the encoder to learn label-blind features and then relies on the
decoder to specialise them. That's a harder optimisation problem at this data
scale (32 subjects, ~730 slices).

Before reaching for an architectural fix (PatchGAN in Exp 1c), we wanted to
rule out **two cheaper hypotheses**:

1. The graininess is a **CFG-amplification artefact** — high guidance scales
   amplify the difference between conditional and null predictions, and that
   difference can carry high-frequency noise that gets amplified along with
   the conditioning signal.
2. The graininess is a **sampler-noise artefact** — 50 DDIM steps may simply
   be too coarse for SPADE's noise predictions to converge cleanly to the
   data manifold.

Both hypotheses are testable without any retraining: just sweep the inference
knobs on the existing EMA checkpoint.

---

## 2. The sweep — what we ran

Inference-only, no retraining, no parity concerns (because the checkpoint is
already trained — we're only varying how it's sampled).

7 grids per variant, on the high-anatomy labels that `inference_validate.py`
picks (top-N by foreground voxel count):

| Combo | Guidance | DDIM steps | Purpose |
|---|---|---|---|
| baseline | 3.0 | 50 | Training default — what we already had |
| `g1.5_s50` | 1.5 | 50 | Low guidance — does it kill the grain? |
| `g2.0_s50` | 2.0 | 50 | Mild guidance |
| `g5.0_s50` | 5.0 | 50 | Strong guidance — does grain get worse? |
| `g7.5_s50` | 7.5 | 50 | SD-default — over-amplified regime |
| `g3.0_s100` | 3.0 | 100 | More sampling steps at default guidance |
| `g3.0_s250` | 3.0 | 250 | Even more steps |
| `g2.0_s100` | 2.0 | 100 | Cross-cell — lower guidance + more steps |

Total runtime: ~15 min on one A100 (each grid is 4 samples × N DDIM steps).

### Issues encountered (sweep)

- **`inference_validate.py` did not accept a `--num-inference-steps` flag.**
  Only `--guidance-scale` was overridable from the CLI; step count was read
  from the YAML. Fixed by adding `--num-inference-steps` (3 lines, mirrors
  the `--guidance-scale` pattern).
- **No quantitative metrics yet** — read was purely visual on the 7 grids
  side-by-side. This is the standing limitation acknowledged in
  `EXP1B_NOTES.md` §10: FID / boundary DSC / LPIPS-NN are Phase-1 endpoints
  planned for after 1c.

---

## 3. Results from the sweep

Visual observations across the 7 grids, all using identical labels (seed-stable
top-4 by voxel count):

| Combo | Texture quality | Organ conditioning visible | Verdict |
|---|---|---|---|
| g1.5 / s50 | Very smooth | Weak — looks like a generic pelvis | Under-guided |
| g2.0 / s50 | Smooth | Mild | Cleaner than baseline, weaker conditioning |
| g3.0 / s50 (baseline) | Grainy | Moderate | The known starting point |
| g3.0 / s100 | Grainier than s50 | Moderate | DDIM path matters — non-monotonic |
| g3.0 / s250 | Smooth | Moderate | More steps does help when pushed far |
| g5.0 / s50 | Grainy + amplified | Strong | Over-sharpens, visible artefacts |
| g7.5 / s50 | "Burned" / over-saturated | Very strong | SD-default too aggressive here |
| **g2.0 / s100** | **Smoothest** | Mild–moderate | **Best texture/conditioning balance** |

### Two takeaways

1. **The graininess is largely a CFG-amplification artefact, not architecture.**
   Dropping guidance to 2.0 cleans the texture. This is partial good news for
   1b — pure SPADE isn't structurally broken; it was just being asked to
   amplify too hard.
2. **There's a tradeoff that inference-only knobs can't dodge.** Lower
   guidance cleans textures but the organ regions become less differentiated
   from the body background. The synthetic at g=2.0 looks like a smooth
   generic pelvis with hints of organ structure; at g=5.0+ the organs pop
   but the whole image looks artefacted. g=2.0/s=100 sits at the best
   balance we found.

### What this implies for the architecture story

1a (concat) gave cleaner textures than 1b at the **same** g=3.0/s=50 settings.
That means concat is more CFG-tolerant: it can sustain stronger guidance
without grain. That's a genuine property of the conditioning mechanism, not
hyperparameter luck.

This still points to **1c (PatchGAN) as the right architectural answer**:
an adversarial texture loss would let SPADE run at strong guidance (g=3.0+)
for sharp conditioning while pulling textures back to realism. PatchGAN does
the job that lowering guidance does for us now, but without sacrificing
conditioning strength.

---

## 4. Adopted defaults change

Applied to both `exp1a.yaml` and `exp1b.yaml`:

| Field | Was | Now |
|---|---|---|
| `sampling.num_inference_steps` | 50 | **100** |
| `sampling.guidance_scale` | 3.0 | **2.0** |

### Why both and not just 1b

The user is explicit that the 1a vs 1b ablation is the spine of this work.
If the inference defaults differ between variants, the comparison is no
longer apples-to-apples — any visual or quantitative difference becomes
confounded by the inference recipe.

The Tier 1 sweep was only run on 1b. The 1a checkpoint has not yet been
swept. So adopting g=2.0/s=100 for both is a **best-known-shared-default**
rather than a per-variant optimum. If a future 1a sweep finds a better local
optimum (e.g. 1a prefers g=3.0/s=100), revisit and update 1b to match. The
YAML comments capture this explicitly.

### Issues encountered (config change)

- **Cheatsheet §12 sweep block annotations were stale** after the change
  ("at the training-default 50 DDIM steps" was no longer accurate). Updated
  the comments and made the loops pass `--num-inference-steps 50` and
  `--guidance-scale 3.0` explicitly so the sweep grid spans the same range
  regardless of what the YAML default is.

---

## 5. Explainability — why these methods

The user asked for GradSHAP, LIME, and any DDPM-specific methods, with the
goal of producing **visual heatmaps to understand what the model is focusing
on**. The XAI literature offers many tools; we filtered to what's actually
informative for label-conditioned diffusion.

### What we ruled out

- **LIME** — viable but expensive. Each LIME heatmap needs ~1000
  perturbations of the input, each requiring a full sampling run (~10 s at
  100 DDIM steps). That's hours per heatmap. LIME is also model-agnostic,
  which is *less* informative for diffusion than gradient methods because it
  discards all temporal / denoising-step structure. Not worth it given the
  alternatives.
- **Robustness / perturbation-stability analysis** — academic interest only,
  not load-bearing for the downstream augmentation goal.
- **Global / dataset-level aggregation** — premature. Needs a research
  question first, plus many more samples.

### What we built

Five methods, ordered by interpretability strength relative to implementation
cost:

1. **Deepest-layer activation map** (proxy for "attention") — model-specific,
   nearly free.
2. **GradientSHAP attribution per label channel** — apples-to-apples between
   1a and 1b.
3. **SPADE γ magnitude maps** (1b only) — the native SPADE interpretation;
   1a has no analogue and the asymmetry IS the point.
4. **Counterfactual label ablation** — zero each organ channel, regenerate,
   diff. Answers "is the model actually using all four conditioning channels."
5. **Per-timestep snapshots** — show the denoising trajectory from pure noise
   to clean image. Free given the sampling chain is already being run.

### Coverage assessment

| Category | Covered? | By method |
|---|---|---|
| Local feature attribution | ✅ | (2) GradientSHAP |
| Internal attention / activation | ✅ | (1) Activation map |
| Module-level mechanistic | ✅ | (3) SPADE γ |
| Counterfactual reasoning | ✅ | (4) Ablation |
| Temporal / per-timestep | ✅ | (5) Snapshots |
| Global / dataset-level | ❌ | Not in scope |
| Boundary fidelity | ❌ | Future: Exp 5 boundary DSC |
| Memorisation | ❌ | Future: Exp 5 NN-LPIPS |
| Robustness | ❌ | Out of scope |

The five methods cover the "where, what-if, and when" axes that are most
informative for DDPM interpretability. The remaining gaps (boundary fidelity,
memorisation) are quantitative endpoints planned for Exp 5, not visual
heatmaps.

---

## 6. Implementation — the five methods in detail

All five live in `src/Generator/explain.py` plus a CLI mirroring
`inference_validate.py`. No Captum dependency — GradientSHAP is implemented
manually (~12 lines) so `requirements.txt` stays unchanged.

### (1) Deepest-layer activation map

**Implementation:** register a forward hook on the deepest module in the
U-Net (1b: `mid_block_2`; 1a: `middle_block` via MONAI's API, with fallback
to the last attention-like module by name). During sampling, capture each
forward pass's output, take `|activation|.mean(dim=channels)`, average across
all timestep forward passes (2× per DDIM step with CFG), and upsample
bilinearly to 512×512.

**Why this proxy, not literal attention probabilities:** capturing attention
probabilities directly requires either modifying `SelfAttention2D.forward()`
to expose its softmax output, or re-running the attention computation inside
the hook (~2× compute). The activation magnitude is a strictly cheaper signal
that captures "where the model is carrying information at the abstract
bottleneck level" — which is the interpretive question we cared about.

**Issue:** for 1a we needed a heuristic to find the right module because
MONAI's `DiffusionModelUNet` doesn't expose a stable `mid_block_2`. The
fallback chain is: `unet.middle_block` → last attention-named module → error.

### (2) GradientSHAP attribution per label channel

**Implementation:** manual GradientSHAP (path-integral form):

```python
for _ in range(n_samples):
    alpha = U(0, 1)
    interp = baseline + alpha * (input - baseline)
    grad = ∂target / ∂interp
    attribution += grad * (input - baseline) / n_samples
```

- **Target:** mean over squared predicted noise at a single mid-range
  timestep (t=500). Single timestep keeps it tractable; t=500 is roughly the
  most informative point (neither pure noise nor near-clean).
- **Baseline:** zero label tensor. Semantically meaningful for a CFG-trained
  model because zero is the null condition the model already knows.
- **Output:** `(B, C, H, W)` absolute attribution per channel.

**Why manual rather than Captum:** the implementation is small enough that
adding a 50-MB dependency to `requirements.txt` (which lives on an HPC env
that's locked down) isn't worth it. The manual version also makes it explicit
what's happening — useful for the methodology write-up later.

**Issue:** gradient flow had to be carefully preserved. The model is loaded
in `eval()` mode and EMA weights are used, but neither of those disables
autograd. The trap is `model.sample()` which is decorated `@torch.no_grad()`.
We use `model.predict_noise()` directly here, which is **not** decorated, so
gradients flow.

### (3) SPADE γ magnitude maps (1b only)

**Implementation:** hook every `SPADE` module in 1b's UNet, capture `|γ|`
averaged across feature channels and timesteps. The hook re-computes γ by
running the captured input through `module.mlp_shared` → `module.mlp_gamma`
(no_grad), because the SPADE forward returns the modulated output, not γ
itself.

**Why re-compute rather than modify SPADE.forward to expose γ:** zero source
changes outside `explain.py` keeps the explainability work neutral with
respect to model behaviour. If we modified SPADE to cache γ during training,
that's a memory cost paid by every training run for an interpretability
feature only used at inference.

**Issue:** SPADE modules live at multiple decoder depths (different spatial
resolutions: 64², 128², 256², 512²). Each γ map has a different shape, so
we upsample each to 512×512 before averaging across timesteps. The figure
shows n_cols evenly-spaced modules by name (deepest → shallowest) — not
all 12+ SPADE modules; that would be illegible.

### (4) Counterfactual label ablation

**Implementation:** sample once with the full label and once per ablated
organ channel (uterus, L-ov, R-ov, em). All 5 runs share **the same initial
noise vector** so per-pixel diffs are meaningful (otherwise stochasticity
swamps the conditioning effect).

The figure also includes a `(−uterus) − full` diff panel in red/blue (bwr
colormap) so you can see exactly where removing the uterus changed the
generated image.

**Why same noise:** without it, the diff between full and ablated samples
is dominated by sampling stochasticity rather than label-channel effect.
The shared-noise version isolates the conditioning influence as the only
variable.

**Cost:** 5 sampling runs per sample × 4 samples = 20 runs, the most
expensive part of the figure. `--skip-counterfactual` available to drop
this when iterating.

**Issue:** had to re-implement the DDIM sampling loop inline
(`_sample_with_noise`) because `model.sample()` generates its own random
noise. Couldn't just call `model.sample()` 5 times with controlled noise.

### (5) Per-timestep snapshots

**Implementation:** during a single sampling run, capture `x_t` at
n_snapshots evenly-spaced step counts (including before any step = pure noise
and after the last step = clean image). Snapshots are stored on CPU as they
happen to avoid GPU memory growth.

**Cost:** free — piggybacks on a sampling run that would happen anyway.

**Why useful:** visualises the coarse-to-fine generation that's structural
to diffusion models. The body silhouette emerges first (~80% noise),
then organ positions (~50% noise), then texture (~10% noise → final). Tells
the "what is built when" story that gradient methods can't.

---

## 7. Figure layout

Per sample, one PNG. 4 rows for 1a, 5 rows for 1b (the SPADE γ row is
1b-only — that asymmetry is itself a signal). 6 columns throughout.

```
Row 1 (always):     real | label | synth | overlay | attn | (blank)
Row 2 (always):     GradSHAP[outside] | [uterus] | [ov_L] | [ov_R] | [em] | [body_other]
Row 3 (always):     counterfactual: full | -ut | -Lov | -Rov | -em | diff(-ut − full)
Row 4 (always):     snapshot @ step 0 | 20 | 40 | 60 | 80 | 100 (with noise % annotation)
Row 5 (1b only):    |γ| at 6 evenly-spaced SPADE modules (deepest → shallowest)
```

For 1a vs 1b comparison: line up `sample_00.png` from each side by side.
Rows 1–4 are directly comparable. Row 5 exists for 1b only.

### Issues encountered (figure)

- **Color-bar scaling on heatmaps:** GradientSHAP attributions and γ
  magnitudes have very different scales per channel/module. Solved by using
  per-panel 99th-percentile vmax (clip outlier pixels). Without this, a few
  high-magnitude pixels blow out the colormap and the rest of the heatmap
  looks black.
- **Counterfactual diff is signed** (positive or negative change). Used
  blue-white-red colormap with symmetric vmin/vmax instead of viridis so
  "no change" reads as white.

---

## 8. Cost summary

Per sample (4 samples is the default `--n`):

| Step | Sampling runs | Wall-clock @ 100 DDIM steps |
|---|---|---|
| (1) attn + main synth | 1 | ~10 s |
| (2) GradientSHAP | 0 (single-t forwards) | ~2 s |
| (3) SPADE γ | 1 | ~10 s |
| (4) counterfactual | 5 | ~50 s |
| (5) per-timestep | 1 | ~10 s |
| **Per sample** | **8** | **~80 s** |
| **For `--n 4`** | **32** | **~6 min** |

Both variants together: ~12 min on an A100, plus a few seconds for figure
assembly. Fits inside a 1 h interactive `srun` session with margin.

---

## 9. What this answers — and what's left

### Answered

- **Is 1b's graininess fixable without retraining?** Partially. g=2.0/s=100
  gives clean textures but trades conditioning strength.
- **Is the 1a vs 1b texture gap structural to the conditioning mechanism?**
  Yes — concat is more CFG-tolerant than pure SPADE at equal compute.
- **Is the model using all label channels?** The counterfactual row will
  tell us, per sample.

### Still open

- **Quantitative comparison.** All judgments here are visual. FID, boundary
  DSC, LPIPS-NN — these are the Exp 5 endpoints. Until they're computed,
  "g=2.0/s=100 is better than g=3.0/s=50" is a visual claim, not a measured
  one.
- **Does the 1a sweep find the same optimum?** We adopted g=2.0/s=100 for
  both for parity but only swept 1b. Symmetric sweep on 1a is the
  immediate next inference experiment.
- **What does the explainability comparison actually show?** The figures
  haven't been generated yet. The hypotheses to test from the figure
  outputs:
  1. SPADE γ shows spatially localised activation (bright at organ
     regions, dim elsewhere) — if not, SPADE isn't using its label-locality
     advantage even when given the chance.
  2. GradientSHAP on the uterus channel highlights pixels inside or near
     the uterus mask in both 1a and 1b — if not, the conditioning is
     degenerate.
  3. The counterfactual diff for `-uterus` shows the strongest change in
     the uterus region (not, say, in the bowel or bladder area) — if not,
     the model is mixing conditioning channels.
  4. The per-timestep snapshots show coarse-to-fine emergence — if instead
     they show the body silhouette appearing late in the chain, the model
     isn't using the high-noise timesteps for layout the way it should.

### Architectural implications going forward

The Tier 1 result reframes the 1c hypothesis. The original framing was
"PatchGAN improves texture realism." The Tier 1 result sharpens this to:
**"PatchGAN should let SPADE keep sharp conditioning (high CFG) while
avoiding the texture cost that high CFG carries for pure SPADE."** That's
a more specific and testable claim — and one that the explainability
metrics introduced here will be useful for adjudicating.

---

## 10. Explainability findings + the per-variant guidance decision

> **⚠ Important correction (added later)**: Finding 2 and Finding 3 below were
> based on visual interpretation of explainability figures. They were
> **partially wrong** — see Section 11 for the corrected analysis from the
> quantitative 2×2 results after Exp 1c. In particular: SPADE *does* achieve
> per-organ localisation (5–10× higher CLR than concat), and SPADE γ heads
> *do* correlate with organ regions (OSI organ_corr ≈ 0.25). The visual
> bwr-diff panels misled us because the colormap normalises per-image:
> SPADE's locally-concentrated change looked "empty" relative to concat's
> globally-distributed change.
>
> The original §10 text is preserved below as a record of what we believed
> at the time, so the reasoning trail is intact. Finding 1 (the per-variant
> guidance decision) and the YAML changes from it are still correct.

After implementing `explain.py` and running it on both 1a and 1b at the
shared g=2.0/s=100 default, plus a control run of 1a at g=3.0/s=100 to test
the under-conditioning hypothesis, three findings emerged.

### Finding 1 — 1a was under-conditioned at the shared g=2.0 default

Direct visual comparison of the synthetic panels (TEST 1 column 3) for
sample 0:

| Variant | Guidance | Synthetic appearance |
|---|---|---|
| 1a (concat) | g=2.0 | Very dark, mostly featureless, organs barely visible |
| **1a (concat)** | **g=3.0** | **Detailed pelvic anatomy, bright organ regions, internal contrast** |
| 1b (SPADE) | g=2.0 | Detailed pelvic anatomy, organ regions visible |

The Tier 1 sweep found g=2.0 best for 1b — but applying it to 1a as a shared
default was a parity mistake. 1a at g=3.0 catches up to 1b at g=2.0 visually:
both produce comparably detailed images. Each variant has a different
local optimum.

**Decision adopted:** revert 1a's `guidance_scale` to 3.0 while keeping 1b
at 2.0. DDIM steps remain shared at 100. This is a deliberate asymmetry —
documented in both `exp1a.yaml` and `exp1b.yaml` — chosen because the
downstream RAovSeg augmentation goal cares about synthetic quality, not
about settings purity. The paper comparison should report results at each
variant's optimum (the "what these architectures can do" framing) and can
optionally include the shared-setting comparison as a strict ablation.

### Finding 2 — Counterfactual sensitivity scales with guidance for concat, not for SPADE

Looking at TEST 3's diff panel (red/blue, signed) across the three runs:

- **1a at g=2.0** — visible but mild blue/red signal in the uterus region.
  Concat conditioning is working at weak strength.
- **1a at g=3.0** — **strong, localised signal in the uterus region**, plus
  secondary effects in surrounding tissue. Concat is genuinely
  channel-specific at proper guidance.
- **1b at g=2.0** — nearly empty diff. Removing the uterus channel barely
  changes the output.

This sharpens the earlier finding that "concat is more channel-sensitive than
SPADE." The actual pattern is:

> **Concat's per-channel sensitivity scales with guidance. SPADE's stays
> bounded regardless of guidance.**

At higher guidance, 1a uses each label channel surgically. SPADE at any
guidance produces a smoother, more joint conditioning effect across channels.
This is an architectural property of the two mechanisms, not a tuning issue.

### Finding 3 — SPADE γ confirms: no per-organ localisation is being learned

The TEST 5 row in 1b's explainability figure (all 6 decoder modules):
all show a near-identical body-vs-outside modulation pattern. Bright orange
across the body region, light at the body silhouette boundary, dim/white
outside the body. No module shows organ-specific modulation (e.g. brighter
inside the uterus mask than in the surrounding bowel area).

Combined with Finding 2, this paints a consistent picture: **SPADE has
learned to use its modulation capacity to encode "inside body vs outside" —
information already given to the model explicitly through the `body_other`
and `outside_body` channels**. The per-organ spatial precision that should
have been SPADE's structural advantage over concat is not being exploited
at this data scale.

### What this means for the original SPADE hypothesis

The synthetic_mri_generator_design.md doc framed SPADE as the way to get
"per-pixel boundary correspondence" and "spatially adaptive denormalisation"
that concat structurally cannot. The empirical result is the opposite:

| Expected (from design doc) | Observed |
|---|---|
| SPADE more localised than concat | SPADE *less* per-channel responsive |
| SPADE encodes per-organ patterns | SPADE encodes body-shape pattern |
| SPADE's γ heads specialise per organ | All γ heads converge to same body modulation |
| Concat is "blunt" channel-by-channel | Concat is surgically channel-specific at proper guidance |

This is a meaningful, paper-worthy negative result for the SPADE-on-small-
medical-data hypothesis as we've implemented it. Two possible explanations
(not mutually exclusive):

1. **Data scale.** 32 training subjects / ~730 slices is too few for SPADE's
   per-pixel γ/β heads to learn organ-specific modulation. Pure SPADE
   originally targeted multi-thousand-image semantic segmentation datasets.
2. **Architectural capacity.** `spade_hidden=64` may be too narrow for the
   γ/β heads to learn organ-specific patterns even with more data.

### Implications for 1c

The earlier framing of 1c was "PatchGAN improves texture realism." With
these findings, that claim becomes ambiguous:

- For **1a (concat)**, texture realism is already fine at g=3.0 — adding
  PatchGAN to 1a tests whether adversarial loss tightens the
  conditional-distribution match, not whether it cleans grain.
- For **1b (SPADE)**, the texture realism issue at g=3.0 was real but is
  now solved by inference-only tuning (g=2.0). PatchGAN on 1b would
  pressure the SPADE+adversarial combo to do **both** sharper conditioning
  AND realistic texture. Whether SPADE's underused localisation capacity
  unlocks under PatchGAN pressure is a real open question.

In other words, **1c is now testing a different hypothesis than originally
scoped**: not "does PatchGAN clean grain" but "does adversarial pressure
force SPADE to learn the per-organ modulation it currently doesn't." If the
answer is no, 1c becomes evidence that SPADE-at-this-data-scale is
fundamentally limited and the next step is either more data (Phase 2 cross-
domain) or a different conditioning mechanism (cross-attention, AdaIN, etc).

### Decisions adopted from this section

1. **Per-variant guidance defaults** — 1a at g=3.0/s=100, 1b at g=2.0/s=100,
   DDIM steps shared. YAML comments updated; this doc is the rationale.
2. **Paper comparison at each variant's optimum** is the primary framing.
   Shared-setting comparison (both at g=2.0 or both at g=3.0) can be
   included as a strict-ablation supplement.
3. **1c hypothesis rescoped** — see above. The original
   `synthetic_mri_generator_design.md` framing of SPADE's advantages
   should be updated to acknowledge the negative findings before paper
   write-up.

---

## 11. CORRECTION — what the quantitative 2×2 actually showed

After implementing the Category 1 (CLR/AILM/OSI) and Category 2 (FID/LPIPS)
metrics and running them on all four variants (1a, 1b, 1c_concat, 1c_spade),
the picture flipped on two of §10's findings.

### What §10 got right
- **Finding 1 (1a under-conditioned at g=2.0)**: confirmed. The per-variant
  guidance defaults (1a→g=3.0, 1b→g=2.0) are correct.
- The architectural framing that concat and SPADE optimise different
  objectives: still correct.

### What §10 got wrong

**§10 Finding 2 — "Concat's per-channel sensitivity scales with guidance,
SPADE's stays bounded"** — *partially wrong*. The visual diff panel was
misleading. Quantitative CLR shows:

| Variant | CLR_uterus | CLR_L-ov | CLR_em |
|---|---|---|---|
| 1a (concat g=3.0) | 0.013 | 0.043 | 0.028 |
| 1b (SPADE g=2.0) | **0.407** | **0.494** | **0.532** |
| 1c_concat (concat+GAN g=3.0) | 0.069 | 0.080 | 0.063 |
| 1c_spade (SPADE+GAN g=2.0) | 0.405 | 0.297 | 0.420 |

**SPADE achieves 10–30× higher per-channel localisation than concat**, not
the other way around. Removing the uterus label changes mostly the uterus
region in SPADE variants (40–53% of the per-pixel change is in the uterus
mask). For concat variants, only 1–8% of the change is in the uterus region;
the rest is spread globally. **Concat is the globally-mixing one, not SPADE.**

The visual bwr-diff panels deceived us because matplotlib auto-normalises
the colormap per-image. Concat's globally-distributed change has a wide
absolute value range so the diff "looked rich." SPADE's locally-concentrated
change has a narrow absolute range so the diff "looked empty." Same
magnitude pattern flipped visually.

**§10 Finding 3 — "SPADE γ encodes body-shape, not organ-specific patterns"**
— *wrong*. Quantitative OSI (Organ Specificity Index, Pearson correlation
between |γ| and organ vs body masks):

| Variant | OSI max_organ_corr | OSI body_corr |
|---|---|---|
| 1b | 0.242 | −0.011 |
| 1c_spade | 0.258 | 0.029 |

SPADE γ heads show meaningful positive correlation with organ regions
(~0.25) and near-zero correlation with the body mask. **They are picking
up per-organ structure**, not just inside-vs-outside body. The earlier
"all γ maps look the same body-shape" reading was a visual artefact of the
magma colormap making body-interior values dominate the visible range.

### What §10's predictions said vs what actually happened

> §10 predicted: PatchGAN should "let SPADE keep sharp conditioning (high
> CFG) while avoiding the texture cost that high CFG carries for pure SPADE."

What actually happened in 1c:
- **PatchGAN on concat (1c_concat)** got the biggest realism wins (FID 188
  → 166, hist_KL 8.15 → 5.79). Localisation barely changed (still globally
  mixed). PatchGAN's expected behaviour, on the wrong architecture.
- **PatchGAN on SPADE (1c_spade)** got modest realism wins on LPIPS
  (best of all 4 at 0.699) but lost a small amount of CLR. Adversarial
  pressure traded a little localisation for a little realism.
- **PatchGAN does different things to different architectures**, not a
  uniform "best of both worlds."

### Corrected architectural map (current understanding)

```
                ← MORE LOCALIZED              MORE GLOBAL →

  best        ┌──────────────────────────────────────────────────┐
  texture     │                                                   │
  realism     │   1c_spade ★★              1c_concat ★★★          │
              │   LPIPS 0.70               FID 166, hist_KL 5.79  │
              │   CLR_ut 0.41              CLR_ut 0.07            │
              │                                                   │
              │       1b                        1a                │
              │   CLR_ut 0.41              CLR_ut 0.01            │
              │   FID 200                  FID 188                │
  decent      │                                                   │
  realism     └──────────────────────────────────────────────────┘
```

**Full analysis** in [RESULTS_2x2.md](../../RESULTS_2x2.md) at project root.
**Per-experiment summaries** in
[EXP1B_SUMMARY.md](../../EXP1B_SUMMARY.md) and
[EXP1C_SUMMARY.md](../../EXP1C_SUMMARY.md).

### Lesson learned for the methodology

Visual inspection of explainability figures is a sanity check, not an
adjudicator. Per-image colormap normalisation can flip the qualitative
reading. **Quantitative metrics computed across many samples are
necessary** to draw architectural conclusions. The Category 1 metrics
(CLR/OSI) we added to `explain.py` after §10 were designed for exactly
this — they should be the primary signal, with the figures used to make
the metrics interpretable to a reader.

### Additional finding from explainability — outside-body hallucinations

After the 2×2 results came in, a separate diagnostic (the
"outside-body hallucinations" investigation) found that the explainability
activation map highlights bright structured content **outside the body
silhouette** in some samples. Root cause is preprocessing-side
(body silhouette computed from image intensity threshold; coil noise +
morphological closing imperfections leak some bright pixels into the
"body_other" channel during training). Fix is post-process masking at
inference (~30 lines) or a tighter body-silhouette computation
(re-preprocessing + re-training).

**This is documented separately** in the project's NEXT_STEPS.md plan.
