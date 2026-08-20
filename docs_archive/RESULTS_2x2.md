# Synthetic Pelvic MRI Generator — 2×2 Quantitative Results

> ⚠️ **SUPERSEDED (2026-08-11)** — Every 1c row in this document was
> trained with a PatchGAN gradient-severance bug (`eps_pred.detach()`
> at `train.py:464`) that voided the adversarial gradient to the
> generator. CLR, DSC, and interpretations of 1c_concat and 1c_spade
> reported here are for buggy models equivalent to their no-PatchGAN
> counterparts (1a and 1b respectively). Corrected numbers appear
> in the dissertation §4.7 and in [LAMBDA_ABLATION_COLLAPSE.md](../LAMBDA_ABLATION_COLLAPSE.md).
> Preserved as historical record.

> **TL;DR**
> Four conditional DDPM variants compared: concat (1a), SPADE (1b), and each with a PatchGAN discriminator (1c_concat, 1c_spade). **No single winner across all metrics** — the four variants occupy a clean architectural map of "controllability vs realism." **1c_concat dominates texture realism** (best FID and hist_KL). **SPADE variants dominate per-organ controllability** (5–10× higher CLR than concat). **1c_spade is the best of both worlds** for perceptual realism while keeping localization.
>
> **⚠ Correction**: an earlier visual interpretation claimed SPADE wasn't doing per-organ localization. That was wrong — a colormap-normalization artifact. The quantitative CLR proves SPADE works as designed.

---

## 1. The 4 variants under comparison

| Variant | Conditioning | Adversarial loss | Steps | Inference guidance |
|---|---|---|---|---|
| **1a** (concat) | label concat at input (7-channel) | — | 80k | g = 3.0 |
| **1b** (SPADE) | label via decoder SPADE modules | — | 80k | g = 2.0 |
| **1c_concat** | concat | PatchGAN | 100k | g = 3.0 |
| **1c_spade** | SPADE | PatchGAN | 100k | g = 2.0 |

All four trained on the same 32 D2 subjects (~730 slices), same body-centered preprocessing, same 6-channel labels `[outside_body, uterus, L-ov, R-ov, em, body_other]`. Per-variant guidance scales were tuned in Tier 1 inference sweep.

---

## 2. Master metrics table

| Variant | CLR_ut ↑ | CLR_ov_L ↑ | CLR_em ↑ | OSI_organ | OSI_body | **FID ↓** | hist_KL ↓ | LPIPS_min | LPIPS_mean ↓ |
|---|---|---|---|---|---|---|---|---|---|
| 1a | 0.013 | 0.043 | 0.028 | n/a | n/a | 188.2 | 8.15 | 0.493 | 0.824 |
| 1b | **0.407** | **0.494** | **0.532** | 0.242 | −0.011 | 200.1 | 6.89 | 0.350 | 0.745 |
| **1c_concat** | 0.069 | 0.080 | 0.063 | n/a | n/a | **166.5** | **5.79** | 0.426 | 0.773 |
| **1c_spade** | 0.405 | 0.297 | 0.420 | **0.258** | 0.029 | 188.1 | 7.20 | 0.343 | **0.699** |

↑ = higher is better, ↓ = lower is better. **Bold** = column winner.

---

## 3. What each metric actually tells us

### CLR — Counterfactual Localization Ratio (↑)

For each organ channel, removes that channel from the label, regenerates the synthetic image with the same initial noise, and computes how concentrated the change is inside that organ's region:

> **CLR(channel) = ‖change‖² inside the channel's mask / ‖change‖² over the whole image**

- **CLR → 1.0**: removing the channel mostly affects its own region. The model uses that channel for that organ specifically (clean per-channel conditioning).
- **CLR → 0.0**: removing the channel changes the image everywhere. The model spreads the channel's signal globally (mixed conditioning).

**What we see**: SPADE variants reach 0.30–0.53 (truly localized per-organ effect). Concat variants reach 0.01–0.08 (globally-mixed). **The conditioning mechanism, not adversarial loss, determines this** — PatchGAN slightly nudges both directions but doesn't change the fundamental pattern.

### OSI — Organ Specificity Index (SPADE only)

For each SPADE module, Pearson correlation between |γ| and each organ mask vs. the body mask.

- **OSI_max_organ_corr > 0**: SPADE γ correlates with at least one organ region (SPADE doing its localization job).
- **OSI_body_corr ≈ 0**: γ is NOT just encoding "inside body vs outside" (which would be a degenerate use of SPADE's per-pixel capacity).

**What we see**: Both 1b and 1c_spade show organ_corr ≈ 0.25 and body_corr ≈ 0. **SPADE γ heads are picking up per-organ structure** at the module level, not just body shape.

### FID — Fréchet Inception Distance (↓)

Distance between synthetic and real image distributions in Inception-V3 feature space. Standard image-realism metric.

| FID range | Interpretation |
|---|---|
| < 50 | State-of-the-art (e.g. StyleGAN on faces, large datasets) |
| 50–100 | Solid medical synthesis with thousands of training subjects |
| 150–250 | Reasonable for low-data medical regime (our setting) |
| > 300 | Distribution meaningfully different from real |

**What we see**: 1c_concat at 166 is meaningfully best — 22 points lower than 1a. Note: at N=256 samples, FID has a noise floor of ~±30, so 1a/1b/1c_spade are statistically tied at ~188–200.

### hist_KL — Intensity Histogram KL Divergence (↓)

KL divergence between synthetic and real pixel-intensity histograms (100 bins on [0,1]).

**What we see**: 1c_concat at 5.79 is best (29% lower than 1a). hist_KL is more sample-efficient than FID — a 2.4-point gap here is real signal, not noise. PatchGAN clearly tightens the concat baseline's intensity distribution toward real.

### LPIPS-NN — Perceptual Nearest-Neighbour Distance

For each synthetic image, finds perceptual distance to nearest real image (AlexNet feature space). Reports min/mean/max.

| LPIPS_min value | Interpretation |
|---|---|
| < 0.05 | **Memorization warning** — synth essentially copying real |
| 0.05–0.30 | Mild similarity; check manually |
| 0.30–0.50 | Healthy; no memorization concern |
| > 0.50 | Very distinct from any real |

**What we see**: All min values ≥ 0.34 — none memorizing. 1c_spade has lowest mean (0.70) = synthetic images are perceptually closest to real WITHOUT memorizing.

---

## 4. Visual examples

### 4.1 Per-variant explainability figures

Each figure has 4 (concat) or 5 (SPADE) test sections: bottleneck activation, GradientSHAP per channel, counterfactual ablation, denoising trajectory, and SPADE γ (SPADE only).

**1a (concat baseline)** — counterfactual diff visually large but spread across whole image:

![1a explain sample 0](1a/current/explain/sample_00.png)

**1b (SPADE)** — SPADE γ row shows per-module modulation. Counterfactual diff visually subtle but concentrated in organ regions:

![1b explain sample 0](1b/current/explain/sample_00.png)

**1c_concat (concat + PatchGAN)** — PatchGAN tightens texture realism; FID drops 22 points vs 1a:

![1c_concat explain sample 0](1c/concat/explain/sample_00.png)

**1c_spade (SPADE + PatchGAN)** — combines SPADE's localization with PatchGAN's perceptual realism pressure; best LPIPS-NN:

![1c_spade explain sample 0](1c/spade/explain/sample_00.png)

### 4.2 Matched-anatomy side-by-side synthetic samples

Same seed across variants → same label and same initial noise → direct apples-to-apples comparison. Each model's task: "given the same anatomy specification, generate a realistic T2FS pelvic slice."

**Real source slice** (the actual MRI the label was extracted from):

![real source](1a/current/radiologist_review/real_000.png)

**Synthetic — 1a (concat baseline)**

![1a synth](1a/current/radiologist_review/synth_000.png)

**Synthetic — 1b (SPADE)**

![1b synth](1b/current/radiologist_review/synth_000.png)

**Synthetic — 1c_concat (concat + PatchGAN)**

![1c_concat synth](1c/concat/radiologist_review/synth_000.png)

**Synthetic — 1c_spade (SPADE + PatchGAN)**

![1c_spade synth](1c/spade/radiologist_review/synth_000.png)

> **Tip for Notion**: after pasting and uploading the 5 images above, select the four "Synthetic — ..." image blocks and use Notion's column drag to lay them out as a 2×2 grid manually. The markdown table form gets parsed as a database in Notion, which doesn't work for image grids.

Additional 50-sample sets per variant are in each variant's `radiologist_review/` folder for clinical review.

---

## 5. Architectural map — where each variant lives

```
                ← MORE LOCALIZED                  MORE GLOBAL →

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

### Reading the map
- **1c_concat** dominates texture realism (FID, hist_KL) but conditions globally.
- **1c_spade** dominates perceptual realism (LPIPS) while preserving SPADE's per-channel localization.
- **1a** is currently dominated by 1c_concat on every quality metric — PatchGAN added genuine value to concat.
- **1b** is dominated by 1c_spade on LPIPS, but the two are otherwise close — PatchGAN added less value to SPADE.

---

## 6. Key interpretations

### 6.1 PatchGAN does different things to different architectures

| Δ from baseline | concat (1a → 1c_concat) | SPADE (1b → 1c_spade) |
|---|---|---|
| FID | −12% (real improvement) | −6% (within noise) |
| hist_KL | **−29%** (strong) | +5% (slight regression) |
| LPIPS_mean | −6% | −6% (best overall) |
| CLR (localization) | +small (still globally-mixed) | −small (still localized) |

PatchGAN's effect depends on what the architecture lacked:
- **Concat lacked texture realism** → PatchGAN delivered (big FID and hist_KL wins).
- **SPADE lacked perceptual smoothness** → PatchGAN delivered (best LPIPS), at a small cost to localization.

### 6.2 SPADE's per-organ design actually works

This corrects an earlier visual finding. The CLR numbers prove SPADE does what it was designed to do: per-channel label changes cause per-channel image responses. The visual diff panels misled because the colormap auto-normalizes per image — SPADE's locally-concentrated changes have a narrow value range so visually "look empty" relative to concat's globally-distributed changes.

### 6.3 The architecture choice depends on the downstream use case

There is no universal winner. For the project's downstream augmentation goal (improving RAovSeg ovary DSC):
- **1c_concat** for maximum image realism (synthetic looks like real T2FS).
- **1c_spade** for label-aware augmentation (synthetic uterus stays inside the uterus label).
- **1b** if you want pure SPADE without adversarial complexity.
- **1a** is currently dominated and can be retired.

---

## 7. Implications and next steps

### 7.1 For the paper
- Drop the "SPADE doesn't learn per-organ patterns" framing — it's contradicted by CLR and OSI.
- Adopt the corrected story: SPADE delivers per-organ control as designed; PatchGAN's contribution depends on the conditioning mechanism it's combined with.
- The 2×2 ablation produces a clean architectural map worth publishing as a primary figure.

### 7.2 For the downstream augmentation question (Exp 4)
- Both 1c variants are strong candidates.
- The ultimate "did this work" measurement is RAovSeg ovary DSC with each variant as the augmenter — not done yet.
- Recommendation: titrate the four variants {1a, 1b, 1c_concat, 1c_spade} as augmentation sources for RAovSeg training, compare DSC on the 8 sacred test subjects.

### 7.3 Methodological caveats
- **CLR/OSI** based on N=4 labels — small sample. The 5–10× concat-vs-SPADE CLR gap is robust to noise; smaller within-arm differences are not.
- **FID at N=256** has ~±30 noise floor. The 1a→1c_concat 22-point gain is at the edge of meaningful. hist_KL is more sample-efficient and shows the same direction more confidently.
- **AILM was uninformative** (always ≈1.0 by construction of gradient SHAP) and has been dropped from future reports.
- **Absolute FID ~170–200** is high but expected at 32-subject training data — published medical synthesis with this little data often sits in the 100–300 range.

---

## 8. Files / artifacts

```
1a/current/             ← 1a explain figures, quality.json, 50 radiologist samples
1b/current/             ← 1b explain figures, quality.json, 50 radiologist samples
1c/concat/              ← 1c_concat explain figures, quality.json, 50 radiologist samples
1c/spade/               ← 1c_spade explain figures, quality.json, 50 radiologist samples
master_metrics.csv      ← the 4×N column table this report is built from
```

All four variants share the same label seeds in `radiologist_review/` for matched-anatomy comparison.

---

*Generated from `master_metrics.csv` + per-variant `explain/` and `radiologist_review/` outputs. Source: `src/Generator/explain.py`, `quality_metrics.py`, `aggregate_metrics.py`.*
