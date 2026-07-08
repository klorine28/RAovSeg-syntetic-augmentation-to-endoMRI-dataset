# Next Steps — what to do after the 1c results

> Snapshot: all four Phase-1 variants (1a, 1b, 1c_concat, 1c_spade) trained,
> quantitative metrics computed, master CSV produced, radiologist review sets
> generated. Outside-body hallucination diagnostic complete (root cause:
> preprocessing-side body silhouette imperfection, not architecture).
>
> See [RESULTS_2x2.md](RESULTS_2x2.md) for the full quantitative comparison.
> See [EXP1B_SUMMARY.md](EXP1B_SUMMARY.md) and [EXP1C_SUMMARY.md](EXP1C_SUMMARY.md)
> for per-experiment context.

---

## Current state — what's done

| | |
|---|---|
| Phase 1 generators | 4 variants trained (1a/1b @ 80k, 1c_concat/1c_spade @ 100k) |
| Per-variant quantitative metrics | FID, hist_KL, LPIPS-NN in each `quality.json` |
| Per-variant interpretability metrics | CLR (counterfactual localisation), OSI (SPADE γ organ correlation), sparsity in `explain/sample_NN_metrics.json` |
| Per-variant explainability figures | 4 multi-panel figures per variant (TEST 1-5) in `explain/` |
| Per-variant radiologist review sets | 50 matched-anatomy samples per variant in `radiologist_review/` |
| Master comparison CSV | `master_metrics.csv` at project root |
| Architectural understanding | Concat=global realism, SPADE=per-organ localisation, PatchGAN adds differentially |
| Per-variant guidance optima | 1a/1c_concat → g=3.0; 1b/1c_spade → g=2.0; DDIM steps=100 shared |

## Open issues — what's not done

| Issue | Status | Impact |
|---|---|---|
| **Outside-body hallucinations** | Diagnosed (preprocessing root cause); not yet fixed | Affects radiologist credibility + quality metrics; minor impact on RAovSeg DSC |
| **Exp 4 — RAovSeg downstream titration** | **DONE (Phase 1 exhausted)** | v3 SPADE @ n=8: **DSC 0.178 ± 0.054** vs baseline 0.290. See RAOVSEG_AUGMENTATION_EXPERIMENT.md §8g. |
| **N=4 sample limit on Cat-1 metrics** | Known limitation | CLR/OSI absolute values noisy; relative ordering robust |
| **N=256 FID noise floor (~±30)** | Known limitation | Cross-variant FID differences <30 points are within noise |
| **Phase 2 cross-domain (D1→D2)** | **DONE (catastrophic collapse)** | exp2 DSC = 0.020 ± 0.010 (n=3), −93% vs baseline. Reinforces the negative-result story: bad synth is worse than no synth. See RAOVSEG_AUGMENTATION_EXPERIMENT.md §8h. |
| **exp2_lam05 (Track 2 tuning)** | In progress | Diagnostic, not rescue. Unlikely to reverse the −93% gap. |
| **Paper draft** | Underway | Outline in PAPER_OUTLINE.md; §7 numbers landed; focus shifts to writing. |

---

## Prioritised next steps

### TIER A — Quick wins (~1 day total)

#### A1. Post-process body-mask the synthetic outputs

**Why**: outside-body hallucinations affect FID, hist_KL, and radiologist credibility. The fix is ~30 lines: after `generate_samples.py` produces a synthetic image, multiply by the body silhouette mask (or just set `outside_body` pixels to -1).

**What changes**:
- Add `--mask-outside-body` flag to `generate_samples.py` and `quality_metrics.py`
- When set: load the label tensor, force `synth[label[0]==1] = -1.0` before saving / before metric computation
- Default: ON (cleaner outputs everywhere)

**Cost**: ~30 lines of code + re-run quality_metrics + re-run generate_samples for the radiologist review = 1 GPU hour total.

**Expected impact**:
- FID likely drops 10-30 points across all 4 variants (much closer to publication-quality range)
- hist_KL drops meaningfully
- Radiologist samples no longer show anatomically nonsensical content outside body

#### A2. Re-render the radiologist review samples with the mask applied

After A1 lands, re-run `generate_samples.py` for all 4 variants. The 50-sample sets become clinically presentable.

#### A3. Update master_metrics.csv with masked numbers

After A1 lands, re-run `quality_metrics.py` and `aggregate_metrics.py`. The published table becomes a "fair" comparison (none of the variants get unfairly penalised by an artefact of the preprocessing).

### TIER B — The actual downstream test (3-5 days)

#### B1. Exp 4 — RAovSeg titration with synthetic augmentation

**The single most important next experiment.** Train RAovSeg on (real + synthetic-from-variant-X), evaluate on the 8 sacred D2 test subjects, report ovary DSC. Do this for all 4 variants (or the top 2: 1c_concat + 1c_spade).

**What's needed**:
- Generate larger synthetic volumes (e.g. 30-60 synth NIfTI volumes per variant)
- Wire synthetic volumes into RAovSeg's preprocessing pipeline
- Train RAovSeg with mixed real+synth, multiple seeds (3+) per variant
- Evaluate DSC on the 8 sacred test subjects
- Compare against RAovSeg-real-only baseline (DSC 0.290)

**The decision tree this produces**:
- If 1c_concat or 1c_spade gets DSC > 0.30: paper has a positive result, write it up
- If both are ≤ 0.290: synthetic augmentation didn't help at this data scale; consider Phase 2 (more diverse training data via D1)
- If 1c_spade > 1c_concat: SPADE's per-organ localisation matters for downstream
- If 1c_concat > 1c_spade: texture realism matters more for downstream

**Cost**: ~3 days of HPC time (synth generation + 3-seed RAovSeg training per variant). The RAovSeg replication is already done so the training pipeline exists.

### TIER C — Parallel with Tier B: paper drafting

#### C1. Start paper outline + figures

**Why parallel**: B1 takes days; the paper structure doesn't depend on B1's outcome (just the headline numbers).

**Outline structure** (suggested):
1. Introduction: low-data medical synthesis problem, RAovSeg baseline, SPADE vs concat conditioning question
2. Methods: 2D conditional DDPM with two conditioning variants + adversarial discriminator
3. Architectural ablation (1a/1b): localisation vs realism tradeoff
4. PatchGAN's differential effect (1c_concat vs 1c_spade)
5. Downstream segmentation (Exp 4): augmentation impact on RAovSeg DSC
6. Discussion: when each variant is best; data-scale limitations
7. Limitations: high FID at this data scale; outside-body artefacts (and fix)

**Figures from existing data**:
- Architectural map (Section 5 of RESULTS_2x2.md)
- Master metrics table
- Per-variant explainability composite (use the 4 figures from `*/explain/sample_00.png`)
- Matched-anatomy synth grid (use radiologist_review samples)
- Outside-body diagnostic before/after (after Tier A1 lands)

**Cost**: ongoing, doesn't block other work.

### TIER D — Conditional on Exp 4 outcome

#### D1. (IF Exp 4 shows DSC < baseline) — Phase 2 cross-domain
- Take the best Phase-1 architecture (probably 1c_concat or 1c_spade)
- Retrain with D1 T2w as generator data, D2 T2FS as discriminator anchor
- ~15-18h HPC + post-training analysis
- Tests the hypothesis that data scale (32→51 subjects with cross-site diversity) is the limiting factor

#### D2. (IF Exp 4 shows positive DSC + paper revision asks for it) — Fix body silhouette and retrain
- Improve `_body_silhouette` in `preprocess_for_generator.py`:
  - Threshold 0.05 → 0.10
  - Add `binary_erosion` step to be conservative about "inside body"
  - Or use TotalSegmentator / a small annotated body-segmenter
- Re-preprocess all subjects (~30 min)
- Retrain best variant from scratch (~18h)
- Expected: cleaner outside-body without needing post-process masking

#### D3. (Optional polish) — Bump N for FID
- Current N=256 has ~±30 FID noise floor
- For paper-quality FID, want N=1024+ per variant
- ~5x more sampling time = ~25 min per variant on A100
- Mostly cosmetic; the relative ordering already shows what matters

---

## Decision points

### Decision 1 — Before Tier A1
**Do we mask outside-body by default or as an option?**
Default ON is cleaner and means no caller has to remember to set the flag. Default OFF preserves the "raw" output for users who want it. **Recommendation**: default ON; the mask is built from the label tensor that's already in the pipeline.

### Decision 2 — Before Tier B1
**Which variants to titrate in Exp 4?**
- Most informative: all 4 (1a, 1b, 1c_concat, 1c_spade) — get a full picture
- Cheapest: top 2 from RESULTS_2x2 (1c_concat, 1c_spade) — answers "does PatchGAN help downstream"
- **Recommendation**: do 1c_concat and 1c_spade first; add 1a and 1b only if the 1c results are inconclusive

### Decision 3 — After Tier B1 results
**Does the paper need Phase 2?**
- If Exp 4 result is clearly positive (DSC ≥ 0.30): Phase 2 is optional polish; the story is complete
- If Exp 4 result is null or negative: Phase 2 is the natural rescue ("more diverse training data was needed")
- **Recommendation**: defer this decision until B1 numbers are in

---

## Recommended sequencing

```
Today / this week:
  Tier A1 → A2 → A3 (post-process masking, re-run metrics + radiologist samples)
  Tier C1 (start paper outline)

Next 1-2 weeks:
  Tier B1 (Exp 4 — RAovSeg titration)
  Tier C1 continues (drafting parallel with B1)

After B1 results:
  Decision 3 (Phase 2 yes/no)
  Tier D1 or D2 if needed
  Paper finalisation
```

If Tier A is done by end of this week and B1 starts immediately, a paper-ready first draft is realistic in 2-3 weeks.

---

## Files referenced in this plan

- [RESULTS_2x2.md](RESULTS_2x2.md) — 2×2 quantitative comparison
- [EXP1B_SUMMARY.md](EXP1B_SUMMARY.md) — what 1b does, how it differs from 1a
- [EXP1C_SUMMARY.md](EXP1C_SUMMARY.md) — what 1c adds, how it complements 1a and 1b
- [src/Generator/TIER1_TUNING_AND_EXPLAINABILITY.md](src/Generator/TIER1_TUNING_AND_EXPLAINABILITY.md) — inference tuning + explainability design (and §11 correction of an earlier wrong finding)
- [architecture_dataflow_v2.md](architecture_dataflow_v2.md) — original architecture plan + current status (§0)
- [master_metrics.csv](master_metrics.csv) — the 4×N column table all this is built from
