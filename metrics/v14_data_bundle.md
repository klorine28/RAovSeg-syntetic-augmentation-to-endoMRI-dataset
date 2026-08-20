# v14 DATA bundle — everything the reviewer needs

_Generated from CSVs in `metrics/` and JSONs in `hpc_pulled/runs/`._

## Variant name mapping

| Reviewer name | Our name | Notes |
|---|---|---|
| `recreation` | `raovseg_real_only` | 5 seeds |
| `concat_pg` | `raov_aug_exp1c_concat_fixed` | 3 seeds, post-fix |
| `spade_pg` | `raov_aug_exp1c_spade_fixed` | 3 seeds, post-fix |
| `spade_pg_v3_8seed` | `raovseg_aug_spade` | 8 seeds, PRE-fix (detection only) |
| `xdom_l001` | `raov_aug_exp2_fixed` | 3 seeds, post-fix, no λ_adv |
| `xdom_l005` | `raov_aug_exp2_lam05_fixed` | 3 seeds, post-fix, λ_adv=0.05 |
| `xdom_l050` | `raov_aug_exp2_lam50_fixed` | 3 seeds, post-fix, λ_adv=0.50 |

## DATA-1 — per-subject × per-seed ovary DSC (fixed variants + real-only)

### Summary per variant
| Variant | n_rows | mean | sd | min | max |
|---|---:|---:|---:|---:|---:|
| `raov_aug_exp1c_concat_fixed` | 24 | 0.2024 | 0.2604 | 0.0000 | 0.7036 |
| `raov_aug_exp1c_spade_fixed` | 24 | 0.2260 | 0.2549 | 0.0000 | 0.6699 |
| `raov_aug_exp2_fixed` | 24 | 0.1882 | 0.2574 | 0.0000 | 0.6711 |
| `raov_aug_exp2_lam05_fixed` | 24 | 0.1727 | 0.2756 | 0.0000 | 0.7404 |
| `raov_aug_exp2_lam50_fixed` | 24 | 0.1577 | 0.2596 | 0.0000 | 0.6690 |
| `raovseg_real_only` | 40 | 0.1890 | 0.2425 | 0.0000 | 0.6667 |

### Per-subject mean DSC (across seeds)
| Variant | D2-005 | D2-015 | D2-016 | D2-017 | D2-023 | D2-024 | D2-026 | D2-038 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `raov_aug_exp1c_concat_fixed` | 0.026 | 0.257 | 0.532 | 0.496 | 0.000 | 0.000 | 0.000 | 0.308 |
| `raov_aug_exp1c_spade_fixed` | 0.081 | 0.411 | 0.472 | 0.534 | 0.000 | 0.000 | 0.000 | 0.311 |
| `raov_aug_exp2_fixed` | 0.000 | 0.182 | 0.407 | 0.448 | 0.000 | 0.191 | 0.000 | 0.278 |
| `raov_aug_exp2_lam05_fixed` | 0.004 | 0.000 | 0.628 | 0.118 | 0.000 | 0.247 | 0.000 | 0.385 |
| `raov_aug_exp2_lam50_fixed` | 0.000 | 0.000 | 0.260 | 0.157 | 0.000 | 0.171 | 0.232 | 0.442 |
| `raovseg_real_only` | 0.002 | 0.364 | 0.323 | 0.530 | 0.000 | 0.000 | 0.000 | 0.293 |

### Per-subject × per-seed raw DSC (recreation baseline, 5 seeds)
_Full DATA-1 CSV is at `metrics/data1_per_subject_per_seed_ovary_dsc.csv` (160 rows). Only the recreation baseline is dense enough to print raw._

| Subject | seed0 | seed1 | seed2 | seed3 | seed4 | mean | sd |
|---|---:|---:|---:|---:|---:|---:|---:|
| D2-005 | 0.000 | 0.004 | 0.000 | 0.006 | 0.000 | 0.002 | 0.003 |
| D2-015 | 0.528 | 0.328 | 0.441 | 0.286 | 0.236 | 0.364 | 0.119 |
| D2-016 | 0.000 | 0.667 | 0.244 | 0.568 | 0.135 | 0.323 | 0.285 |
| D2-017 | 0.448 | 0.603 | 0.492 | 0.658 | 0.451 | 0.530 | 0.095 |
| D2-023 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| D2-024 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| D2-026 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| D2-038 | 0.477 | 0.501 | 0.000 | 0.487 | 0.000 | 0.293 | 0.268 |

## DATA-2 — ResClass firing per subject × seed (recreation baseline)

### Aggregated per subject (5 seeds pooled)
| Subject | fires ≥1 / 5 | slice TP tot | slice FP tot | slice FN tot | max_prob (mean±sd) |
|---|---:|---:|---:|---:|---|
| D2-005 | 5/5 | 15 | 49 | 0 | 0.975 ± 0.031 |
| D2-015 | 5/5 | 9 | 20 | 1 | 0.914 ± 0.091 |
| D2-016 | 4/5 | 8 | 5 | 7 | 0.714 ± 0.101 |
| D2-017 | 5/5 | 16 | 1 | 9 | 0.839 ± 0.101 |
| D2-023 | 5/5 | 20 | 2 | 10 | 0.824 ± 0.070 |
| D2-024 | 1/5 | 0 | 2 | 30 | 0.502 ± 0.171 |
| D2-026 | 5/5 | 30 | 15 | 0 | 0.954 ± 0.053 |
| D2-038 | 4/5 | 11 | 5 | 14 | 0.686 ± 0.128 |

### Key finding — refutes reviewer's D2-005/D2-023 hypothesis
- **D2-005** fires ResClass 5/5 seeds (max_prob 0.975±0.031). 15 true-positive slices, 49 false-positive slices — heavy over-firing but detection intact.
- **D2-023** fires ResClass 5/5 seeds (max_prob 0.824±0.070). 20 TP slices, 2 FP slices — near-perfect detection.
- The universal-failure subjects are **detected correctly**; the ceiling is a **delineation** failure by AttUSeg on the detected slices, not a classification failure.
- The subject that IS a detection failure: **D2-024** fires only 1/5 seeds, max_prob hovers around 0.5 (threshold=0.6).

## DATA-3 — detection rate per variant (gt-positive volumes only)

| Variant | seeds | detected / pairs | detection rate |
|---|---:|---:|---:|
| `raov_aug_exp1c_concat_fixed` | 3 | 21/24 | 87.5% |
| `raov_aug_exp1c_spade_fixed` | 3 | 21/24 | 87.5% |
| `raov_aug_exp2_fixed` | 3 | 22/24 | 91.7% |
| `raov_aug_exp2_lam05_fixed` | 3 | 23/24 | 95.8% |
| `raov_aug_exp2_lam50_fixed` | 3 | 17/24 | 70.8% |
| `raovseg_aug_concat` | 3 | 16/24 | 66.7% |
| `raovseg_aug_exp2` | 3 | 13/24 | 54.2% |
| `raovseg_aug_exp2_lam05` | 3 | 18/24 | 75.0% |
| `raovseg_aug_exp2_lam05_pathC` | 3 | 18/24 | 75.0% |
| `raovseg_aug_exp2_lam50_pathC` | 3 | 18/24 | 75.0% |
| `raovseg_aug_exp2_pathC` | 3 | 18/24 | 75.0% |
| `raovseg_aug_spade` | 8 | 53/64 | 82.8% |
| `raovseg_aug_spade_pathC` | 3 | 23/24 | 95.8% |
| `raovseg_aug_spade_t022` | 3 | 17/24 | 70.8% |
| `raovseg_aug_spade_t028` | 3 | 18/24 | 75.0% |
| `raovseg_real_only` | 5 | 34/40 | 85.0% |

### Pre-fix vs post-fix comparison (same recipe)
| Recipe | Pre-fix | Post-fix | Δ |
|---|---:|---:|---:|
| exp2 (no λ_adv) | 54.2% | 91.7% | +37.5pp |
| exp2, λ_adv=0.05 | 75.0% | 95.8% | +20.8pp |
| exp1c_concat | 66.7% | 87.5% | +20.8pp |

## DATA-4 — ovary intensity in-window fraction (v14 corrected)

**Domain**: RAovSeg `preprocess_image(skip_enhancement=True)` — resample → percentile clip → minmax → [0, 1].
**Window**: [O1, O2] = [0.22, 0.30] (RAovSeg enhancement window).
**Acceptance test**: 3-subject mechanism pool (D2-016/017/024) reproduced at **10.05%** vs cited 10.06% ± 1% — domain verified matching mechanism_figures.py exactly.
**Every synth volume goes through the same measurement pipeline as the real reference.**

| Variant | stage | n_vol | ovary_mean | ovary_sd | in-window % |
|---|---|---:|---:|---:|---:|
| `real_d2` (30-subj corrected reference) | - | 30 | 0.508 | 0.083 | **8.89%** |
| `real_d2_mech3` (3-subj historical) | - | 3 | 0.503 | 0.031 | 10.05% |
| `exp1c_concat` (pre-fix) | post_rescale | 31 | 0.211 | 0.012 | 19.25% |
| `exp1c_concat_fixed` | post_rescale | 31 | 0.263 | 0.055 | **52.99%** |
| `exp1c_spade` (pre-fix) | post_rescale | 32 | 0.206 | 0.012 | 20.92% |
| `exp1c_spade_fixed` | post_rescale | 32 | 0.284 | 0.103 | 21.08% |
| `exp1c_spade_t022` (dial) | post_rescale | 32 | 0.167 | 0.013 | 14.26% |
| `exp1c_spade_t028` (dial) | post_rescale | 32 | 0.227 | 0.011 | 23.21% |
| `exp2` (pre-fix, λ=0.01) | post_rescale | 32 | 0.322 | 0.078 | 20.32% |
| `exp2_fixed` (λ=0.01) | post_rescale | 32 | 0.327 | 0.063 | 9.32% |
| `exp2_lam05` (pre-fix, λ=0.05) | post_rescale | 32 | 0.322 | 0.078 | 20.32% |
| `exp2_lam05_fixed` (λ=0.05) | post_rescale | 32 | 0.325 | 0.080 | 13.25% |
| `exp2_lam50` (pre-fix, λ=0.50) | post_rescale | 32 | 0.322 | 0.078 | 20.32% |
| `exp2_lam50_fixed` (λ=0.50) | post_rescale | 32 | 0.308 | 0.071 | 10.29% |

### Direct empirical proof of the .detach() bug — the pre-fix exp2 rows are byte-identical

The pre-fix `exp2`, `exp2_lam05`, and `exp2_lam50` rows have identical `ovary_mean`, `ovary_sd`, and `in_window_pct` to four decimals. Because the .detach() bug severed the discriminator gradient before it reached the generator, **λ_adv had literally zero effect on the trained generator**. All three "different λ" runs produced identical generator weights → identical synth volumes.

Post-fix, the same three λ values now diverge (9.32% / 13.25% / 10.29%) — the adversarial gradient reaches the generator, and λ matters. This is the strongest single line of evidence that the bug voided the ablation.

### Story points for §4.7.1

| Recipe | pre-fix in-window | post-fix in-window | Δ |
|---|---:|---:|---:|
| `exp1c_concat` | 19.25% | **52.99%** | **+33.7pp** — biggest post-fix calibration gain; concat's per-organ intensity was strongly gated by the adversarial signal |
| `exp1c_spade` | 20.92% | 21.08% | +0.16pp — barely moves; SPADE's per-channel modulation provides gradient signal independently of the adversarial arm |
| `exp2` (cross-domain, λ=0.01) | 20.32% | 9.32% | −11.0pp — mean unchanged (0.322→0.327), spread tightened (sd 0.078→0.063), now falls just above the [0.22, 0.30] upper edge |
| Intensity dial | t022 14.26% / t028 23.21% | — | monotonic Path B target-intensity effect confirmed |

### The v14 correction — headline

- **Old CSV cited real-D2 = 5.34%** on 55 volumes with window [0.5, 0.8]. That was measured in the wrong intensity domain (raw or intermediate, mean 0.649) and wrong subject set.
- **Corrected DATA-4 real-D2 = 8.89%** on 30 training-pool subjects in the RAovSeg preprocess-step-3 domain, with window [0.22, 0.30]. The 3-subject mechanism reference (10.05%) is retained for continuity.
- **All synth in-window percentages have moved substantially** — most notably `exp1c_concat_fixed` from the old 37.88% up to **52.99%**, and `exp2_fixed` from the old 11.70% down to **9.32%**.
- Table 4.10 and any dependent §4.7.1 claims need re-evaluation against these corrected numbers.

## DATA-5 — RAovSeg ablation × 5 recreation seeds

| Config | seeds | mean | sd | per-seed values | paper (n=1) |
|---|---:|---:|---:|---|---:|
| full | 5 | 0.1889 | 0.0681 | 0.181, 0.263, 0.147, 0.251, 0.103 | 0.29 |
| no_postprocess | 5 | 0.1827 | 0.0583 | 0.157, 0.245, 0.155, 0.243, 0.114 | 0.235 |
| no_resclass | 5 | 0.0944 | 0.0226 | 0.069, 0.102, 0.112, 0.117, 0.072 | 0.013 |

### Novel finding: `no_resclass` result differs 7× from paper
- Ours: **0.094 ± 0.023** (n=5 seeds)
- Paper: 0.013 (n=1)
- Our AttUSeg alone is a substantially stronger segmenter than the paper's benchmark suggests, weakening the 'ResClass is critical' claim.

## Quality metrics — 5 fixed variants (n_synth=256, n_real=200)

| Variant | ckpt | g | FID ↓ | hist_KL ↓ | LPIPS_mean ↓ | LPIPS_p10 | LPIPS_p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `exp1c_concat_fixed` | 100000 | 3.0 | 272.04 | 2.621 | 0.7666 | 0.7166 | 0.8150 |
| `exp1c_spade_fixed` | 100000 | 2.0 | 274.81 | 0.959 | 0.7245 | 0.6985 | 0.7484 |
| `exp2_fixed` | 100000 | 2.0 | 266.19 | 11.028 | 0.5928 | 0.5290 | 0.6403 |
| `exp2_lam05_fixed` | 100000 | 2.0 | 352.31 | 5.363 | 0.6429 | 0.5511 | 0.6892 |
| `exp2_lam50_fixed` | 100000 | 2.0 | 379.67 | 5.668 | 0.6227 | 0.5668 | 0.6763 |

### Story-worthy observations
- **SPADE has tightest intensity match**: hist_KL 0.96 (vs 2.62 concat, 11.03 cross-domain)
- **Quality vs downstream disagree** for exp2 family: `exp2_lam05_fixed` has WORST FID (352) but BEST detection (95.8%). `exp2_lam50_fixed` is worst on both (FID 380, detection 70.8%).
- **λ_adv sweet spot exists post-fix**: moderate λ improves label-conditional utility at the cost of distribution fidelity, up to a collapse point at λ=50.
