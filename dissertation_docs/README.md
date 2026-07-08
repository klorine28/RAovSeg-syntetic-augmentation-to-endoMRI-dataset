# Dissertation master docs

Consolidated, dissertation-ready master files. Each doc corresponds to a
dissertation chapter and coalesces the relevant material from the wider
project MD collection (which remains intact as source of truth).

## Reading order

| Doc | Chapter | Contents |
|---|---|---|
| [01_introduction.md](01_introduction.md) | 1. Introduction | Clinical motivation, data scarcity, thesis, contributions, roadmap |
| [02_background.md](02_background.md) | 2. Background | DDPMs, SPADE, PatchGAN, CFG/EMA, medical image synthesis, cross-domain MRI translation, RAovSeg |
| [03_data_and_pipeline.md](03_data_and_pipeline.md) | 3. Data and downstream pipeline | UT-EndoMRI D1/D2, splits, sacred test set, 6-channel labels, RAovSeg (ResClass + AttUSeg + postprocess + evaluate), preprocessing chain and the ovary enhancement rule |
| [04_methods.md](04_methods.md) | 4. Methods | 2D conditional DDPM backbone, concat vs SPADE conditioning, PatchGAN + λ schedule, CFG/EMA/6-channel/ISCS, Phase 2 cross-domain setup, ablation-parity principle |
| [05_experiments_and_results.md](05_experiments_and_results.md) | 5. Experiments and results | Phase 1 generator quality (2×2 map), Phase 1 downstream (v1 → v2 → v3 → Options B/C), n=8 variance study, Phase 2 exp2 catastrophic collapse, exp2_lam05 placeholder |
| [06_discussion_and_conclusion.md](06_discussion_and_conclusion.md) | 6. Discussion and conclusion | Four headline claims, meta-lessons, limitations, future work, conclusion |
| [07_appendix_reproducibility.md](07_appendix_reproducibility.md) | 7. Appendix | GitHub repo pointer, HPC layout, YAML configs, SLURM scripts, per-experiment reproduction recipes |

## Placeholders to fill

- **exp2_lam05 DSC results** in `05_experiments_and_results.md` §5.6.2
  and `05_experiments_and_results.md` §5.7 (row `Phase 2 exp2_lam05`).
  Interpretation matrix already written in §5.6.2 — paste in numbers
  and pick the row.
- **GitHub URL** in `07_appendix_reproducibility.md` §7.1.
- **Full Liang et al. (2025) citation** with volume/pages when
  published, `07_appendix_reproducibility.md` §7.8.

## Relationship to existing project MDs

The dissertation docs consolidate the following existing MDs (which stay
as-is for detail-level reference):

- `../docs_archive/PAPER_OUTLINE.md` → primary source for 01 and 06
- `../docs_archive/RESULTS_2x2.md` → primary source for 05 §5.2
- `../docs_archive/EXP1B_SUMMARY.md`, `../docs_archive/EXP1C_SUMMARY.md` → source for 04 and 05 §5.2
- `../docs_archive/RAOVSEG_AUGMENTATION_EXPERIMENT.md` → primary source for 03 and 05
  §5.3 – §5.5, plus 06 §6.1
- `../docs_archive/TRAINING_OVERVIEW.md` → source for 04
- `../docs_archive/architecture_dataflow_v2.md` → source for 03 and 04
- `../docs_archive/synthetic_mri_generator_design.md` → source for 04
- `../docs_archive/NEXT_STEPS.md` → source for 06 §6.4 (future work) and 07
- `../docs_archive/stanage_cheatsheet.md` → source for 07
- Memory files (`project_overview.md`, `project_phase2_design.md`,
  `project_phase2_result.md`, `project_variance_findings.md`,
  `feedback_ablation_parity.md`) → cross-cutting design principles
  woven into 01, 04, 05, 06

If a number in a dissertation doc conflicts with the source MD, the
dissertation doc's number is canonical (source MDs were snapshots at
different points; dissertation docs are the current consolidated view).
