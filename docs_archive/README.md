# docs_archive

Source-of-truth reference material consolidated into `dissertation_docs/`.
Kept here for detail-level reference; not primary reading.

If a number here conflicts with `dissertation_docs/`, the dissertation
doc is canonical (these were snapshots at different points).

## Files

| File | Consolidated into |
|---|---|
| `PAPER_OUTLINE.md` | 01 and 06 |
| `RESULTS_2x2.md` | 05 §5.2 |
| `EXP1B_SUMMARY.md` | 04 and 05 §5.2 |
| `EXP1C_SUMMARY.md` | 04 and 05 §5.2 |
| `RAOVSEG_AUGMENTATION_EXPERIMENT.md` | 03 and 05 §5.3 – §5.5, 06 §6.1 |
| `TRAINING_OVERVIEW.md` | 04 |
| `architecture_dataflow_v2.md` | 03 and 04 |
| `synthetic_mri_generator_design.md` | 04 |
| `NEXT_STEPS.md` | 06 §6.4 and 07 |
| `stanage_cheatsheet.md` | 07 |

## Note on broken relative links

Files here may contain relative paths (e.g. `1a/current/explain/sample_00.png`,
`master_metrics.csv`) that assumed they lived at the project root. After
the move, those links resolve as:

- `1a/`, `1b/`, `1c/` → still at project root, so `../1a/...` from here
- `master_metrics.csv` → now `../metrics/master_metrics.csv`
- Diagnostic PNGs → now `../figures/`

These files were not rewritten because they are archives; refer to the
dissertation docs for current, working references.
