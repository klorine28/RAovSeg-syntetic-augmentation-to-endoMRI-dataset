# Dissertation master docs

Consolidated, dissertation-ready master files. Structure matches the
distinction-level Sheffield dissertations aca22mmm (Mahmoud, 2025) and
acu23ns (Selvam, 2025), aligned with the IJC403/404 rubric sections.

**Total main-body target: 12,000 words** (within the 9,000–14,000
range specified in the assessment brief). Research Diary is a separate
1,000-word budget.

## Chapter structure and targets

| Doc | Chapter | Words | Rubric section |
|---|---|---|---|
| `01_introduction.md` | 1. Introduction | 1,000 | Introduction |
| `02_literature_review.md` | 2. Literature Review | 2,500 | Literature review |
| `03_methodology.md` | 3. Methodology | 3,000 | Methodology & Implementation |
| `04_experiments_and_results.md` | 4. Experiments and Results | 3,500 | Results and Discussion (part 1) |
| `05_discussion.md` | 5. Discussion | 1,300 | Results and Discussion (part 2) |
| `06_conclusion.md` | 6. Conclusion | 700 | Conclusions |
| `07_appendix.md` | Technical Appendix | uncounted | — |
| `08_research_diary.md` | Research Diary & Reflection | 1,000 (separate) | Research Diary (in Appendix) |
| **Main body total** | | **12,000** | |

Word-count targets are annotated at the top of each doc and at each
section heading. When drafting, use these to enforce compression on
sections that currently exceed target (Chapter 4 in particular).

## Rubric-graded items and coverage

The IJC403/404 rubric grades: Structured abstract, Introduction,
Literature review, Methodology & Implementation, Results and
Discussion, Conclusions, Research Diary & Reflection, Use of English,
Use of references (APA).

- Structured abstract: **deferred** (write last, ~250–300 words).
- Introduction → doc 01.
- Literature review → doc 02.
- Methodology & Implementation → doc 03 (methodology) + doc 07
  (implementation detail in appendix).
- Results and Discussion → doc 04 (results) + doc 05 (discussion).
- Conclusions → doc 06.
- Research Diary & Reflection → doc 08 (drafted from user interview
  questions).
- Use of English + References → cross-cutting.

## Distinction-example structural parallels

Both distinction-level Sheffield dissertations reviewed use a 5–6
chapter structure with Literature Review as its own chapter. This
dissertation adopts the acu23ns (Selvam) Chapter 2 pattern —
domain-first (clinical → imaging → existing methods → data
scarcity → augmentation → generative solutions → gap) — because the
topic and technical framing are closest to Selvam's Generative AI +
Liver MRI work.

## Placeholders to fill

- **exp2_lam05 DSC results** in `04_experiments_and_results.md` §4.6.
  Interpretation matrix ready in the doc; paste in numbers and pick
  the row.
- **GitHub repo URL** (currently public at
  github.com/klorine28/RAovSeg-syntetic-augmentation-to-endoMRI-dataset).
- **Full Liang et al. (2025) citation** with volume/pages once
  finalised (`07_appendix.md` and Chapter 2 references).
- **Structured abstract** (write last).

## Relationship to existing project MDs

The dissertation docs consolidate the following existing MDs (which
stay in `docs_archive/` for detail-level reference):

- `../docs_archive/PAPER_OUTLINE.md` → primary source for 01 and 06.
- `../docs_archive/RESULTS_2x2.md` → primary source for 04 §4.2.
- `../docs_archive/EXP1B_SUMMARY.md`,
  `../docs_archive/EXP1C_SUMMARY.md` → source for 03 and 04 §4.2.
- `../docs_archive/RAOVSEG_AUGMENTATION_EXPERIMENT.md` → primary
  source for 03 and 04 §4.3–§4.5, 05.
- `../docs_archive/TRAINING_OVERVIEW.md` → source for 03.
- `../docs_archive/architecture_dataflow_v2.md` → source for 03.
- `../docs_archive/synthetic_mri_generator_design.md` → source for 03.
- `../docs_archive/NEXT_STEPS.md` → source for 06 future work and 07.
- `../docs_archive/stanage_cheatsheet.md` → source for 07.

If a number in a dissertation doc conflicts with the source MD, the
dissertation doc's number is canonical (source MDs were snapshots at
different points; dissertation docs are the current consolidated view).
