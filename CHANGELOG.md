# Changelog

## 2.0.2 — 2026-10-12

Adds Scientific Reports main and supplementary documents, separate code availability, Gmail-first correspondence and version-specific source archive. Adds a portable stored-array calibration replay (36 rows, 1,224 exact field matches); clarifies legacy scorer docstrings with executable-AST identity and original-byte provenance. Numerical research outputs and geometry implementations are unchanged.

## 2.0.1 — 2026-10-11

Published pagination correction: one sentence interrupted by a table was repaired, with stronger page-flow audit checks. DOI 10.5281/zenodo.23301675.

## 2.0.0 — 2026-10-11

Accompanies the manuscript *Reference-anchored diagnostics of single-cell geometry and native allocation coherence*, which replaces the study described by release 1.0.0.

### Added

- **Identity-explicit scorer and its validation.** `scripts/local_raag_v3.py`, an independent decimal oracle, 19 numerical tests and a validation protocol (`validation/`); fixture validation and a remeasurement of all 444 stored batteries with no disagreement (`evidence/raag-v3/`); `scripts/make_replay_receipt.py` to replay them.
- **Follow-up analyses** (`evidence/followup-20261010/`). The connected-reference identity of reachability and a component-based alternative for fragmented references; permutation-null expectations and corrected edge survival and separation for every candidate; a tie-aware far-versus-edge AUROC; a graded calibration with independent candidate noise on five topologies and a comparator panel; reference-aware coherence baselines, including sharpness-matched mixtures; five-seed refits and a cortical-layer anchor; regenerated representation arrays for seven tissue sections and one blood sample.
- **Tissue and blood objects** (`evidence/spatial-reference-20261009/`, `evidence/diversity-20261009/`): protocols, saved scores and the coordinate views plotted in the manuscript.
- **Tie sensitivity of tissue references and diagnosis of the official trajectory metric** (`evidence/submission-20261011/`).
- **Consistency audit** of the compiled manuscript against the saved evidence (`scripts/audit_manuscript.py`), `scripts/make_release_manifest.py`, and `requirements.txt` with the pinned software environment.
- Vendored copies of the two fixture-generator modules used by the follow-up analyses (`evidence/followup-20261010/upstream/audit/`), so those analyses no longer need a companion checkout.
- Six new figures (reachability and graded calibration, chance correction, coherence baselines, spatial references, coordinate views, blood sample) and two new tables; twelve figures and four tables in all. `references.bib` replaces the two earlier bibliography files.
- `CHANGELOG.md`, `CLAIM_LEDGER.md`, and READMEs for the validation folder and the evidence folders that need one.

### Changed

- The manuscript was rewritten around the findings above. With a connected reference, reachability reduces to the fraction of candidate pairs in the same component; edge survival and separation are compared with each candidate's own permutation null; native allocation coherence is read against sharpness-matched baselines; representation claims are limited to the saved fixed objects.
- Five figures of release 1.0.0 were redrawn from unchanged records; the schematic was relabelled. `scripts/build_figures.R` no longer draws the superseded disconnection-response figure.
- Workstation paths in scripts and saved records were replaced by environment-variable lookups and neutral names, and docstrings that pointed to removed working files were reworded. No computed value changed.
- `README.md`, `AUDIT.md`, `NOTICE.md`, `CITATION.cff`, `.zenodo.json`, `release-info.json`, `figure-build.json`, `release-manifest.json` and `SHA256SUMS` describe this release.

### Removed

- `figures/graph_disconnection_response.pdf` (superseded), `extra.bib` and `standalone-assets/refs.bib` (merged into `references.bib`).

### Unchanged from 1.0.0

The archived scorer `scripts/local_raag.py`, the four evidence folders of release 1.0.0 (`measurement-inventory`, `scale-correspondence`, `paired-grid-permutation`, `robustness-audit`) and the scripts that read them are byte-identical.

## 1.0.0 — 2026-09-23

First public release: manuscript, seven vector figures, analysis code, synthetic calibration arrays, saved measurement tables and 24 matched representation and reference arrays. [10.5281/zenodo.22914409](https://doi.org/10.5281/zenodo.22914409).
