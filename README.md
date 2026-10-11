# Reference-anchored diagnostics of single-cell geometry and native allocation coherence

Research release 2.0.0 accompanying the manuscript of the same title by Zeyu Fu. The study asks how graph construction, chance expectations, reference choice and allocation baselines change the interpretation of geometric preservation scores for single-cell representations. It separates reachability, edge survival and separation from allocation coherence, calibrates them on controlled synthetic interventions, and examines 24 stored representations, seven tissue sections and one blood sample.

- Repository: [PeterPonyu/reference-anchored-geometry](https://github.com/PeterPonyu/reference-anchored-geometry)
- This version: [10.5281/zenodo.23299047](https://doi.org/10.5281/zenodo.23299047). All versions: [10.5281/zenodo.22914408](https://doi.org/10.5281/zenodo.22914408), which resolves to the latest version.
- Citation: `CITATION.cff`. Changes since 1.0.0: `CHANGELOG.md`.

The objects analysed are fixed computational objects, not independent biological replicates, and the results make no claim of biological or method superiority. `AUDIT.md` states what was verified and where the boundary lies. `CLAIM_LEDGER.md` maps each family of manuscript claims to its evidence.

## Contents

| Path | Contents |
| --- | --- |
| `paper.tex`, `paper.pdf`, `references.bib`, `standalone-assets/` | Manuscript source, compiled manuscript, bibliography and LaTeX preamble |
| `figures/`, `tables/` | Twelve vector figures (one with an editable SVG source) and the table sources used by the manuscript |
| `scripts/` | Scorers, analysis drivers, figure builders and the manuscript audit |
| `validation/` | Independent decimal oracle, 19 numerical tests and the validation protocol for the identity-explicit scorer |
| `evidence/measurement-inventory/` | Stored calibration results and synthetic arrays, the score inventory and the native-allocation null |
| `evidence/scale-correspondence/` | The 24 matched candidate and reference arrays with checksums, protocol and sensitivity results |
| `evidence/paired-grid-permutation/`, `evidence/robustness-audit/` | Paired grid contrasts, 999-draw permutation streams and independent numerical checks |
| `evidence/raag-v3/` | Reports of the identity-explicit scorer: fixture validation and the remeasurement of all 444 stored batteries |
| `evidence/spatial-reference-20261009/`, `evidence/diversity-20261009/` | Protocols and saved scores for the tissue-section and blood objects, and the coordinate views plotted in the manuscript |
| `evidence/followup-20261010/` | Chance correction, reference-aware component statistics, graded calibration, coherence baselines and refits, with the regenerated representation arrays |
| `evidence/submission-20261011/` | Tie sensitivity of tissue references and the diagnosis of the official trajectory metric |
| `release-manifest.json`, `SHA256SUMS`, `figure-build.json`, `requirements.txt` | Checksums of every distributed file, the figure-build inputs and the pinned software environment |

Each evidence folder has its own protocol or README where one is needed.

## Reproduce

Python 3 with NumPy, SciPy, pandas and scikit-learn runs everything below except where noted; the pinned versions are in `requirements.txt`. All commands run from the repository root. No optimizer update, GPU or network access is used unless stated.

```bash
# numerical tests of the identity-explicit scorer against the decimal oracle
python3 -m unittest validation.test_raag_v3

# fixed-object checks on the 24 stored objects
python3 scripts/paired_grid_permutation.py
python3 scripts/robustness_check.py
python3 scripts/tied_graph_probe.py
python3 scripts/scale_correspondence.py

# tissue-reference ties (about seven minutes) and the official trajectory diagnosis (about one minute)
python3 evidence/submission-20261011/scripts/tie_sensitivity.py
python3 evidence/submission-20261011/scripts/trajectory_diagnosis.py     # needs scanpy and scib 1.1.7

# figures and manuscript
python3 scripts/build_architecture.py && /usr/bin/python3 scripts/render_schematic.py figures/architecture.svg figures/architecture.pdf
Rscript scripts/build_figures.R
Rscript scripts/build_followup_figures.R
bash build.sh

# consistency audit of the compiled manuscript against the saved evidence
python3 scripts/audit_manuscript.py
```

The fixed-object scripts verify input checksums before computing and overwrite only their own result files. Plotting needs R with ggplot2, gridExtra, gtable, jsonlite and Cairo; the SVG conversion needs PyGObject, librsvg and Cairo and uses `/usr/bin/python3` unless `SVG_PYTHON` names another interpreter. Figure text is Arial. The manuscript needs XeLaTeX, BibTeX, Arial and the packages in `standalone-assets/preamble.tex`. `scripts/audit_manuscript.py` additionally needs PyMuPDF and `pdffonts` (poppler-utils), and the compiled manuscript, so run `bash build.sh` first. Rebuilding changes the bytes of PDF files, not their content, so the audit compares rebuilt PDFs with the checksum list only as a warning.

Scripts that read raw expression data or retrain scVI and UMAP take their input locations from environment variables (`RAAG_SPATIAL_DATA`, `RAAG_SPATIAL_EXTERNAL`, `RAAG_PBMC3K_ARCHIVE`); the expected folder layouts, the GPU requirement and the two vendored files of the synthetic fixture generator are described in `evidence/followup-20261010/README.md`. The saved outputs of those scripts are included, so the figures, tables and audit do not need the raw data.

The two numerical drivers of the identity-explicit scorer refuse to start without a run receipt. `scripts/make_replay_receipt.py` writes one for the files in a checkout; see `evidence/raag-v3/README.md`.

## What is and is not included

Included: the manuscript, figures and tables; all code; the synthetic calibration arrays; the 24 matched representation and reference arrays; the representation arrays, tissue coordinates and spot or cell identifiers of the seven tissue sections and the blood sample; every saved score table.

Not included: the count matrices of the cited public datasets, the complete expression caches of the four stored backgrounds, the remaining stored latent files of the historical inventory, and the script that ran every metric library on the calibration arrays. The public datasets are available from the sources cited in the manuscript. The release therefore supports the documented fixed-object analyses, the follow-up analyses from saved arrays and the reconstruction of every figure and table, but not regeneration of every historical model from raw expression data.

## Rights

Author-owned software is MIT licensed. The author's manuscript, figures and generated results are CC BY 4.0. Source-study data, labels, annotations and other third-party materials keep their original terms and are not relicensed. Read `LICENSE`, `LICENSE-MIT`, `LICENSE-CC-BY-4.0` and `NOTICE.md` before reusing mixed-content files.
