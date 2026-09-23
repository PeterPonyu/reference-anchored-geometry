# Reference-anchored geometry diagnostics

## Repository, archive and citation

Study repository: [PeterPonyu/reference-anchored-geometry](https://github.com/PeterPonyu/reference-anchored-geometry). Versioned archive: [10.5281/zenodo.22914409](https://doi.org/10.5281/zenodo.22914409). Use `CITATION.cff` for release 1.0.0; `release-manifest.json` records distributed-file checksums. The archive and source repository describe this study only.

Author-owned software is MIT licensed. The author's manuscript, figures, generated results and model weights are CC BY 4.0. Source-study data, labels, annotations and other third-party materials retain their original terms; they are not relicensed. Read `LICENSE` and `NOTICE.md` before reusing mixed-content files.

This repository accompanies **Reference-anchored geometry diagnostics with noise-matched calibration and native-allocation checks** by Zeyu Fu. It examines how graph construction, reference dimensions, row correspondence and native probability allocation affect the interpretation of representation-quality measurements.

The study separates graph reachability, edge survival and separation from allocation coherence. It includes matched synthetic perturbations, a historical score inventory, and a sensitivity analysis of 24 saved representations across four biological backgrounds. These are fixed computational objects, not independent biological replicates, and the results do not establish recovery of a biological lineage.

## Contents and reproducibility

- `paper.pdf`, `paper.tex` and bibliography: the manuscript and its source.
- `figures/`: seven vector figures, including an editable SVG measurement workflow as Figure 1.
- `scripts/`: graph scoring, row-permutation analysis, numerical cross-checks, plotting and SVG conversion.
- `evidence/measurement-inventory/`: saved calibration results, synthetic arrays, historical score tables and descriptive provenance.
- `evidence/scale-correspondence/`: the 24 included candidate/reference arrays, their checksums, protocol and sensitivity results.
- `evidence/paired-grid-permutation/` and `evidence/robustness-audit/`: paired grid contrasts, permutation results and numerical checks.

All commands below run from the repository root and use its own included inputs. No second research repository is required. The complete historical expression caches, all 1,587 inventoried latent files and the complete historical multi-library calibration driver are **not** supplied. Thus the release supports the documented fixed-object analyses and figure reconstruction, not regeneration of every historical model or every inventory measurement from raw expression data. Historical object identifiers in provenance are descriptive identifiers, not downloadable paths.

## Run the numerical checks

Python 3 with NumPy, SciPy and scikit-learn is required. No optimizer updates or GPU are used.

```bash
python3 scripts/paired_grid_permutation.py
python3 scripts/robustness_check.py
python3 scripts/tied_graph_probe.py
```

The first command calculates paired contrasts over nine graph/reference settings and 999 row-order draws for each of 12 native allocation objects. The second uses a separate dense-distance/Jensen-Shannon implementation to check all 216 grid measurements and two permutation streams. The third intentionally exposes the saved nearest-neighbour constructor's failure on exact duplicate coordinates; this disclosed stress-test failure must not be mistaken for a successful general-purpose tie policy.

The complete included sensitivity calculation can also be repeated:

```bash
python3 scripts/scale_correspondence.py
```

This evaluates 216 scale settings, 180 row-correspondence perturbations and 48 implementation controls. The scripts verify input SHA256 digests before computation and overwrite only their own result files. Protocol and code digests in a saved historical record identify its original computation; `release-manifest.json` identifies the distributed files.

## Rebuild figures and manuscript

Plotting requires R with ggplot2, gridExtra, jsonlite and Cairo. SVG conversion requires Python PyGObject, librsvg and Cairo. The SVG interpreter defaults to `/usr/bin/python3`; set `SVG_PYTHON` to another interpreter with those bindings. Figure text uses Arial. Manuscript compilation requires XeLaTeX, BibTeX, Arial and the packages specified in `standalone-assets/preamble.tex`.

```bash
python3 scripts/export_plot_inputs.py
Rscript scripts/build_figures.R
bash build.sh
```

`bash build.sh` alone compiles from the included vector figures. Raw source-study expression data are not distributed here; the manuscript cites the original hematopoiesis, dentate gyrus, pancreatic endocrinogenesis and lung studies, including lung accession GSE130148. Refer to those providers for original data and applicable terms.

## Interpretation and audit

`AUDIT.md` explains the verification boundary, duplicate-coordinate limitation, and distinction between historical provenance and current release checksums. Graph settings, corruption draws, optimizer seeds and repeated file aliases are not donor-level replication. The benchmark is not evidence of comparative biological superiority.

## Rights

Author-owned software is MIT licensed. The author's manuscript, figures and generated results are CC BY 4.0. Third-party materials, if present, retain their stated original terms; neither license grants new rights to source-study data. See `LICENSE`, `LICENSE-MIT`, `LICENSE-CC-BY-4.0` and `NOTICE.md` for the file-category boundaries.
