# Follow-up analyses

This folder extends the archived measurements with a connected-reference identity for reachability, candidate-specific chance corrections, a graded multi-topology calibration, reference-aware coherence baselines and fit-to-fit refits. The manuscript interprets them; this folder holds the code and the saved outputs behind its figures and tables.

## Scorer and additions

All reachability, edge-survival and separation values come from the archived scorer `scripts/local_raag.py`, imported unchanged; coherence uses its `allocation_coherence`. Quantities the archived scorer does not provide are defined in `scripts/common.py`: exact permutation-null expectations of edge survival and separation (derivation in the file header), the corrected scores E\* and S\*, a tie-aware far-versus-edge AUROC, and pair-counting scores for reference and candidate component partitions.

Statistical units. Fixture seeds, model seeds and permutation draws are computational replicates, not biological ones. The 24 scale objects come from four backgrounds, and the three seeds of a background share one reference matrix, so they are not independent references.

## Scripts and saved outputs

| Script | What it computes | Saved outputs |
| --- | --- | --- |
| `regen_diversity_arrays.py` | Re-runs the archived preprocessing and training recipe for the eight diversity objects and saves the scored arrays | `arrays/diversity/*.npz`, `results/regen_diversity_check.json` |
| `a1_reachability_degeneracy.py` | Reachability against the candidate component-pair fraction in every connected-reference battery; reference-aware component scores on a fragmented-reference fixture | `results/a1_reachability_identity.csv`, `a1_fragmented_reference_fixture.csv`, `a1_summary.json` |
| `a2_chance_correction.py` | Exact and Monte Carlo null expectations E0 and S0, corrected scores E\* and S\*, and the far-versus-edge AUROC for the scale and diversity objects; UMAP-specific null; ranking changes; native-minus-continuous contrasts | `results/a2_*.csv`, `a2_summary.json` |
| `a3_comparators_and_graded_calibration.py` | Stored comparator panel; graded calibration with independent candidate noise on five topologies | `results/a3_stored_comparators_{long,wide}.csv`, `a3_graded_raw.csv`, `a3_graded_response_summary.csv`, `a3_neighbour_scale.csv` |
| `a3c_noise_coupling_check.py` | Whether reference and faithful jitter in the archived calibration arrays are the same normal deviates | `results/a3c_noise_coupling.json` |
| `a4_coherence_baseline.py`, `a4b_tempered_gmm.py` | Unclipped coherence of native allocations against mixture and random-softmax baselines, perplexity-matched (tempered) baselines, a paired cell bootstrap, the native permutation null, and Geary and Moran analogues | `results/a4_*.csv`, `a4_summary.json`, `a4b_*` |
| `a5_refits_and_external_anchor.py` | Five-seed refits of scVI and UMAP on three tissue sections; association of corrected edge survival with cortical-layer purity on two sections | `results/a5_refit_scores.csv`, `a5_summary.json` |
| `make_tables.py` | Compact tables from the saved results | `results/tables.md` |
| `build_manifest.py` | SHA-256 of every input, script and output, plus package versions | `MANIFEST.json` |

The tie sensitivity of tissue references and the diagnosis of the official trajectory metric are in `../submission-20261011/`.

## Regenerated diversity arrays

The first diversity run saved scores only. `regen_diversity_arrays.py` re-ran the archived `representations` recipe and loaders unchanged and re-scored every array with the archived scorer. `results/regen_diversity_check.json` compares each regenerated raw score triple with the archived one: of 75 triples, 73 are bit-identical. The two exceptions are UMAP on one prefrontal section, one per reference. All chance-correction results use the regenerated arrays.

## Re-running

Package versions are pinned in the repository's `requirements.txt` and recorded again in `MANIFEST.json`. A1, A2, A3, A3c, A4, A4b and `make_tables.py` read only files in this repository: the synthetic fixture generator that A1, A3 and A3c import is vendored in `upstream/audit/` (`upstream_validation.py` and `reference_metrics.py`, whose SHA-256 digests are pinned in the manifest; the vendored versions are the ones the analyses used). Set `RAAG_UPSTREAM` to use another copy. A3 is long: it scores 8,160 rows and took about 26 minutes on 22 processes.

`regen_diversity_arrays.py` and `a5_refits_and_external_anchor.py` need the public datasets, whose locations are read from environment variables and whose file names and SHA-256 digests are listed in `MANIFEST.json`:

| Variable | Expected layout below it |
| --- | --- |
| `RAAG_SPATIAL_DATA` | `st_stahl2016_mob_rep11/Rep11_MOB_0_expr.txt` and `Rep11_MOB_0_location.txt`; `dlpfc_libd_151673/151673_filtered_feature_bc_matrix.h5`, `tissue_positions_list.txt` and `layer_guesses_151673.csv`; `visium_bclm_A_0069226/A_0069226_matrix.mtx.gz`, `_barcodes.tsv.gz`, `_features.tsv.gz` and `_tissue_positions.csv` |
| `RAAG_SPATIAL_EXTERNAL` | `Banksy/data/DLPFC/151507/151507_raw_feature_bc_matrix.h5` and `tissue_positions_list.txt`; `Banksy/data/DLPFC/barcode_level_layer_map.tsv`; `SEDR_analyses/data/Human_Lymph_Node/`, `SEDR_analyses/data/BRCA1/V1_Human_Breast_Cancer_Block_A_Section_1/` and `SEDR_analyses/data/Human_Ovarian_Cancer/`, each with `filtered_feature_bc_matrix.h5` and `spatial/tissue_positions_list.csv` |
| `RAAG_PBMC3K_ARCHIVE` | the file `pbmc3k_filtered_gene_bc_matrices.tar.gz` (10x Genomics filtered matrices) |

Retraining scVI additionally needs `scvi-tools` and `torch` and a GPU: the trainers request `accelerator="gpu"`. Trained values can differ across hardware and library builds, which is why the saved arrays are distributed.
