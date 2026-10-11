# Spatial-coordinate reference case — frozen before execution

Date: 2026-10-09, America/Los_Angeles. This file is the analysis contract for one supplementary case. It was written before scores were computed. Disappointing datasets are retained.

## Question

When the reference geometry is the tissue coordinate of each spot, do expression-derived continuous representations preserve that spatial neighbourhood better than a row-permuted copy of the same representation?

This is a measurement on matched spots. It is not a tumor-detection claim, a clustering benchmark, or a native-allocation result. Continuous latents are not model-declared probability allocations, so native coherence is not quoted.

## Objects, chosen because the matrices are already on disk

1. Ståhl et al. 2016 mouse olfactory bulb, replicate 11.
2. Maynard et al. 2021 LIBD DLPFC Visium section 151673, in-tissue spots only.
3. The public BCLM Visium sample A_0069226, in-tissue spots only (Zenodo 10.5281/zenodo.18306028).

No other section is added after seeing scores. Layer labels are not inputs to the scores.

## Representations

Reference: the stored two-dimensional tissue coordinates, unscaled. Neighbour graphs use hop distance, so isotropic rescaling would not change the scores.

Candidates, same spot order:

- `spatial_identity`: the reference coordinates themselves. Expected reachability and edge survival are 1 if the scorer is consistent.
- `expression_pca`: PCA of library-size-normalized, log1p expression on highly variable genes. Top 2,000 genes when at least that many remain after the gene filter; otherwise every remaining gene. At most 50 components. PCA seed 42. Genes present in fewer than 3 spots are removed first. Spots with no remaining counts are removed together with their coordinates.
- `expression_scvi`: a 10-dimensional scVI latent of the same highly variable counts, one hidden layer, negative-binomial likelihood, batch size at most 128, at most 200 epochs, early stopping on, seed 0, one GPU.
- `pca_row_permutation`: one permutation of the PCA rows, NumPy Generator seed 0. The reference stays fixed.

Neighbour count `k=15`. Geometry scores are reachability, edge survival, and separation from the historical `scripts/local_raag.py` functions. The same three quantities are reported for every object.

## What a result is allowed to say

A higher edge survival than the matched row permutation means that this representation keeps more of the spatial neighbourhood than a correspondence-destroyed copy of itself. It does not identify a biological boundary, a preferred embedding for every tissue, or a patient-level effect. Each object is one section.
