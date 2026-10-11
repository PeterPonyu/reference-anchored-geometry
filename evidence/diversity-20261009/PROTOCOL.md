# Diversity extension — frozen before execution

Date: 2026-10-09. Objects and methods below were fixed before scores were read. A disappointing object or method is retained.

## Questions

1. Spatial reference: when the reference is tissue position, which expression-only representations preserve that neighbourhood?
2. Expression reference: on the same spots, and on one dissociated blood sample, which representations preserve the expression principal-component neighbourhood?

These are matched-row geometry scores. They are not clustering benchmarks, tumor calls, or native-allocation results.

## Data, already on disk

Spatial, tissue coordinates, in-tissue spots:

- Ståhl mouse olfactory bulb, replicate 11
- Maynard LIBD DLPFC Visium 151673 and 151507
- BCLM Visium A_0069226
- 10x Genomics Visium demonstrations: human lymph node, human breast cancer block A section 1, human ovarian cancer

Dissociated, expression reference only:

- 10x Genomics PBMC 3k filtered count matrix

No further section is added after seeing scores. If a matrix and its coordinates cannot be aligned, that object is recorded as a failure and not replaced.

## Representations

Shared preprocessing: drop genes seen in fewer than 3 spots, library-size normalize, log-transform, keep the 2,000 most variable genes. PCA uses 50 components and seed 42. scVI uses those genes' counts, 10 latent dimensions, one hidden layer, negative-binomial likelihood, at most 200 epochs, early stopping, seed 0, one GPU. UMAP is two-dimensional, fit on the 15-neighbour PCA graph, seed 0. The random projection is a 10-dimensional Gaussian map of the scaled genes, seed 0. The permutation reorders PCA rows once, seed 0.

Neighbour count is 15. Reported geometry scores are reachability, edge survival, and separation. Native coherence is not quoted.

The spatial identity candidate, which repeats the tissue coordinates, is computed only as a scorer check and is not a biological method.
