# Numerical and reproducibility audit

## What was verified, and how

**Identity-explicit scorer.** `scripts/local_raag_v3.py` is tested against an independent decimal oracle that does not import it (`validation/`). All 19 numerical tests pass with none skipped. The fixture validation passes every comparison (`evidence/raag-v3/full-20261009-c/`): graph adjacency in persistent-identity space, JSD, total variation, perplexity, coherence and statuses agree with the oracle within the protocol tolerances.

**The 24 stored objects.** The sensitivity driver evaluates 216 graph and reference settings, 180 row-correspondence perturbations and 48 implementation controls without training. Three further computations agree with the archived scorer:

- a separate dense-distance and Jensen–Shannon implementation recomputes all 216 graph settings (geometry difference 0, largest coherence difference 6.7e-16) and both 999-draw permutation streams;
- the identity-explicit scorer reproduces all 444 batteries and both streams with no disagreement (largest differences: reachability 0, edge survival 2.2e-16, separation 0, native coherence 1.6e-15);
- the 24 objects contain no neighbour-cutoff ties and no zero-distance non-self pairs, so the tie rule of the new scorer is never exercised on them.

**Follow-up analyses** (`evidence/followup-20261010/`). Exact permutation-null expectations of edge survival and separation agree with Monte Carlo means (largest differences 0.0014 and 0.0054). The regenerated tissue and blood arrays reproduce 73 of 75 archived score triples exactly; both exceptions are UMAP on one prefrontal section.

**Tissue-reference ties** (`evidence/submission-20261011/`). Visium coordinates are integer lattice points, so the fifteenth and sixteenth neighbours are equidistant for 2.6–41.4% of spots in six of seven sections. Every tissue-reference comparison was rescored on 62 reference graphs per section (saved order, 30 reference reorderings, 30 random tie breaks, one tie-complete graph). Edge survival and corrected edge survival move by at most 0.0015 and 0.0021, separation and corrected separation by at most 0.013 and 0.043, and every reported ordering holds in every graph.

**Official trajectory metric** (`evidence/submission-20261011/`). The low value of the official scIB trajectory metric on the identity control is reproduced for all saved values (largest difference 2.5e-5). It comes from reversal of both arms in diffusion pseudotime, not from the choice of root, and depends on the number of diffusion components, the neighbour count and the jitter level.

**Clean replay.** The release commit was cloned into empty directories and every documented path was run there: the 19 tests; the paired-grid, robustness, tied-graph and scale scripts; the tie-sensitivity and trajectory-diagnosis scripts; the fixture validation (pilot, then full) and the remeasurement, each started from a replay receipt; the component, chance-correction, comparator, coherence-baseline and table scripts of the follow-up analyses, including a complete recomputation of the 8,160-row graded calibration; all twelve figures; and the manuscript. Every recomputed value matched the saved record to a relative difference of 1e-9, with timings, memory and the recorded scale-protocol digest excluded (the protocol text was reformatted after the historical run). The rebuilt figures and the 36 pages of the rebuilt manuscript were pixel-identical to the distributed files, the validation replays bound the same source digests as the distributed reports, and the audit passed on the rebuilt manuscript. Steps that read raw expression data, the regeneration of the diversity arrays and the five-seed refits, were not replayed because the raw matrices are not distributed.

**Manuscript audit** (`scripts/audit_manuscript.py`). Against the compiled manuscript it checks section structure, that floats appear only in Results, that Introduction and Discussion are continuous prose, citation order and use, banned process language, the abstract length, the LaTeX log, embedded fonts, page-bottom whitespace, float call-out order and numbering, figure typography, the checksum list, and every number that the text takes from saved records.

## Boundary of the claims

The checks apply to the included fixed computational objects. Graph settings, corruption draws, optimizer seeds and repeated file aliases are not independent biological samples, and the study makes no donor-level or method-superiority claim. The historical inventory records files that are not distributed; their identifiers do not claim that those files are available here. Regeneration of every historical model from raw expression data is outside this release.

## Limitations retained

- The archived scorer assumes that the first returned neighbour is the query itself and fails to exclude self-neighbours when exact duplicate coordinates occur. `scripts/tied_graph_probe.py` demonstrates this. The archived scorer is not repaired, because that would change the historical measurement; the identity-explicit scorer removes the assumption. The 24 stored objects were checked separately for boundary ties.
- Tissue references have cutoff ties; the bounds above quantify their effect on the reported comparisons.
- The numerical checks are separate from visual review: PDF text and geometry checks find many layout defects but are not a substitute for reading the pages.

## Provenance and packaging

Producer digests inside historical records identify the computation that produced them. `release-manifest.json` and `SHA256SUMS` hash the bytes actually distributed. Workstation paths in provenance were replaced by environment-variable lookups in scripts and neutral names in saved records; no computed value changed. Files whose digests are bound by validation reports are distributed unmodified. Raw count matrices and run logs are not distributed.
