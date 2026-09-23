# Independent fixed-object numerical audit

This is an explicitly post hoc check of published-in-draft results. Recompute
all 216 k/d grid scores on the 24 hash-verified arrays using a separate SciPy
implementation: explicit diagonal exclusion, stable distance sorting, an
undirected union graph, and graph shortest paths. Count exact ties across each
neighbour cutoff. Require geometry agreement within 1e-12.

For 12 native objects independently calculate the full Jensen-Shannon matrix
using scipy.special.xlogy and the exact distinct-pair denominator. Reproduce
the original 999-draw stream (seed 20260922); then run a separately chosen
stream (seed 104729), each reinitialized per object. Report both separately;
do not pool them into a retrospectively changed primary test. Include ties in
the one-sided null tail and retain the +1 correction. Apply Holm adjustment to
the declared family of 12 original fixed-object p-values, explicitly post hoc.
The family excludes historical 144-run results and other measurement endpoints.
No draw, setting, or repeated object is a biological replicate. No training or
new biological data are involved.

Additional failure probe: twenty identical candidate coordinate rows, a fixed
seed-91 Gaussian reference, k=5, and joint permutations with seeds 0/17/42.
Retain the archived neighbour routine; record diagonal self-loops and changes
in graph scores rather than rewriting the frozen historical estimator.
