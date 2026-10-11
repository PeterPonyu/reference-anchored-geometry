# RAAG-v3 P1 fixture protocol — frozen before execution

Protocol identity: `raag-v3-p1-20261009-c`. Scientific fixtures and tolerances remain frozen from `raag-v3-p1-20261008-a`. Revision b corrected process-RSS instrumentation; revision c uses compensated row totals after a retained small-range precision failure. Execution statuses live in separately identified run reports.

This protocol checks a measurement implementation, not biological usefulness. It is separate from the archived RAAG-v2 calibration, 444 fixed-object batteries and both 999-draw streams. All manuscript numbers, figures, PDFs and historical manifests still describe their archived computations. This source revision does not reassign those outputs to the new scorer.

## Inputs, identity and graph contract

Use `scripts/local_raag_v3.py`, the independently authored `validation/raag_v3_oracle.py`, and `validation/test_raag_v3.py`. The oracle must not import the production scorer. It uses ordinary Python distance enumeration and ID sorting, breadth-first hops and precision-80 Decimal logarithms. Ordered IDs are explicit unique `(namespace, nonnegative integer index)` pairs; no positional fallback. For saved inputs, a dataset namespace plus unchanged `test_idx` is a computational identity, not a recovered biological barcode.

The production graph uses float64 squared Euclidean distances, excludes its diagonal explicitly, obtains full distance rows before exact cutoff-tie selection, chooses exactly `min(k,n-1)` directed nonself neighbours, then takes an undirected union. Canonical namespace/numeric-index ordering breaks exact distance ties. The oracle independently enumerates the same mathematical contract. Require exact adjacency in persistent-ID space and exact directed degree/zero diagonal. Deterministic tie selection is bookkeeping, not biological correctness, uniqueness of tied geometry or near-tie robustness. Constant reference or candidate geometry is noninformative; a constant reference also withholds reference-edge coherence.

Source tests cover the archived 20-zero-row counterexample, duplicate groups larger than `k+1`, distinct equidistant points, boundary ties, missing/duplicate/misaligned IDs, nonfinite/invalid shapes, noninteger/nonpositive k, explicit native semantics, constant support guards, recipient identities in a correspondence intervention, joint row reorder, and a tie-free geometric oracle. Check `k=1`, `k=5` and `k=n-1`; software tests are not all 444 remeasurements.

## Absolute-variation ladder and controls

There are 32 cells, reference `x_i=i/31`, `i=0..31`, one reference coordinate, `k=5`, IDs `("raag-v3-tail-fixture", i)`.

Define nine-coordinate rows exactly as rational numbers:

`q_ij = ((31-i)(j+1)+i(9-j))/(31*45)`, `j=0..8`.

For every epsilon, the ten-coordinate input is `p_i=(1-epsilon, epsilon*q_i)`. The primary ladder is `{1, 10^-1, ..., 10^-14, 0}`. The oracle defines `D_q` as the unclipped mean JSD over all 496 distinct q pairs. Before production outcomes are read, add the deduplicated points `{0.5,1,2}` times each of:

- Floor onset for the smooth q: `f/min(q)=45*f`.
- Complete tail flooring for smooth q: `f/max(q)=5*f`.
- Ideal, unclipped denominator boundary: `g/D_q`.

Here `f=g` is the exact Decimal representation of the delivered float64 `10^-12` constant. Derivation of `D_q` is independent protocol arithmetic, not an outcome-selected tuning step. These points bracket specified **ideal** boundaries; clipping/renormalisation can shift the implemented guard crossing, so a claimed actual crossing must come from the independent measured-input status calculation. Do not drop disappointing or redundant points after execution. Persist the Decimal epsilon and the float64 epsilon actually used.

Run the complete ladder for five prespecified tail constructions:

1. Smooth q, unchanged.
2. Correspondence-damaged q, recipient row `i` receives donor `(13*i+7) mod 32`. This fixed bijection is the declared shuffled stream, not a fresh random selection; retain recipient IDs and save the donor map.
3. Alternating vertices: even rows have tail mass at coordinate zero, odd rows at coordinate eight.
4. Constant q: every row equals smooth `q_0`.
5. Uniform q: every tail coordinate equals `1/9`.

Also include a ten-coordinate varying near-uniform control: `(1-10^-4)/10` in every coordinate, with an additional `10^-4` in coordinate zero for even rows and coordinate nine for odd rows. Its predeclared expected status is `near_uniform`; both constant constructions and epsilon zero have expected status `degenerate`, irrespective of their perplexity flag. For other cases, expected status is the independently calculated priority rule below, not the production status or a favoured coherence value. No utility threshold, monotonicity assertion or biological endpoint is introduced.

Independently check the **ideal unclipped** identity `D_all((1-epsilon,epsilon*q))=epsilon*D_q` on the primary ladder using exact Decimal q and epsilon. That identity is not applied to the implementation's floored/normalised arrays.

## Measured inputs, expected statuses and tolerances

The executable checks operate on the same delivered float64 probabilities. The oracle converts those floats exactly to Decimal, independently floors entries below the **float64 value** of `f`, and independently normalises each row. It computes JSD by its defining logarithmic expression, not the production's even-series approximation. Record input arrays, measured arrays, flooring events, reference directed/union adjacency, `D_edge`, `D_all`, unclipped relative ratio, clipped/guard-convention raw coherence, quotable coherence, pairwise total variation, maximum coordinate range, mean perplexity and both guard flags.

Expected status uses the independent measured arrays and reference support, in this order: unsupported reference -> `unsupported_reference`; `D_all <= g` -> `degenerate`; mean perplexity divided by K `>0.95` -> `near_uniform`; otherwise `eligible`. A non-native input has `non_native` allocation status and separately named softmax-coordinate diagnostics. Quotable coherence is null outside eligible native inputs. Guarded raw coherence follows the disclosed zero convention for a degenerate denominator; retain the unguarded ratio separately wherever defined.

Frozen comparison rule is `abs(actual-expected) <= atol + rtol*abs(expected)`:

| Quantity | atol | rtol |
| --- | ---: | ---: |
| JSD, each local/global mean | 1e-28 | 1e-10 |
| Mean pair TV / max coordinate range | 2e-16 | 1e-10 |
| Probabilities actually measured | 2e-16 | 2e-15 |
| Mean perplexity / perplexity fraction | 1e-12 | 1e-12 |
| Applicable coherence or relative ratio | 1e-12 | 1e-10 |
| Applicable graph scores | 1e-12 | 0 |
| Ideal Decimal scaling identity | 1e-70 | 1e-65 |

Adjacency in logical-ID space, canonical graph hash agreement under reorder, floor counts, null/quotable handling and eligibility status require exact agreement. The ordered-ID hash deliberately changes with row order; it is not the canonical graph hash. A status disagreement is a failure, not something to hide inside a scalar tolerance. The JSD absolute floor is substantially below the guard; the tolerance does not certify all possible tiny/ill-conditioned inputs. Tolerances are protocol choices awaiting empirical verification, not established accuracy bounds. If they fail, preserve the failure and classify it before changing code; any protocol/tolerance change gets a new identity and the first outcomes become design evidence.

The relative-ratio tolerance applies only where the independent denominator exceeds the guard (including a near-uniform case). Below that guard, retain any defined unguarded ratio for audit but do not certify its relative accuracy: dividing two tiny values is ill-conditioned. Require the exact degenerate raw-zero convention and null quotable value there, and still apply the stringent absolute JSD comparison.

## Prespecified precision probes

For each dtype `float32` and `float64`, use the same four 2D points `(0,0),(1,0),(-1,0),(0,2)` with IDs ordered `0..3`, `k=1`. Probe the third point as `(-1+delta,0)` for `delta in {0,+2^-25,-2^-25,+2^-40,-2^-40}`. Also use near-duplicate points `(0,0),(2^-30,0),(2^-29,0),(1,0)` with `k=1`. Evaluate each actual cast matrix against the independent graph. Save actual coordinates, cutoff tie/gap metadata, directed-neighbour membership differences between precision/perturbation cases, and applicable score differences. Exact graph/scalar invariance is required only for a joint reorder of unchanged delivered coordinates and IDs, not for coordinate changes or float32 rounding. A precision-induced change is a reported response, not automatically a test failure or a stability guarantee.

## Execution, admission and output safety

No calculation is admitted by this document. `resource-admission.md` in the coordinator evidence directory records the closed drain gate. Numerical invocation requires a separately authorised admission receipt bound to the exact current protocol/scorer/oracle/test/driver hashes. The runner checks that receipt, never writes into historical output directories, and refuses an existing output directory. Inputs are generated fixtures only; no checkpoints, GPU, network access, fitting, optimizer updates or new dependency.

The first admitted run is a pilot (smooth epsilon 1, 10^-10 and zero; no production suite). Proposed ceiling: one numerical-library thread, 120 seconds wall time and 512 MiB peak RSS. Pilot objectives are the frozen numerical error tolerances above plus measured wall/RSS. A later full fixture run requires a separate receipt with the successful pilot report hash and an independently approved budget (hard ceiling 600 seconds wall and 512 MiB peak RSS). Exceeding a ceiling, any numerical failure, incomplete suite or absent input ends that run without PASS; do not restart automatically or relax thresholds. No calendar or runtime completion prediction is made before the pilot.

After admission, commands from the revision root will be:

```bash
python3 scripts/validate_raag_v3.py --pilot --admission-receipt /absolute/path/to/approved-pilot.json --output-dir evidence/raag-v3/pilot-unique
python3 scripts/validate_raag_v3.py --admission-receipt /absolute/path/to/approved-full.json --output-dir evidence/raag-v3/full-unique
```

The full runner records the unit suite as a separately named check, the ladder, ideal scaling and precision probes. A fixture PASS cannot close P2/P3, public replay, manuscript/PDF QA, external-anchor validation, author approval, venue or archival gates.

The first pilot stopped because Linux `getrusage` retained a 661.52-MiB launcher peak across exec, while `/proc/self/status` reported a fresh 11.08-MiB process peak in the diagnostic. The revised Linux budget measurement uses process `VmHWM`; the inherited `getrusage` value is retained separately. The 512-MiB ceiling, all scientific cases, expected statuses and numerical tolerances are unchanged. The failed first report is preserved, not promoted to a numerical pass. Any non-Linux fallback is explicitly labelled.

The first full fixture run passed all 19 unit tests but failed the maximum-coordinate-range tolerance at smooth epsilon `1e-11`: the float64 row reduction gave `1.7780221739371882e-12` versus the independent Decimal value `1.7777777777568395e-12`, an absolute error of approximately `2.44e-16`. The corrected scorer uses `math.fsum` for row-normalisation totals. The failure remains design evidence; no tolerance, guard or fixture was changed. All affected fixtures must be rerun on the new source identity before reporting a pass.
