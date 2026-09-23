# Numerical and reproducibility audit

## What is supported

The repository includes 24 matched candidate/reference input arrays with immutable SHA256 checksums. The sensitivity driver evaluates 216 graph/reference settings, 180 row-correspondence perturbations and 48 implementation controls without training. A separate implementation checks the graph and allocation calculations, and permutation code evaluates 999 draws per native object with explicit plus-one Monte Carlo correction.

These checks apply to the included fixed computational objects. Repeated settings, seeds and file aliases are not independent biological samples. The historical inventory includes records for files not included in this release; their identifiers do not claim that those files are downloadable here. The raw expression-cache reconstruction and full multi-library historical calibration cannot be repeated from this release alone.

## Known limitation retained

The saved scorer assumes a nearest-neighbour ordering that fails to exclude self-neighbours correctly when exact duplicate coordinates occur. `scripts/tied_graph_probe.py` demonstrates this limitation. It is not silently repaired because doing so would change the historical measurement implementation. The 24 retained objects are checked separately for boundary ties; agreement on those objects does not establish correctness for arbitrary tied data.

## Provenance and packaging

Public provenance substitutes semantic historical-object identifiers for workstation paths and task labels. Original array digests, computation identities, measurements and scientific parameters are retained. The source protocol has been reformatted only to describe its selected objects without workstation-specific labels. Historical code/protocol/input-manifest digests remain historical provenance, not assertions about reformatted release metadata. A new release manifest supplies the distributed-file checksums. The complete historical source files are retained by the author separately from this release.

Numerical verification is separate from visual review: PDF text and geometry checks can find many layout defects but are not evidence of a human visual sign-off.
