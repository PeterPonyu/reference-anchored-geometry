# Reports of the identity-explicit scorer

| Folder | What it holds |
| --- | --- |
| `pilot-20261009-c/report.json` | Pilot of the fixture validation: the smooth allocation at three tail masses. A full run requires a successful pilot on the same source files. |
| `full-20261009-c/report.json` | Full fixture validation against an independent 80-digit decimal oracle: 19 unit tests (none skipped), the tail-mass ladder and its controls (126 rows), the ideal scaling identity and the precision probes. Status `PASS_P1_FIXTURES`. |
| `remeasure-20261009-a/` | Remeasurement of the 24 stored objects with the identity-explicit scorer: 216 scale settings, 180 correspondence perturbations and 48 controls (444 batteries) and both 999-draw native row-order streams, compared with the values saved by the archived scorer. Status `PASS_AGREES_WITH_ARCHIVED_V2`, no mismatch under the frozen tolerances. |

The two fixture reports (`pilot-20261009-c`, `full-20261009-c`) record the SHA-256 digests of the exact protocol, scorer, oracle, test and driver files that produced them; the remeasurement records those of its driver and scorer in `results.json`. All of those files are distributed unmodified (`validation/`, `scripts/`), so the recorded digests match the distributed bytes. Where a header or docstring in those files describes a step as not yet run, it was written before the run that these reports record; the reports are authoritative.

## Run receipts

The two drivers (`scripts/validate_raag_v3.py` and `scripts/remeasure_v3.py`) start only when given a JSON receipt that binds the digests of the files they use to a compute ceiling, and they never overwrite an existing output folder. The receipts of the published runs recorded compute ceilings only; they are not distributed, and each report keeps the digest of its receipt. `scripts/make_replay_receipt.py` writes an equivalent receipt for the files in a checkout. The remeasurement tolerances (geometry 1e-12; coherence 1e-10 absolute and relative) are those of the published run.

```bash
python3 scripts/make_replay_receipt.py validate-pilot /tmp/pilot-receipt.json
python3 scripts/validate_raag_v3.py --pilot --admission-receipt /tmp/pilot-receipt.json --output-dir evidence/raag-v3/replay-pilot

python3 scripts/make_replay_receipt.py validate-full /tmp/full-receipt.json --pilot-report evidence/raag-v3/replay-pilot/report.json
python3 scripts/validate_raag_v3.py --admission-receipt /tmp/full-receipt.json --output-dir evidence/raag-v3/replay-full

python3 scripts/make_replay_receipt.py remeasure /tmp/remeasure-receipt.json
python3 scripts/remeasure_v3.py --admission-receipt /tmp/remeasure-receipt.json --output-dir evidence/raag-v3/replay-remeasure
```

The fixture validation takes under a minute on one thread; the remeasurement uses at most two threads.
