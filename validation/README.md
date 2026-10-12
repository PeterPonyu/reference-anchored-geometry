# Validation of the identity-explicit scorer

| File | Role |
| --- | --- |
| `PROTOCOL_RAAG_V3.md` | Fixtures, expected statuses, tolerances and precision probes, fixed before the validation runs |
| `raag_v3_oracle.py` | Independent oracle: ordinary Python distance enumeration, identity sorting, breadth-first hops and precision-80 decimal logarithms. It does not import the production scorer. |
| `test_raag_v3.py` | The 19 numerical tests: self and tie semantics, duplicate groups larger than `k+1`, identity alignment, native semantics, support guards, joint row reorder and a tie-free geometric oracle |

The scorer under test is `scripts/local_raag_v3.py`; the drivers are `scripts/validate_raag_v3.py` and `scripts/remeasure_v3.py`.

These files are distributed exactly as they were when the reports in `evidence/raag-v3/` were produced, because the two fixture reports bind to the digests of all five and the remeasurement binds to those of its driver and scorer. Their headers and the protocol text were written before the runs: statements that the numerical suite has not been run, and references to resource receipts and a coordinator directory, describe that earlier state or internal scheduling and do not affect any computed value. The tolerances in the protocol are the ones the reports use. The protocol also records two failed first attempts (a memory-instrumentation error and a reduction-order error in a row total) and the repairs that followed; the failures were kept as design evidence and no tolerance, guard or fixture was relaxed.

Run the tests from the repository root with `python3 -m unittest validation.test_raag_v3` (a few seconds; `pytest` also works). To replay the fixture validation and the remeasurement, follow `evidence/raag-v3/README.md`.
