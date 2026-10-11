#!/usr/bin/env python3
"""Write the run receipts that the two numerical drivers require.

``validate_raag_v3.py`` and ``remeasure_v3.py`` refuse to start unless a JSON receipt
binds the exact bytes of the protocol, scorer, oracle, tests and driver, together with a
compute ceiling. The receipts used for the published runs are not distributed. This helper
writes equivalent receipts for the files in this checkout, so a reader can replay the runs;
the tolerances are those recorded for the published remeasurement.

    python3 scripts/make_replay_receipt.py validate-pilot RECEIPT.json
    python3 scripts/validate_raag_v3.py --pilot --admission-receipt RECEIPT.json --output-dir evidence/raag-v3/replay-pilot

    python3 scripts/make_replay_receipt.py validate-full RECEIPT.json --pilot-report evidence/raag-v3/replay-pilot/report.json
    python3 scripts/validate_raag_v3.py --admission-receipt RECEIPT.json --output-dir evidence/raag-v3/replay-full

    python3 scripts/make_replay_receipt.py remeasure RECEIPT.json
    python3 scripts/remeasure_v3.py --admission-receipt RECEIPT.json --output-dir evidence/raag-v3/replay-remeasure

Receipts are written outside the evidence tree, are never overwritten, and bind the bytes
present at the time they are written; editing a bound file afterwards invalidates them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATION_SOURCES = (
    "validation/PROTOCOL_RAAG_V3.md", "scripts/local_raag_v3.py", "validation/raag_v3_oracle.py",
    "validation/test_raag_v3.py", "scripts/validate_raag_v3.py",
)
REMEASURE_BOUND = (
    "validation/PROTOCOL_RAAG_V3.md", "scripts/local_raag_v3.py", "validation/raag_v3_oracle.py",
    "validation/test_raag_v3.py", "scripts/remeasure_v3.py", "evidence/scale-correspondence/inputs.json",
    "evidence/scale-correspondence/results.json", "evidence/paired-grid-permutation/results.json",
)
REMEASURE_TOLERANCES = {
    "geometry_atol": 1e-12,
    "coherence_atol": 1e-10,
    "coherence_rtol": 1e-10,
    "control_atol": 1e-12,
    "exact_fields": [
        "n_reference_connected_pairs", "n_reference_edges", "n_far_pairs",
        "separation_far_cutoff_hops", "separation_threshold_hops",
    ],
    "statuses": "near-uniform and degenerate flags must agree; permutation exceedance counts for the archived stream must be identical",
    "note": "A disagreement is reported as a failure, not absorbed by loosening a tolerance.",
}
REMEASURE_SCOPE = (
    "Re-score the 24 hash-verified fixed objects (216 scale settings, 180 correspondence perturbations, "
    "48 controls = 444 batteries) and both 999-draw native row-order streams (seeds 20260922 and 104729) "
    "with the identity-explicit scorer; compare with the archived scorer. Descriptive measurement-agreement "
    "check only: no training, GPU, network, new data or change to archived evidence."
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, record: dict) -> None:
    if path.exists():
        sys.exit(f"refusing to overwrite {path}")
    if path.resolve().is_relative_to((ROOT / "evidence").resolve()):
        sys.exit("write the receipt outside the evidence tree")
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print("wrote", path)


def validation_receipt(mode: str, pilot_report: Path | None) -> dict:
    record = {
        "status": "APPROVED",
        "task": "raag_v3_p1_fixtures",
        "mode": mode,
        "authorization_basis": "Replay of the published fixture validation by the reader running it.",
        "drain_resolution": "Not applicable: a replay reserves no shared resource.",
        "numerical_library_threads": 1,
        "max_wall_seconds": 120 if mode == "pilot" else 600,
        "max_peak_rss_mib": 512,
        "source_sha256": {name: digest(ROOT / name) for name in VALIDATION_SOURCES},
    }
    if mode == "full":
        if pilot_report is None:
            sys.exit("validate-full needs --pilot-report from a successful replay pilot")
        resolved = (ROOT / pilot_report).resolve() if not pilot_report.is_absolute() else pilot_report.resolve()
        record["successful_pilot"] = {"report": str(resolved), "sha256": digest(resolved)}
    return record


def remeasure_receipt() -> dict:
    return {
        "schema": "raag-v3-p2-remeasurement-admission-v1",
        "protocol_identity": "raag-v3-p2-replay",
        "issued": datetime.now(timezone.utc).isoformat(),
        "authorisation": "Replay of the published remeasurement by the reader running it.",
        "scope": REMEASURE_SCOPE,
        "ceilings": {"threads": 2, "wall_seconds": 1800, "peak_rss_mib": 2048},
        "frozen_tolerances": REMEASURE_TOLERANCES,
        "bound_sha256": {name: digest(ROOT / name) for name in REMEASURE_BOUND},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="kind", required=True)
    for kind in ("validate-pilot", "validate-full", "remeasure"):
        item = sub.add_parser(kind)
        item.add_argument("receipt", type=Path)
        if kind == "validate-full":
            item.add_argument("--pilot-report", type=Path)
    args = parser.parse_args()
    if args.kind == "remeasure":
        record = remeasure_receipt()
    elif args.kind == "validate-pilot":
        record = validation_receipt("pilot", None)
    else:
        record = validation_receipt("full", args.pilot_report)
    write_new(args.receipt, record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
