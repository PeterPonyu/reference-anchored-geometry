#!/usr/bin/env python3
"""Audit the current Scientific Reports documents and frozen numeric claims.

The predecessor's stricter consolidated-layout audit and build script remain
byte-exact in provenance/previous-version, with their original version identity.
Current journal layout uses the declared generator's citation/float/size rules,
full-page font and bounds checks, and independent recorded visual inspection.
"""
from pathlib import Path
import json
import sys
from verify_journal_candidates import check
from verify_saved_claims import verify

ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    check(ROOT)
    report = verify(ROOT)
    report['document_scope'] = 'Current Scientific Reports main, inline main and supplement'
    print(json.dumps(report, indent=2))
    if '--write' in sys.argv:
        target = ROOT / 'qa/scientific-reports/claims-audit.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2)+'\n')
    sys.exit(0 if report['status'] == 'PASS' else 1)
