#!/usr/bin/env python3
"""Replay RAAG quantities on the archived synthetic arrays, without fitting.

This verifies the RAAG stack of 36 saved calibration rows. It does not rerun
the separate annotation-aware or historical external metric libraries.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from local_raag import raag_battery

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Fresh directory for the replay receipt')
    args = parser.parse_args()
    target = args.output_dir.resolve()
    target.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'evidence/measurement-inventory'
    saved = json.loads((source / 'calibration.json').read_text())
    arrays = {seed: np.load(source / 'calibration_arrays' / f'seed{seed}.npz')
              for seed in saved['seeds']}
    rows, comparisons, largest = [], 0, 0.0
    for row in saved['rows']:
        seed, candidate = row['seed'], row['candidate']
        arr = arrays[seed]
        actual = raag_battery(arr['coords'], arr[candidate], k=15,
                              random_state=seed, already_simplex=False)
        expected = row['stacks']['raag']['values']
        for key, value in expected.items():
            observed = actual[key]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                difference = abs(float(value) - float(observed))
                largest = max(largest, difference)
                assert np.isclose(observed, value, atol=1e-12, rtol=1e-10), (
                    seed, candidate, key, value, observed)
            elif value is None:
                # The archived JSON encodes non-finite, unquotable diagnostics
                # as null; the scorer returns NaN for those same missing values.
                assert observed is None or (isinstance(observed, (int, float))
                                            and np.isnan(observed)), (
                    seed, candidate, key, value, observed)
            else:
                assert observed == value, (seed, candidate, key, value, observed)
            comparisons += 1
        rows.append({'seed': seed, 'candidate': candidate, 'status': 'PASS'})
    receipt = {'status': 'PASS', 'rows': len(rows), 'comparisons': comparisons,
               'maximum_absolute_difference': largest,
               'absolute_tolerance': 1e-12, 'relative_tolerance': 1e-10,
               'scope': 'RAAG stack only; no new optimization or external metric replay',
               'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [
                   source / 'calibration.json',
                   *sorted((source / 'calibration_arrays').glob('seed*.npz')),
                   ROOT / 'scripts/local_raag.py']}, 'row_checks': rows}
    (target / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ('status', 'rows', 'comparisons',
                                             'maximum_absolute_difference')}))


if __name__ == '__main__':
    main()
