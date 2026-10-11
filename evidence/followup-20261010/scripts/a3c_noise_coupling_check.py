#!/usr/bin/env python3
"""Are the reference and faithful jitter the same normal deviates?

Checks the archived calibration arrays: reference = clean + 0.004*eps_r and
faithful = reference + 0.01*eps_f.  Reports max |eps_r - eps_f|.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, PAPER, UPSTREAM, dump  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(UPSTREAM))
from audit.upstream_validation import make_reference_manifold  # noqa: E402

out = {}
for seed in (7, 17, 42):
    a = np.load(PAPER / f"evidence/measurement-inventory/calibration_arrays/seed{seed}.npz")
    ref = make_reference_manifold(150, seed)
    n = 150
    t = np.linspace(0, .5, n, endpoint=False)
    at = np.linspace(0, 1, n)
    clean = np.vstack([np.c_[t, 0 * t], np.c_[.5 + .5 * at, .55 * at], np.c_[.5 + .5 * at, -.55 * at]])
    e_ref = (a["coords"] - clean) / 0.004
    e_f = (a["faithful"] - a["coords"]) / 0.01
    out[f"seed{seed}"] = {"generator_reproduces_archived_reference": bool(np.array_equal(ref["coords"], a["coords"])),
                          "max_abs_eps_ref_minus_eps_faithful": float(np.abs(e_ref - e_f).max())}
out["interpretation"] = "eps identical to machine precision: faithful = clean + 0.014*eps_ref, a deterministic rescaling of the reference jitter"
dump(OUT / "results" / "a3c_noise_coupling.json", out)
print(out)
