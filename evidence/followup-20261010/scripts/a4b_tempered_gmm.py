#!/usr/bin/env python3
"""A4b. Sensitivity: GMM responsibilities tempered to the native perplexity.

GMM responsibilities (A4) are sharper than the native allocations (median
perplexity fraction 0.10 vs 0.19).  Here the A4 GMM fits (same settings) are
tempered, r_T ∝ r^(1/T), with T chosen by bisection so the mean perplexity
fraction equals the native object's; C is then recomputed with the archived
allocation_coherence.  Also reports a K=20 diagonal GMM (untempered).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, PAPER, dump  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

from local_raag import allocation_coherence, to_simplex  # noqa: E402


def pf(p):
    return float(np.mean(np.exp(-np.sum(p * np.log(p), 1))) / p.shape[1])


def temper(logr, T):
    # softmax, then the archived 1e-12 clip/renormalise used for native rows
    return to_simplex(to_simplex(logr / T, already_simplex=False), already_simplex=True)


def unclipped(x, p):
    ac = allocation_coherence(x, p, k=15, already_simplex=True)
    return 1 - ac["allocation_edge_js"] / ac["allocation_random_js"], ac["allocation_perplexity_fraction"]


def main():
    meta = [m for m in json.loads((PAPER / "evidence/scale-correspondence/inputs.json").read_text()) if m["native"]]
    rows = []
    for item in meta:
        a = np.load(PAPER / "evidence/scale-correspondence" / item["file"])
        x = a["reference"]
        nat = to_simplex(np.asarray(a["latent"], float), already_simplex=True)
        target = pf(nat)
        for K, cov, cols in ((10, "diag", 50), (10, "full", 10), (20, "diag", 50)):
            gm = GaussianMixture(K, covariance_type=cov, random_state=0, n_init=3, reg_covar=1e-6).fit(x[:, :cols])
            logr = np.log(np.clip(gm.predict_proba(x[:, :cols]), 1e-300, None))
            c_raw, pf_raw = unclipped(x, to_simplex(gm.predict_proba(x[:, :cols]), already_simplex=True))
            lo, hi = 1e-2, 1e4
            for _ in range(80):
                mid = np.sqrt(lo * hi)
                if pf(temper(logr, mid)) < target:
                    lo = mid
                else:
                    hi = mid
            T = np.sqrt(lo * hi)
            c_t, pf_t = unclipped(x, temper(logr, T))
            rows.append(dict(id=item["id"], gmm=f"K{K}_{cov}_{cols}pc", C_native=unclipped(x, nat)[0], native_pf=target,
                             C_gmm_raw=c_raw, pf_raw=pf_raw, T=T, C_gmm_tempered=c_t, pf_tempered=pf_t))
        print(item["id"], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results" / "a4b_tempered_gmm.csv", index=False)
    g = df.groupby("gmm")
    summ = {gm: {"median_C_gmm_raw": float(h.C_gmm_raw.median()), "median_C_gmm_tempered": float(h.C_gmm_tempered.median()),
                 "n_native_exceeds_tempered": int((h.C_native > h.C_gmm_tempered).sum()),
                 "median_native_minus_tempered": float((h.C_native - h.C_gmm_tempered).median()),
                 "range_native_minus_tempered": [float((h.C_native - h.C_gmm_tempered).min()), float((h.C_native - h.C_gmm_tempered).max())]}
            for gm, h in g}
    dump(OUT / "results" / "a4b_summary.json", summ)
    print(json.dumps(summ, indent=1))
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
