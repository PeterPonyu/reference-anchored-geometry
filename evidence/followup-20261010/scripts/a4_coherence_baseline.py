#!/usr/bin/env python3
"""A4. Reference-aware positive baselines for allocation coherence C.

For each of the 12 native objects (k=15, 50-PC reference X):
  * C_native via the archived allocation_coherence (already_simplex=True);
  * C for GMM responsibilities, K=10, fitted on X itself
      - primary: diagonal covariance on all 50 PCs (full covariance in 50-D
        with n=450 is ill-conditioned), random_state 0, n_init 3;
      - sensitivity: full covariance on the first 10 PCs;
  * C for 20 random smooth allocations softmax(beta * Xs W), Xs = X / sd(X),
    W_ij ~ N(0, 1/d), K=10:
      - "matched": beta chosen by bisection so mean row perplexity / K equals
        the native object's value;
      - "beta1": beta = 1;
  * unclipped C = 1 - D_E/D_all (archived C clips to [0,1]);
  * 200 row permutations of the native allocation (unclipped and clipped);
  * multivariate Geary (reported as 1 - Geary's c on squared Euclidean simplex
    distance, which equals the Euclidean analogue of C exactly) and Moran's I
    on the binary union reference graph;
  * paired cell bootstrap (B=1000, multinomial cell weights, reference graph
    held fixed) of C_native - C_baseline.
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, PAPER, dump, knn_hop_graph  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

from local_raag import _jensen_shannon, allocation_coherence, to_simplex  # noqa: E402

RES = OUT / "results"
K = 15
NRAND = 20
NPERM = 200
NBOOT = 1000


def jsd_matrix(p):
    n = len(p)
    J = np.zeros((n, n))
    for i in range(n - 1):
        J[i, i + 1:] = _jensen_shannon(np.broadcast_to(p[i], p[i + 1:].shape), p[i + 1:])
    return J + J.T


def edges(x):
    src, dst = knn_hop_graph(x, K).nonzero()
    u = src < dst
    return src[u], dst[u]


def coh_from_J(J, src, dst, w=None):
    """Unclipped coherence from a JSD matrix; optional cell multiplicity weights."""
    n = len(J)
    if w is None:
        de = J[src, dst].mean()
        da = J[np.triu_indices(n, 1)].mean()
    else:
        we = w[src] * w[dst]
        de = (we * J[src, dst]).sum() / we.sum()
        tot = w @ J @ w  # diagonal of J is 0
        den = w.sum() ** 2 - (w ** 2).sum()
        da = tot / den
    return 1.0 - de / da


def geary_moran(p, src, dst):
    n = len(p)
    pc = p - p.mean(0)
    ss = (pc ** 2).sum()
    d2 = ((p[src] - p[dst]) ** 2).sum(1)
    m = len(src)
    # symmetric weights, W = 2m
    geary_c = (n - 1) * 2 * d2.sum() / (2 * (2 * m) * ss)
    moran_i = n / (2 * m) * 2 * (pc[src] * pc[dst]).sum() / ss
    return 1.0 - geary_c, moran_i


def perplexity_fraction(p):
    return float(np.mean(np.exp(-np.sum(p * np.log(p), 1))) / p.shape[1])


def softmax_alloc(xs, W, beta):
    return to_simplex(beta * (xs @ W), already_simplex=False)


def match_beta(xs, W, target):
    lo, hi = 1e-3, 1e3
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        if perplexity_fraction(softmax_alloc(xs, W, mid)) > target:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


def job(args):
    idx, item = args
    a = np.load(PAPER / "evidence/scale-correspondence" / item["file"])
    x, p_nat = a["reference"], np.asarray(a["latent"], float)
    src, dst = edges(x)
    allocs = {}
    allocs["native"] = to_simplex(p_nat, already_simplex=True)
    gm = GaussianMixture(10, covariance_type="diag", random_state=0, n_init=3, reg_covar=1e-6).fit(x)
    allocs["gmm_diag50"] = to_simplex(gm.predict_proba(x), already_simplex=True)
    gm2 = GaussianMixture(10, covariance_type="full", random_state=0, n_init=3, reg_covar=1e-6).fit(x[:, :10])
    allocs["gmm_full10"] = to_simplex(gm2.predict_proba(x[:, :10]), already_simplex=True)
    xs = x / x.std()
    target = perplexity_fraction(allocs["native"])
    rng = np.random.default_rng(5000 + idx)
    for r in range(NRAND):
        W = rng.normal(scale=1 / np.sqrt(x.shape[1]), size=(x.shape[1], 10))
        allocs[f"random_matched_{r:02d}"] = softmax_alloc(xs, W, match_beta(xs, W, target))
        allocs[f"random_beta1_{r:02d}"] = softmax_alloc(xs, W, 1.0)
    rows, Js = [], {}
    for name, p in allocs.items():
        ac = allocation_coherence(x, p, k=K, already_simplex=True)
        J = jsd_matrix(p)
        Js[name] = J
        unc = coh_from_J(J, src, dst)
        assert abs(unc - (1 - ac["allocation_edge_js"] / ac["allocation_random_js"])) < 1e-9
        g, mi = geary_moran(p, src, dst)
        rows.append(dict(id=item["id"], dataset=item["dataset"], seed=item["seed"], allocation=name,
                         family=name.rsplit("_", 1)[0] if name.startswith("random") else name,
                         C_clipped=ac["allocation_coherence_raw"], C_unclipped=unc, status=ac["allocation_status"],
                         perplexity_fraction=ac["allocation_perplexity_fraction"], D_all=ac["allocation_random_js"],
                         one_minus_geary=g, moran_I=mi))
    # permutation null of the native allocation (recipient rows fixed)
    Jn = Js["native"]
    prng = np.random.default_rng(91000 + idx)
    null = []
    for _ in range(NPERM):
        q = prng.permutation(len(Jn))
        null.append(coh_from_J(Jn[np.ix_(q, q)], src, dst))
    null = np.asarray(null)
    nat = rows[0]
    perm = dict(id=item["id"], C_unclipped=nat["C_unclipped"], null_mean=null.mean(), null_sd=null.std(ddof=1),
                null_min=null.min(), null_max=null.max(), frac_null_clipped_to_zero=float((null < 0).mean()),
                z=(nat["C_unclipped"] - null.mean()) / null.std(ddof=1))
    # paired cell bootstrap
    brng = np.random.default_rng(77000 + idx)
    n = len(Jn)
    rand_m = [k for k in Js if k.startswith("random_matched")]
    rand_b = [k for k in Js if k.startswith("random_beta1")]
    boots = []
    for _ in range(NBOOT):
        w = np.bincount(brng.integers(0, n, n), minlength=n).astype(float)
        cn = coh_from_J(Jn, src, dst, w)
        boots.append(dict(native=cn,
                          d_gmm_diag50=cn - coh_from_J(Js["gmm_diag50"], src, dst, w),
                          d_gmm_full10=cn - coh_from_J(Js["gmm_full10"], src, dst, w),
                          d_random_matched=cn - np.mean([coh_from_J(Js[k], src, dst, w) for k in rand_m]),
                          d_random_beta1=cn - np.mean([coh_from_J(Js[k], src, dst, w) for k in rand_b])))
    b = pd.DataFrame(boots)
    point = {r["allocation"]: r["C_unclipped"] for r in rows}
    boot = dict(id=item["id"])
    for col, base in (("d_gmm_diag50", point["gmm_diag50"]), ("d_gmm_full10", point["gmm_full10"]),
                      ("d_random_matched", np.mean([point[k] for k in rand_m])),
                      ("d_random_beta1", np.mean([point[k] for k in rand_b]))):
        boot[col] = point["native"] - base
        boot[col + "_ci_lo"], boot[col + "_ci_hi"] = np.percentile(b[col], [2.5, 97.5])
    print(item["id"], "done", flush=True)
    return rows, perm, boot


def main():
    RES.mkdir(parents=True, exist_ok=True)
    meta = [m for m in json.loads((PAPER / "evidence/scale-correspondence/inputs.json").read_text()) if m["native"]]
    assert len(meta) == 12
    with Pool(12) as pool:
        res = pool.map(job, list(enumerate(meta)))
    rows = pd.DataFrame([r for rs, _, _ in res for r in rs])
    perm = pd.DataFrame([p for _, p, _ in res])
    boot = pd.DataFrame([b for _, _, b in res])
    rows.to_csv(RES / "a4_coherence_allocations.csv", index=False)
    perm.to_csv(RES / "a4_native_permutation_unclipped.csv", index=False)
    boot.to_csv(RES / "a4_bootstrap_differences.csv", index=False)
    fam = rows.groupby(["id", "family"])[["C_clipped", "C_unclipped", "perplexity_fraction", "one_minus_geary", "moran_I"]].mean().reset_index()
    fam.to_csv(RES / "a4_family_means.csv", index=False)
    piv = fam.pivot(index="id", columns="family", values="C_unclipped")
    summary = {
        "C_unclipped_by_family_median": piv.median().to_dict(),
        "C_unclipped_by_family_range": {c: [piv[c].min(), piv[c].max()] for c in piv},
        "n_native_exceeds_gmm_diag50": int((piv["native"] > piv["gmm_diag50"]).sum()),
        "n_native_exceeds_gmm_full10": int((piv["native"] > piv["gmm_full10"]).sum()),
        "n_native_exceeds_random_matched_mean": int((piv["native"] > piv["random_matched"]).sum()),
        "n_native_exceeds_random_beta1_mean": int((piv["native"] > piv["random_beta1"]).sum()),
        "bootstrap_CI_excludes_zero_positive": {c: int((boot[c + "_ci_lo"] > 0).sum()) for c in ("d_gmm_diag50", "d_gmm_full10", "d_random_matched", "d_random_beta1")},
        "bootstrap_CI_excludes_zero_negative": {c: int((boot[c + "_ci_hi"] < 0).sum()) for c in ("d_gmm_diag50", "d_gmm_full10", "d_random_matched", "d_random_beta1")},
        "native_perm_null_unclipped": {"mean_of_means": perm.null_mean.mean(), "max_null_max": perm.null_max.max(),
                                       "frac_null_below_zero_mean": perm.frac_null_clipped_to_zero.mean(),
                                       "z_range": [perm.z.min(), perm.z.max()]},
        "spearman_C_vs_one_minus_geary_native": float(pd.Series(fam[fam.family == "native"].C_unclipped.values).corr(
            pd.Series(fam[fam.family == "native"].one_minus_geary.values), method="spearman")),
        "perplexity_fraction_by_family_median": fam.pivot(index="id", columns="family", values="perplexity_fraction").median().to_dict(),
        "one_minus_geary_by_family_median": fam.pivot(index="id", columns="family", values="one_minus_geary").median().to_dict(),
        "moran_I_by_family_median": fam.pivot(index="id", columns="family", values="moran_I").median().to_dict(),
    }
    dump(RES / "a4_summary.json", summary)
    print(json.dumps(json.loads((RES / "a4_summary.json").read_text()), indent=1))
    print(piv.round(3).to_string())
    print(boot.round(3).to_string())


if __name__ == "__main__":
    main()
