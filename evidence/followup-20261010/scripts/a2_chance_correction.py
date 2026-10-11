#!/usr/bin/env python3
"""A2. Chance-corrected edge survival E* and separation S*, plus a tie-robust AUROC.

Null: rows of THAT candidate are permuted (reference fixed), 50 draws.
E* = (E - E0)/(1 - E0) with E0 the Monte Carlo permutation mean; the exact
permutation expectation (common.py docstring) is reported alongside.
AUROC = P(h_Z far pair > h_Z reference-edge pair) with mid-rank ties; its
permutation expectation is exactly 0.5.

Part 1 (24 scale objects x 9 (k,d)): every permuted score is a direct call of
the archived edge_survival / separation_survival on latent[perm].
Part 2 (diversity: 8 objects x 5 candidates x 1-2 references, k=15): the
observed E,S are archived-function calls; permutation draws use the relabelled
candidate hop matrix (identical when the candidate kNN graph has no cutoff
ties).  Two draws per combination are re-checked by direct archived calls.
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (OUT, PAPER, dump, e_s_from_hops, edge_survival, hops,  # noqa: E402
                    perm_null_from_hops, ref_sets, separation_survival)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

NPERM = 50
RES = OUT / "results"


def cutoff_tie_rows(x, k):
    d, _ = NearestNeighbors(n_neighbors=min(k + 2, len(x))).fit(x).kneighbors(x)
    return int(np.sum(d[:, k] == d[:, k + 1]))


# ---------------------------------------------------------------- part 1
def scale_job(args):
    idx, item = args
    a = np.load(PAPER / "evidence/scale-correspondence" / item["file"])
    x, z = a["reference"], a["latent"]
    rng = np.random.default_rng(20261010 + idx)
    perms = [rng.permutation(len(z)) for _ in range(NPERM)]
    rows = []
    for k in (5, 15, 30):
        zh = hops(z, k)
        for rank in (10, 25, 50):
            ref = x[:, :rank]
            e = edge_survival(ref, z, k=k)["edge_survival"]
            s = separation_survival(ref, z, k=k)["separation_survival"]
            rh = hops(ref, k)
            sets = ref_sets(rh)
            ex = e_s_from_hops(rh, zh, sets)
            assert abs(ex["E"] - e) < 1e-12 and abs(ex["S"] - s) < 1e-12
            pe, ps, pa = [], [], []
            for p in perms:
                zp = z[p]
                pe.append(edge_survival(ref, zp, k=k)["edge_survival"])
                ps.append(separation_survival(ref, zp, k=k)["separation_survival"])
                pa.append(e_s_from_hops(rh, zh[np.ix_(p, p)], sets)["AUROC_far_vs_edge"])
            pe, ps, pa = map(np.asarray, (pe, ps, pa))
            rows.append(dict(
                id=item["id"], dataset=item["dataset"], seed=item["seed"], native=item["native"], k=k, rank=rank, n=len(z),
                E=e, E0=pe.mean(), E0_sd=pe.std(ddof=1), E0_exact=ex["E0_exact"], E_star=(e - pe.mean()) / (1 - pe.mean()),
                E_z=(e - pe.mean()) / pe.std(ddof=1),
                S=s, S0=ps.mean(), S0_sd=ps.std(ddof=1), S0_exact=ex["S0_exact"], S_star=(s - ps.mean()) / (1 - ps.mean()),
                S_perm_max=ps.max(), S_below_perm_mean=bool(s < ps.mean()),
                AUROC=ex["AUROC_far_vs_edge"], AUROC0=pa.mean(), AUROC0_sd=pa.std(ddof=1),
                S_threshold_hops=ex["S_threshold_hops"], n_far=ex["n_far"], n_edges=ex["n_edges"],
                cand_cutoff_tie_rows=cutoff_tie_rows(z, k),
            ))
    print(item["id"], "done", flush=True)
    return rows


# ---------------------------------------------------------------- part 2
def diversity_job(path):
    a = np.load(path)
    refs = {"expression": a["pca"]}
    if "tissue" in a.files:
        refs["tissue"] = a["tissue"]
    cand_names = ("pca", "scvi", "umap", "random", "permuted_pca")
    cand_h = {c: hops(a[c], 15) for c in cand_names}
    rows, checks = [], []
    for ri, (refname, ref) in enumerate(refs.items()):
        rh = hops(ref, 15)
        sets = ref_sets(rh)
        for ci, c in enumerate(cand_names):
            z = a[c]
            e = edge_survival(ref, z, k=15)["edge_survival"]
            s = separation_survival(ref, z, k=15)["separation_survival"]
            ex = e_s_from_hops(rh, cand_h[c], sets)
            assert abs(ex["E"] - e) < 1e-12 and abs(ex["S"] - s) < 1e-12, (path.stem, refname, c)
            seed = 777 + 10 * ri + ci
            null = perm_null_from_hops(rh, cand_h[c], NPERM, seed, sets)
            # direct-call validation for the first two draws of the same stream
            rng = np.random.default_rng(seed)
            for b in range(2):
                p = rng.permutation(len(z))
                de = edge_survival(ref, z[p], k=15)["edge_survival"]
                ds = separation_survival(ref, z[p], k=15)["separation_survival"]
                checks.append(dict(object=path.stem, reference=refname, candidate=c, draw=b,
                                   dE=de - null["E"][b], dS=ds - null["S"][b]))
            E0, S0 = null["E"].mean(), null["S"].mean()
            rows.append(dict(
                object=path.stem, reference=refname, candidate=c, n=len(z), E=e, E0=E0, E0_sd=null["E"].std(ddof=1),
                E0_exact=ex["E0_exact"], E_star=(e - E0) / (1 - E0), E_z=(e - E0) / null["E"].std(ddof=1),
                E_perm_max=null["E"].max(),
                S=s, S0=S0, S0_sd=null["S"].std(ddof=1), S0_exact=ex["S0_exact"], S_star=(s - S0) / (1 - S0),
                S_perm_min=null["S"].min(), S_perm_max=null["S"].max(),
                AUROC=ex["AUROC_far_vs_edge"], AUROC0=null["AUROC_far_vs_edge"].mean(),
                AUROC0_sd=null["AUROC_far_vs_edge"].std(ddof=1),
                S_threshold_hops=ex["S_threshold_hops"], n_far=ex["n_far"], n_edges=ex["n_edges"],
                cand_cutoff_tie_rows=cutoff_tie_rows(z, 15), ref_cutoff_tie_rows=cutoff_tie_rows(ref, 15),
            ))
    print(path.stem, "done", flush=True)
    return rows, checks


def tau(a, b):
    t = kendalltau(a, b).statistic
    return float(t) if np.isfinite(t) else float("nan")


def summarise(sc: pd.DataFrame, dv: pd.DataFrame, checks: pd.DataFrame) -> dict:
    out = {}
    # ---------- (i) k-range explained by the null shift (24 objects)
    rng_rows = []
    for oid, g in sc.groupby("id"):
        r = dict(id=oid, native=bool(g.native.iloc[0]),
                 E_range=g.E.max() - g.E.min(), E0_range=g.E0.max() - g.E0.min(),
                 Estar_range=g.E_star.max() - g.E_star.min(),
                 S_range=g.S.max() - g.S.min(), Sstar_range=g.S_star.max() - g.S_star.min(),
                 AUROC_range=g.AUROC.max() - g.AUROC.min())
        for rank in (10, 25, 50):
            h = g[g["rank"] == rank].set_index("k")
            dE = h.loc[30, "E"] - h.loc[5, "E"]
            dE0 = h.loc[30, "E0"] - h.loc[5, "E0"]
            r[f"dE_k5_to_k30_d{rank}"] = dE
            r[f"dE0_k5_to_k30_d{rank}"] = dE0
            r[f"frac_null_d{rank}"] = dE0 / dE if dE != 0 else np.nan
            r[f"dEstar_k5_to_k30_d{rank}"] = h.loc[30, "E_star"] - h.loc[5, "E_star"]
        rng_rows.append(r)
    rr = pd.DataFrame(rng_rows)
    rr.to_csv(RES / "a2_scale24_ranges.csv", index=False)
    out["scale24"] = {
        "median_within_object_E_range": rr.E_range.median(),
        "median_within_object_E0_range": rr.E0_range.median(),
        "median_within_object_Estar_range": rr.Estar_range.median(),
        "median_within_object_S_range": rr.S_range.median(),
        "median_within_object_Sstar_range": rr.Sstar_range.median(),
        "median_within_object_AUROC_range": rr.AUROC_range.median(),
        "frac_of_dE_k5_to_k30_from_null_d50": {"median": rr.frac_null_d50.median(), "min": rr.frac_null_d50.min(), "max": rr.frac_null_d50.max()},
        "frac_of_dE_k5_to_k30_from_null_all_d": {"median": float(np.median(rr[["frac_null_d10", "frac_null_d25", "frac_null_d50"]].values)),
                                                 "iqr": np.percentile(rr[["frac_null_d10", "frac_null_d25", "frac_null_d50"]].values, [25, 75])},
        "n_objects_Estar_increases_k5_to_k30_d50": int((rr.dEstar_k5_to_k30_d50 > 0).sum()),
        "median_dE_k5_to_k30_d50": rr.dE_k5_to_k30_d50.median(),
        "median_dEstar_k5_to_k30_d50": rr.dEstar_k5_to_k30_d50.median(),
        "E0_mc_vs_exact_max_abs_diff": float((sc.E0 - sc.E0_exact).abs().max()),
        "S0_mc_vs_exact_max_abs_diff": float((sc.S0 - sc.S0_exact).abs().max()),
        "S0_range_over_all_rows": [sc.S0.min(), sc.S0.max()],
        "E0_range_over_all_rows": [sc.E0.min(), sc.E0.max()],
        "AUROC0_mean_range": [sc.AUROC0.min(), sc.AUROC0.max()],
        "n_rows_S_below_own_perm_mean": int(sc.S_below_perm_mean.sum()),
        "min_E_z": sc.E_z.min(),
        "cand_cutoff_tie_rows_total": int(sc.cand_cutoff_tie_rows.sum()),
    }
    # native-minus-continuous paired sign before/after correction
    pair = []
    for (ds, sd, k, rank), g in sc.groupby(["dataset", "seed", "k", "rank"]):
        nat = g[g.native].iloc[0]
        con = g[~g.native].iloc[0]
        pair.append(dict(dataset=ds, seed=sd, k=k, rank=rank,
                         dE=nat.E - con.E, dEstar=nat.E_star - con.E_star,
                         dS=nat.S - con.S, dSstar=nat.S_star - con.S_star, dAUROC=nat.AUROC - con.AUROC))
    pdf = pd.DataFrame(pair)
    pdf.to_csv(RES / "a2_scale24_native_minus_continuous.csv", index=False)
    out["scale24_native_minus_continuous"] = {
        "n": len(pdf),
        "dE_negative": int((pdf.dE < 0).sum()), "dEstar_negative": int((pdf.dEstar < 0).sum()),
        "sign_changes_E_vs_Estar": int((np.sign(pdf.dE) != np.sign(pdf.dEstar)).sum()),
        "dS_negative": int((pdf.dS < 0).sum()), "dSstar_negative": int((pdf.dSstar < 0).sum()),
        "dAUROC_negative": int((pdf.dAUROC < 0).sum()),
    }
    taus = []
    for (k, rank), g in sc.groupby(["k", "rank"]):
        taus.append(dict(k=k, rank=rank, tau_E_Estar=tau(g.E, g.E_star), tau_S_Sstar=tau(g.S, g.S_star),
                         tau_S_AUROC=tau(g.S, g.AUROC), tau_Estar_AUROC=tau(g.E_star, g.AUROC)))
    out["scale24_object_ranking_tau_per_setting"] = taus

    # ---------- (ii) UMAP vs its own permutation null; (iii) rankings (diversity)
    dv = dv.copy()
    pca_perm = dv[dv.candidate == "permuted_pca"].set_index(["object", "reference"])
    umap = []
    for _, r in dv[dv.candidate == "umap"].iterrows():
        pp = pca_perm.loc[(r.object, r.reference)]
        umap.append(dict(object=r.object, reference=r.reference, E_umap=r.E, E_permuted_pca=pp.E,
                         below_pca_permutation=bool(r.E < pp.E),
                         E0_umap_own=r.E0, E0_umap_own_sd=r.E0_sd, E_star_umap=r.E_star, E_z_umap=r.E_z,
                         below_own_null=bool(r.E <= r.E_perm_max)))
    um = pd.DataFrame(umap)
    um.to_csv(RES / "a2_diversity_umap_null.csv", index=False)
    out["diversity_umap"] = {
        "tissue_ref_below_pca_permutation": f"{int(um[um.reference=='tissue'].below_pca_permutation.sum())}/{int((um.reference=='tissue').sum())}",
        "tissue_ref_at_or_below_own_null_max": f"{int(um[um.reference=='tissue'].below_own_null.sum())}/{int((um.reference=='tissue').sum())}",
        "tissue_ref_min_E_star_umap": float(um[um.reference == 'tissue'].E_star_umap.min()),
        "tissue_ref_min_z_umap": float(um[um.reference == 'tissue'].E_z_umap.min()),
        "rows": um.to_dict(orient="records"),
    }
    rank_rows = []
    for (o, ref), g in dv.groupby(["object", "reference"]):
        keep = ["pca", "scvi", "umap", "random"] if ref == "tissue" else ["scvi", "umap", "random"]
        h = g.set_index("candidate").loc[keep]
        rank_rows.append(dict(object=o, reference=ref, candidates=",".join(keep),
                              tau_E_Estar=tau(h.E, h.E_star), tau_S_Sstar=tau(h.S, h.S_star),
                              tau_S_AUROC=tau(h.S, h.AUROC), tau_Estar_AUROC=tau(h.E_star, h.AUROC),
                              order_E=",".join(h.E.sort_values(ascending=False).index),
                              order_Estar=",".join(h.E_star.sort_values(ascending=False).index),
                              order_S=",".join(h.S.sort_values(ascending=False).index),
                              order_Sstar=",".join(h.S_star.sort_values(ascending=False).index),
                              order_AUROC=",".join(h.AUROC.sort_values(ascending=False).index)))
    rk = pd.DataFrame(rank_rows)
    rk.to_csv(RES / "a2_diversity_rankings.csv", index=False)
    out["diversity_rankings"] = {
        "mean_tau_E_Estar_tissue": rk[rk.reference == "tissue"].tau_E_Estar.mean(),
        "mean_tau_E_Estar_expression": rk[rk.reference == "expression"].tau_E_Estar.mean(),
        "n_order_changes_E_vs_Estar": int((rk.order_E != rk.order_Estar).sum()), "n_combos": len(rk),
        "mean_tau_S_Sstar": rk.tau_S_Sstar.mean(), "n_order_changes_S_vs_Sstar": int((rk.order_S != rk.order_Sstar).sum()),
        "mean_tau_S_AUROC": rk.tau_S_AUROC.mean(), "n_order_changes_S_vs_AUROC": int((rk.order_S != rk.order_AUROC).sum()),
    }
    # pooled ranking across all (object, reference, candidate)
    nonid = dv[~((dv.reference == "expression") & (dv.candidate == "pca"))]
    out["diversity_pooled_tau"] = {"tau_E_Estar": tau(nonid.E, nonid.E_star), "tau_S_Sstar": tau(nonid.S, nonid.S_star),
                                  "tau_S_AUROC": tau(nonid.S, nonid.AUROC), "n": len(nonid)}
    out["diversity_null_ranges"] = {
        "E0_by_candidate_tissue": dv[dv.reference == "tissue"].groupby("candidate").E0.agg(["min", "max"]).to_dict(orient="index"),
        "S0_by_candidate_tissue": dv[dv.reference == "tissue"].groupby("candidate").S0.agg(["min", "max"]).to_dict(orient="index"),
        "E0_mc_vs_exact_max_abs_diff": float((dv.E0 - dv.E0_exact).abs().max()),
        "S0_mc_vs_exact_max_abs_diff": float((dv.S0 - dv.S0_exact).abs().max()),
        "AUROC0_range": [dv.AUROC0.min(), dv.AUROC0.max()],
    }
    out["diversity_S_below_own_perm_mean"] = dv[dv.S < dv.S0][["object", "reference", "candidate", "S", "S0"]].to_dict(orient="records")
    out["hop_relabel_vs_direct_call_max_abs_diff"] = {"E": float(checks.dE.abs().max()), "S": float(checks.dS.abs().max()),
                                                     "n_checked": len(checks)}
    out["diversity_cutoff_ties"] = dv.groupby(["object", "reference", "candidate"]).cand_cutoff_tie_rows.first()[lambda s: s > 0].to_dict()
    out["diversity_ref_cutoff_ties"] = dv.groupby(["object", "reference"]).ref_cutoff_tie_rows.first().to_dict()
    return out


def main():
    RES.mkdir(parents=True, exist_ok=True)
    meta = json.loads((PAPER / "evidence/scale-correspondence/inputs.json").read_text())
    sc_path = RES / "a2_scale24.csv"
    dv_path = RES / "a2_diversity.csv"
    ck_path = RES / "a2_diversity_relabel_checks.csv"
    if not sc_path.exists():
        with Pool(20) as pool:
            rows = [r for chunk in pool.map(scale_job, list(enumerate(meta))) for r in chunk]
        pd.DataFrame(rows).to_csv(sc_path, index=False)
    if not dv_path.exists():
        paths = sorted((OUT / "arrays" / "diversity").glob("*.npz"))
        with Pool(8) as pool:
            res = pool.map(diversity_job, paths)
        pd.DataFrame([r for rows, _ in res for r in rows]).to_csv(dv_path, index=False)
        pd.DataFrame([c for _, cs in res for c in cs]).to_csv(ck_path, index=False)
    summary = summarise(pd.read_csv(sc_path), pd.read_csv(dv_path), pd.read_csv(ck_path))
    dump(RES / "a2_summary.json", summary)
    print(json.dumps(json.loads((RES / "a2_summary.json").read_text()), indent=1)[:6000])


if __name__ == "__main__":
    main()
