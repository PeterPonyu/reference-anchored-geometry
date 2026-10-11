#!/usr/bin/env python3
"""A3. Full stored comparator panel and a graded multi-topology calibration.

(a) Every scalar stored in evidence/measurement-inventory/calibration.json is
    extracted for each seed x intervention (no recomputation).
(b) Graded calibration.  Generators: the repository Y fixture
    (audit.upstream_validation.make_reference_manifold / make_candidates,
    imported unchanged).  Candidate noise is DECOUPLED from reference jitter by
    calling make_candidates with seed+100000 ("Y"); the archived coupled
    construction (same seed) is run as "Y_coupled".  New topologies use the
    same noise levels (reference 0.004, candidate 0.01) and a comparable total
    curve length (~1.98): "cycle" (circle, upper half shifted +a in y: two
    gaps), "branch3" (trunk + three arms; the arms shifted apart by a: four
    pieces), "clusters" (three disjoint Gaussian blobs, a genuinely fragmented
    reference; intervention = merging, centroid distance scaled by 1-u).
    Shift grid: 0, geomspace(0.002, 0.5, 12), 2.5, 9.  k in {5,15,30},
    n_per_piece giving n ~ 450 or ~2000, 20 seeds.
RAAG scores: archived reachability/edge/separation functions are called for
every n~450 case and for seeds 0-2 at n~2000; otherwise the identical
quantities are computed from archived knn_hop_graph/_hop_distances hop
matrices (equality asserted wherever both exist).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (OUT, PAPER, UPSTREAM, clean, components, dump, e_s_from_hops, edge_survival,  # noqa: E402
                    hops, partition_pair_scores, reachability_survival, ref_sets, separation_survival)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(UPSTREAM))
from audit.reference_metrics import (_continuity, coranking_matrix, coranking_metrics,  # noqa: E402
                                     graph_geodesic_metrics)
from audit.upstream_validation import make_candidates, make_reference_manifold  # noqa: E402

RES = OUT / "results"
SHIFTS = [0.0] + [float(v) for v in np.geomspace(0.002, 0.5, 12)] + [2.5, 9.0]
MERGE = [0.0, 0.5, 0.8, 0.9, 0.95, 0.97, 0.99, 1.0]
KS = (5, 15, 30)
SEEDS = list(range(20))
NSIZES = {"450": 150, "2000": 667}  # per-piece count for Y; others scale to the same total


# ------------------------------------------------------------- stored panel (a)
def stored_panel():
    cal = json.loads((PAPER / "evidence/measurement-inventory/calibration.json").read_text())
    rows = []
    for r in cal["rows"]:
        for stack, payload in r["stacks"].items():
            for key, val in (payload.get("values") or {}).items():
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    rows.append(dict(seed=r["seed"], candidate=r["candidate"], stack=stack, metric=key, value=val))
    long = pd.DataFrame(rows)
    long.to_csv(RES / "a3_stored_comparators_long.csv", index=False)
    wide = long.pivot_table(index=["seed", "candidate"], columns="metric", values="value").reset_index()
    wide.to_csv(RES / "a3_stored_comparators_wide.csv", index=False)
    return wide


# ------------------------------------------------------------- generators (b)
def y_pair(n_piece, seed, a, coupled):
    ref = make_reference_manifold(n_piece, seed)
    cand = make_candidates(ref, seed if coupled else seed + 100_000, shatter_amplitude=a)["shatter"]["embedding"]
    return ref["coords"], cand


def _arc_pieces(seed, n_total):
    return np.random.default_rng(seed), np.random.default_rng(seed + 100_000)


def cycle_pair(n_total, seed, a):
    rr, rc = _arc_pieces(seed, n_total)
    r = 1.98 / (2 * np.pi)
    th = np.linspace(0, 2 * np.pi, n_total, endpoint=False) + np.pi / n_total  # no point exactly on y=0
    clean_xy = np.c_[r * np.cos(th), r * np.sin(th)]
    ref = clean_xy + rr.normal(scale=0.004, size=clean_xy.shape)
    cand = ref + rc.normal(scale=0.01, size=clean_xy.shape)
    cand[clean_xy[:, 1] > 0] += [0.0, a]
    return ref, cand


def branch3_pair(n_piece, seed, a):
    rr, rc = _arc_pieces(seed, 4 * n_piece)
    t = np.linspace(0, 0.5, n_piece, endpoint=False)
    s = np.linspace(0, 0.74, n_piece + 1)[1:]
    pieces = [np.c_[t, np.zeros_like(t)]]
    lab = [np.zeros(n_piece, int)]
    for j, ang in enumerate((np.deg2rad(40), 0.0, -np.deg2rad(40))):
        pieces.append(np.c_[0.5 + s * np.cos(ang), s * np.sin(ang)])
        lab.append(np.full(n_piece, j + 1))
    clean_xy, lab = np.vstack(pieces), np.concatenate(lab)
    ref = clean_xy + rr.normal(scale=0.004, size=clean_xy.shape)
    cand = ref + rc.normal(scale=0.01, size=clean_xy.shape)
    cand[lab == 1] += [0.0, a]
    cand[lab == 2] += [a, 0.0]
    cand[lab == 3] += [0.0, -a]
    return ref, cand


def clusters_pair(n_piece, seed, u):
    rr, rc = _arc_pieces(seed, 3 * n_piece)
    centers = 3.0 * np.array([[0.0, 1.0], [np.cos(np.deg2rad(210)), np.sin(np.deg2rad(210))],
                              [np.cos(np.deg2rad(330)), np.sin(np.deg2rad(330))]]) / np.sqrt(3)
    lab = np.repeat(np.arange(3), n_piece)
    local = rr.normal(scale=0.1, size=(3 * n_piece, 2))
    ref = centers[lab] + local
    cand_local = local + rc.normal(scale=0.01, size=local.shape)
    cand = (1.0 - u) * centers[lab] + cand_local
    return ref, cand


def make_pair(topo, nkey, seed, a):
    npc = NSIZES[nkey]
    if topo == "Y":
        return y_pair(npc, seed, a, coupled=False)
    if topo == "Y_coupled":
        return y_pair(npc, seed, a, coupled=True)
    if topo == "cycle":
        return cycle_pair(3 * npc, seed, a)
    if topo == "branch3":
        return branch3_pair(int(round(3 * npc / 4)), seed, a)
    if topo == "clusters":
        return clusters_pair(npc, seed, a)
    raise KeyError(topo)


# ------------------------------------------------------------- scoring
def lcmc_from_q(q, k):
    n = q.shape[0] + 1
    return float(q[:k, :k].sum() / (k * n) - k / (n - 1.0))


def score_job(args):
    topo, nkey, seed, a = args
    from scipy.stats import spearmanr
    from sklearn.manifold import trustworthiness
    from sklearn.metrics import pairwise_distances
    from zadu.measures import steadiness_cohesiveness as sc_mod

    t0 = time.time()
    ref, cand = make_pair(topo, nkey, seed, a)
    n = len(ref)
    tri = np.triu_indices(n, 1)
    base = dict(topology=topo, n_key=nkey, n=n, seed=seed, amplitude=a)
    # k-independent comparators
    co = coranking_metrics(ref, cand)
    q = coranking_matrix(ref, cand)
    dsp = float(spearmanr(pairwise_distances(ref)[tri], pairwise_distances(cand)[tri]).statistic)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stc = sc_mod.measure(ref, cand, random_state=seed, n_jobs=1)
    glob = dict(distance_spearman=dsp, Q_global=co["q_global"], Q_local=co["q_local"], AUC_RNX=co["auc_ln_rnx"],
                steadiness=float(stc["steadiness"]), cohesiveness=float(stc["cohesiveness"]))
    rows = []
    for k in KS:
        rh, ch = hops(ref, k), hops(cand, k)
        sets = ref_sets(rh)
        ex = e_s_from_hops(rh, ch, sets)
        rfin = np.isfinite(rh[tri])
        R = float(np.isfinite(ch[tri])[rfin].mean())
        direct = (nkey == "450") or seed < 3
        if direct:
            rd = reachability_survival(ref, cand, k=k)["reachability_survival"]
            ed = edge_survival(ref, cand, k=k)["edge_survival"]
            sd = separation_survival(ref, cand, k=k)["separation_survival"]
            assert abs(rd - R) < 1e-12 and abs(ed - ex["E"]) < 1e-12 and abs(sd - ex["S"]) < 1e-12
        part = partition_pair_scores(components(ref, k), components(cand, k))
        geo = graph_geodesic_metrics(ref, cand, k=k)
        rows.append(dict(**base, k=k, R=R, E=ex["E"], E0=ex["E0_exact"], E_star=ex["E_star"], S=ex["S"],
                         S0=ex["S0_exact"], S_star=ex["S_star"], AUROC=ex["AUROC_far_vs_edge"],
                         pair_precision=part["pair_precision"], pair_f1=part["pair_f1"], component_ari=part["component_ari"],
                         n_cand_components=part["n_cand_components"], n_ref_components=part["n_ref_components"],
                         cand_lcc=geo["candidate_lcc_fraction"], geodesic_spearman=geo["geodesic_pairwise_spearman"],
                         trustworthiness=float(trustworthiness(ref, cand, n_neighbors=k)),
                         continuity=_continuity(ref, cand, k), LCMC=lcmc_from_q(q, k),
                         direct_archived_call=direct, **glob))
    rows[-1]["seconds"] = time.time() - t0
    return rows


def jobs():
    out = []
    for topo in ("Y", "Y_coupled", "cycle", "branch3", "clusters"):
        grid = MERGE if topo == "clusters" else SHIFTS
        for nkey in NSIZES:
            for seed in SEEDS:
                for a in grid:
                    out.append((topo, nkey, seed, a))
    # big jobs first for load balance
    out.sort(key=lambda j: j[1] != "2000")
    return out


# ------------------------------------------------------------- response summary
METRICS = ["R", "E", "E_star", "S", "S_star", "AUROC", "pair_precision", "component_ari", "n_cand_components",
           "cand_lcc", "geodesic_spearman", "trustworthiness", "continuity", "LCMC", "distance_spearman",
           "Q_global", "Q_local", "AUC_RNX", "steadiness", "cohesiveness"]


def interp_cross(amps, r, level=0.5):
    """First amplitude (log-linear interpolation; 0 handled linearly) where r >= level."""
    for i in range(1, len(amps)):
        if r[i] >= level:
            a0, a1, r0, r1 = amps[i - 1], amps[i], r[i - 1], r[i]
            if r1 == r0:
                return a1
            f = (level - r0) / (r1 - r0)
            if a0 <= 0:
                return a0 + f * (a1 - a0)
            return float(np.exp(np.log(a0) + f * (np.log(a1) - np.log(a0))))
    return np.nan


def response_summary(df):
    recs = []
    for (topo, nkey, k), g in df.groupby(["topology", "n_key", "k"]):
        amps = sorted(g.amplitude.unique())
        amax = amps[-1]
        for m in METRICS:
            piv = g.pivot_table(index="seed", columns="amplitude", values=m)
            d = piv.sub(piv[0.0], axis=0)  # paired change from faithful
            full = d[amax]
            full_mean = full.mean()
            a50 = []
            for s in piv.index:
                if abs(full[s]) < 1e-12:
                    a50.append(np.nan)
                    continue
                r = (d.loc[s] / full[s]).to_numpy()
                a50.append(interp_cross(amps, r))
            a50 = np.asarray(a50, float)
            # detection: smallest amplitude whose 20-seed paired mean change exceeds 3 SD (of paired changes), same sign as full
            det = np.nan
            for a in amps[1:]:
                mu, sd = d[a].mean(), d[a].std(ddof=1)
                if np.sign(mu) == np.sign(full_mean) and full_mean != 0 and abs(mu) > 3 * max(sd, 1e-12) and abs(mu) > 1e-9:
                    det = a
                    break
            recs.append(dict(topology=topo, n_key=nkey, k=k, metric=m, faithful_mean=piv[0.0].mean(),
                             faithful_seed_sd=piv[0.0].std(ddof=1), full_change_mean=full_mean, full_change_sd=full.std(ddof=1),
                             A50_median=np.nanmedian(a50) if np.isfinite(a50).any() else np.nan,
                             A50_seed_sd=np.nanstd(a50, ddof=1) if np.isfinite(a50).sum() > 1 else np.nan,
                             A50_n_finite=int(np.isfinite(a50).sum()), A_detect_3sd=det))
    return pd.DataFrame(recs)


def neighbour_scale(df_unused=None):
    """Median reference k-th neighbour distance per topology/n/k (amplitude scale)."""
    from sklearn.neighbors import NearestNeighbors
    out = []
    for topo in ("Y", "cycle", "branch3", "clusters"):
        for nkey in NSIZES:
            ref, _ = make_pair(topo, nkey, 0, 0.0)
            for k in KS:
                d, _ = NearestNeighbors(n_neighbors=k + 1).fit(ref).kneighbors(ref)
                out.append(dict(topology=topo, n_key=nkey, k=k, median_kth_nn_distance=float(np.median(d[:, k]))))
    return pd.DataFrame(out)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    stored_panel()
    raw = RES / "a3_graded_raw.csv"
    if not raw.exists():
        t = time.time()
        with Pool(22) as pool:
            rows = [r for chunk in pool.imap_unordered(score_job, jobs(), chunksize=1) for r in chunk]
        df = pd.DataFrame(rows)
        df.to_csv(raw, index=False)
        print("graded grid", len(df), "rows", f"{time.time()-t:.0f}s", flush=True)
    df = pd.read_csv(raw)
    summ = response_summary(df)
    summ.to_csv(RES / "a3_graded_response_summary.csv", index=False)
    neighbour_scale().to_csv(RES / "a3_neighbour_scale.csv", index=False)
    print(summ[(summ.k == 15)].round(4).to_string())


if __name__ == "__main__":
    main()
