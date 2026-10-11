"""Sensitivity of tissue-reference scores to exact distance ties in the reference graph.

Visium spot positions are integer pixel coordinates on a lattice, so the k-th and
(k+1)-th nearest tissue neighbours are exactly equidistant for a share of spots.
The archived neighbour constructor then chooses among the tied spots by array position.
Candidate graphs (PCA, scVI, UMAP, random projection, permuted PCA) have no cutoff ties.

For every section the five candidates are rescored against 62 reference graphs:

  A0     archived constructor, saved row order (must reproduce the saved scores exactly)
  A      archived constructor after 30 random reorderings of the reference rows; edges are mapped back to
         the original cell identities (candidate graphs stay in the saved order and have no cutoff ties)
  B      30 random breaks of every cutoff tie group (explicit self removal)
  C      one tie-complete graph that keeps every strictly nearer neighbour and every neighbour at the
         cutoff distance, so out-degree can exceed k

Run from any directory:  python tie_sensitivity.py            (full run, about 7 minutes)
                         python tie_sensitivity.py --summarise (summaries from the saved raw table)
Inputs are read from ../../followup-20261010/arrays/diversity; outputs go to ../results.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[2]
sys.path.insert(0, str(PAPER / "scripts"))
from local_raag import _hop_distances, knn_hop_graph  # noqa: E402  (archived scorer, imported unchanged)

K = 15
NPERM = 30
NRAND = 30
ARRAYS = PAPER / "evidence/followup-20261010/arrays/diversity"
SAVED = PAPER / "evidence/followup-20261010/results/a2_diversity.csv"
RESULTS = HERE.parent / "results"
CANDIDATES = ("pca", "scvi", "umap", "random", "permuted_pca")
SECTIONS = ("bulb", "pfc_151673", "metastasis", "breast", "ovary", "lymph", "pfc_151507")
FOUR = ("pca", "scvi", "umap", "random")


# ---------------------------------------------------------------- score helpers
# Same definitions as scripts/local_raag.py and evidence/followup-20261010/scripts/common.py.
def reference_sets(ref_h, far_quantile=0.9):
    tri = np.triu_indices(ref_h.shape[0], k=1)
    rh = ref_h[tri]
    finite = np.isfinite(rh)
    cutoff = float(np.quantile(rh[finite], far_quantile))
    return tri, rh == 1, finite & (rh >= cutoff), cutoff


def chance_correct(value, null):
    return float((value - null) / (1.0 - null)) if null < 1.0 else float("nan")


def auroc_far_near(pair_hops, far, near):
    a, b = pair_hops[far], pair_hops[near]
    finite = np.isfinite(pair_hops)
    big = np.max(pair_hops[finite]) + 1 if finite.any() else 1.0
    a, b = np.where(np.isfinite(a), a, big), np.where(np.isfinite(b), b, big)
    values = np.concatenate([a, b])
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    midrank = (np.cumsum(counts) - (counts - 1) / 2.0)[inverse]
    return float((midrank[: len(a)].sum() - len(a) * (len(a) + 1) / 2.0) / (len(a) * len(b)))


def pair_count(labels):
    _, c = np.unique(labels, return_counts=True)
    return int(np.sum(c * (c - 1) // 2))


def score(ref_h, cand_h, sets):
    tri, edge, far, cutoff = sets
    ch = cand_h[tri]
    inv = np.zeros_like(ch)
    finite = np.isfinite(ch)
    inv[finite] = 1.0 / np.maximum(ch[finite], 1.0)
    median = float(np.median(ch[finite])) if finite.any() else np.inf
    separated = (~finite) | (ch >= median)
    E, E0 = float(inv[edge].mean()), float(inv.mean())
    S, S0 = float(separated[far].mean()), float(separated.mean())
    return dict(E=E, E0_exact=E0, E_star=chance_correct(E, E0), S=S, S0_exact=S0,
                S_star=chance_correct(S, S0), AUROC=auroc_far_near(ch, far, edge),
                n_far=int(far.sum()), n_edges=int(edge.sum()))


def reachability(ref_labels, cand_labels):
    joint = ref_labels.astype(np.int64) * (cand_labels.max() + 1) + cand_labels
    return pair_count(joint) / pair_count(ref_labels)


# ------------------------------------------------------------- reference graphs
def adjacency(n, rows, cols):
    g = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    return g.maximum(g.T)


def tie_structure(x, k, extra=40):
    n = len(x)
    dist, index = cKDTree(x).query(x, k=min(n, k + 1 + extra))
    out = []
    for i in range(n):
        keep = index[i] != i
        d, j = dist[i][keep], index[i][keep]
        cut = d[k - 1]
        if d[-1] <= cut:
            raise RuntimeError("tie group exceeds the query window")
        out.append((j[d < cut], j[d == cut], k - int((d < cut).sum())))
    return out


def graph_from_structure(n, structure, mode, rng=None):
    rows, cols = [], []
    for i, (closer, tied, need) in enumerate(structure):
        if mode == "complete" or len(tied) == need:
            chosen = np.concatenate([closer, tied])
        else:
            chosen = np.concatenate([closer, rng.choice(tied, size=need, replace=False)])
        rows.append(np.full(len(chosen), i))
        cols.append(chosen)
    return adjacency(n, np.concatenate(rows), np.concatenate(cols))


def archived_reordered_graph(x, k, order):
    g = knn_hop_graph(x[order], k).tocoo()
    return adjacency(len(x), order[g.row], order[g.col])


def score_graph(ref_graph, cand_hops, cand_labels):
    ref_h = _hop_distances(ref_graph)
    sets = reference_sets(ref_h)
    _, ref_labels = connected_components(ref_graph, directed=False)
    out = {}
    for name in CANDIDATES:
        s = score(ref_h, cand_hops[name], sets)
        s["R"] = reachability(ref_labels, cand_labels[name])
        out[name] = s
    return out


def run_section(name):
    t0 = time.time()
    z = np.load(ARRAYS / f"{name}.npz", allow_pickle=True)
    ref = z["tissue"].astype(float)
    n = len(ref)
    cand_hops, cand_labels = {}, {}
    for c in CANDIDATES:
        g = knn_hop_graph(z[c], K)
        cand_hops[c] = _hop_distances(g)
        _, cand_labels[c] = connected_components(g, directed=False)
    structure = tie_structure(ref, K)
    tie_rows = int(sum(len(t) > need for _, t, need in structure))
    rows = []

    def add(variant, rid, scores):
        for c, s in scores.items():
            rows.append(dict(object=name, variant=variant, realization=rid, candidate=c, n=n, **s))

    add("A0_archived_saved_order", 0, score_graph(knn_hop_graph(ref, K), cand_hops, cand_labels))
    rng = np.random.default_rng(20261011 + sum(map(ord, name)))
    for r in range(NPERM):
        add("A_archived_row_reordered", r + 1,
            score_graph(archived_reordered_graph(ref, K, rng.permutation(n)), cand_hops, cand_labels))
    for r in range(NRAND):
        add("B_random_tie_break", r + 1,
            score_graph(graph_from_structure(n, structure, "random", rng), cand_hops, cand_labels))
    add("C_tie_complete", 0, score_graph(graph_from_structure(n, structure, "complete"), cand_hops, cand_labels))
    meta = dict(object=name, n=n, cutoff_tie_rows=tie_rows, cutoff_tie_fraction=round(tie_rows / n, 4),
                archived_edges=int(knn_hop_graph(ref, K).nnz // 2),
                tie_complete_edges=int(graph_from_structure(n, structure, "complete").nnz // 2),
                seconds=round(time.time() - t0, 1))
    return rows, meta


# ------------------------------------------------------------------- summaries
def summarise(raw):
    saved = pd.read_csv(SAVED)
    saved = saved[saved.reference == "tissue"]
    base = raw[raw.variant == "A0_archived_saved_order"].merge(
        saved[["object", "candidate", "E", "S", "E0_exact", "S0_exact", "AUROC", "n_edges", "n_far"]],
        on=["object", "candidate"], suffixes=("", "_saved"))
    reproduction = {c: float((base[c] - base[c + "_saved"]).abs().max())
                    for c in ("E", "S", "E0_exact", "S0_exact", "AUROC", "n_edges", "n_far")}
    metrics = ["E", "E_star", "S", "S_star", "R", "AUROC"]
    rows = []
    for (obj, cand), g in raw.groupby(["object", "candidate"]):
        b = g[g.variant == "A0_archived_saved_order"].iloc[0]
        c = g[g.variant == "C_tie_complete"].iloc[0]
        for m in metrics:
            x = g[m].values
            rows.append(dict(object=obj, candidate=cand, metric=m, saved_order=b[m], mean=x.mean(),
                             sd=x.std(ddof=1), minimum=x.min(), maximum=x.max(),
                             range=x.max() - x.min(), tie_complete=c[m]))
    summary = pd.DataFrame(rows)

    raw = raw.copy()
    raw["world"] = raw.variant + "#" + raw.realization.astype(str)
    secs = sorted(raw.object.unique())

    def order(g, col):
        d = g.set_index("candidate")[col]
        return list(d.loc[list(FOUR)].sort_values(ascending=False).index)

    def one_world(g):
        pick = lambda s, c, m: float(g[(g.object == s) & (g.candidate == c)][m].iloc[0])  # noqa: E731
        return dict(
            umap_above_random_Estar=all(pick(s, "umap", "E_star") > pick(s, "random", "E_star") for s in secs),
            scvi_not_below_pca_Estar=all(pick(s, "scvi", "E_star") >= pick(s, "pca", "E_star") for s in secs),
            umap_Estar_positive=all(pick(s, "umap", "E_star") > 0 for s in secs),
            umap_below_scvi_Estar=sum(pick(s, "umap", "E_star") < pick(s, "scvi", "E_star") for s in secs),
            umap_below_pca_Estar=sum(pick(s, "umap", "E_star") < pick(s, "pca", "E_star") for s in secs),
            random_last_Estar=sum(order(g[g.object == s], "E_star")[-1] == "random" for s in secs),
            pca_above_scvi_raw=sum(pick(s, "pca", "E") > pick(s, "scvi", "E") for s in secs),
            umap_below_permuted_pca_raw=sum(pick(s, "umap", "E") < pick(s, "permuted_pca", "E") for s in secs),
            ordering_changed_sections=sum(order(g[g.object == s], "E") != order(g[g.object == s], "E_star")
                                          for s in secs),
            metastasis_Sstar_negative=all(pick("metastasis", c, "S_star") < 0 for c in ("pca", "scvi", "umap")),
            random_Estar_min=float(g[g.candidate == "random"].E_star.min()),
            random_Estar_max=float(g[g.candidate == "random"].E_star.max()),
            umap_Estar_min=float(g[g.candidate == "umap"].E_star.min()),
            umap_Estar_max=float(g[g.candidate == "umap"].E_star.max()))

    worlds = pd.DataFrame({w: one_world(g) for w, g in raw.groupby("world")}).T
    base_world = worlds.loc["A0_archived_saved_order#0"]
    claims = dict(n_reference_graphs_per_section=int(len(worlds)),
                  baseline=base_world.to_dict(),
                  held_in_every_graph={c: bool((worlds[c] == base_world[c]).all()) for c in worlds.columns
                                       if not c.endswith(("min", "max"))},
                  random_Estar_range_all_graphs=[float(worlds.random_Estar_min.min()), float(worlds.random_Estar_max.max())],
                  umap_Estar_range_all_graphs=[float(worlds.umap_Estar_min.min()), float(worlds.umap_Estar_max.max())])
    spread = summary.groupby("metric").agg(largest_range=("range", "max"), largest_sd=("sd", "max")).round(5)
    claims["largest_range_over_all_62_graphs"] = spread["largest_range"].to_dict()
    claims["largest_sd_over_all_62_graphs"] = spread["largest_sd"].to_dict()
    shift = (summary.tie_complete - summary.saved_order).abs()
    claims["largest_tie_complete_shift"] = shift.groupby(summary.metric).max().round(5).to_dict()
    claims["reproduction_max_abs_difference_vs_saved"] = reproduction
    return summary, claims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarise", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    raw_path = RESULTS / "tie_sensitivity_raw.csv"
    if not args.summarise:
        with Pool(len(SECTIONS)) as pool:
            results = pool.map(run_section, SECTIONS)
        pd.DataFrame([r for rows, _ in results for r in rows]).to_csv(raw_path, index=False)
        json.dump([m for _, m in results], open(RESULTS / "tie_sections.json", "w"), indent=1)
    summary, claims = summarise(pd.read_csv(raw_path))
    summary.to_csv(RESULTS / "tie_sensitivity_summary.csv", index=False)
    json.dump(claims, open(RESULTS / "tie_claims.json", "w"), indent=1, default=float)
    print(json.dumps(claims, indent=1, default=float))


if __name__ == "__main__":
    main()
