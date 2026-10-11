"""Shared helpers for the follow-up analyses.

All RAAG metric values are obtained from the archived scorer
``scripts/local_raag.py`` (imported, never edited).  Helpers here only add
quantities the archived scorer does not provide:

* exact permutation-null expectations of E and S (derived below),
* a tie-robust far-vs-near AUROC separation statistic,
* pair-counting comparisons of reference/candidate component partitions.

Exact null expectations.  Let pi be a uniformly random permutation applied to
candidate rows, so h_{Z pi}(i, j) = h_Z(pi(i), pi(j)).  For any fixed distinct
pair (i, j), (pi(i), pi(j)) is uniform over ordered distinct pairs, so for any
fixed reference pair set A,

    E_pi[ |A|^-1 sum_{(i,j) in A} f(h_Z(pi i, pi j)) ] = mean_{u<v} f(h_Z(u, v)).

E uses f(h) = 1/h (1/inf = 0); S uses f(h) = 1[h >= m_Z or h = inf] and the
candidate median m_Z is permutation invariant.  Hence

    E0 = mean_{u<v} 1/h_Z(u,v),     S0 = mean_{u<v} 1[h_Z(u,v) >= m_Z or inf].

These are exact expectations (not Monte Carlo estimates) whenever the candidate
k-NN graph is invariant to row relabelling, i.e. no neighbour-cutoff ties.
Monte Carlo permutations are run alongside where requested and compared.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.sparse.csgraph import connected_components

PAPER = Path(__file__).resolve().parents[3]
OUT = PAPER / "evidence" / "followup-20261010"
SCRIPTS = PAPER / "scripts"
UPSTREAM = Path(os.environ.get("RAAG_UPSTREAM", OUT / "upstream"))
sys.path.insert(0, str(SCRIPTS))

import local_raag  # noqa: E402
from local_raag import (  # noqa: E402
    _hop_distances,
    edge_survival,
    knn_hop_graph,
    reachability_survival,
    separation_survival,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    return x


def dump(path: Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(clean(obj), indent=2, allow_nan=False) + "\n")


def hops(x: np.ndarray, k: int) -> np.ndarray:
    """Archived-scorer hop matrix (knn_hop_graph + _hop_distances)."""
    return _hop_distances(knn_hop_graph(x, k))


def components(x: np.ndarray, k: int) -> np.ndarray:
    _, labels = connected_components(knn_hop_graph(x, k), directed=False)
    return labels


def pair_count_connected(labels: np.ndarray) -> int:
    _, c = np.unique(labels, return_counts=True)
    return int(np.sum(c * (c - 1) // 2))


def component_formula_R(labels: np.ndarray) -> float:
    """sum_c C(n_c,2) / C(n,2) of a candidate component partition."""
    n = len(labels)
    return pair_count_connected(labels) / (n * (n - 1) / 2)


def partition_pair_scores(ref_labels: np.ndarray, cand_labels: np.ndarray) -> dict:
    """Pair-counting comparison of reference vs candidate component partitions.

    recall    = |pairs connected in both| / |pairs connected in reference|  (== archived R)
    precision = |pairs connected in both| / |pairs connected in candidate|
    Also ARI of the two partitions.
    """
    from sklearn.metrics import adjusted_rand_score

    joint = ref_labels.astype(np.int64) * (cand_labels.max() + 1) + cand_labels
    both = pair_count_connected(joint)
    nref = pair_count_connected(ref_labels)
    ncand = pair_count_connected(cand_labels)
    recall = both / nref if nref else float("nan")
    precision = both / ncand if ncand else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "pair_recall": recall,
        "pair_precision": precision,
        "pair_f1": f1,
        "component_ari": float(adjusted_rand_score(ref_labels, cand_labels)),
        "n_ref_components": int(len(np.unique(ref_labels))),
        "n_cand_components": int(len(np.unique(cand_labels))),
    }


def ref_sets(ref_h: np.ndarray, far_quantile: float = 0.9):
    """Reference edge list and far set exactly as the archived scorer defines them."""
    n = ref_h.shape[0]
    tri = np.triu_indices(n, k=1)
    rh = ref_h[tri]
    finite = np.isfinite(rh)
    cutoff = float(np.quantile(rh[finite], far_quantile))
    far = finite & (rh >= cutoff)
    edge = rh == 1
    return tri, edge, far, cutoff


def e_s_from_hops(ref_h: np.ndarray, cand_h: np.ndarray, sets=None) -> dict:
    """E, S, analytic nulls, chance-corrected scores, and AUROC from hop matrices.

    E and S reproduce local_raag.edge_survival / separation_survival given the
    same hop matrices (asserted by callers on validation subsets).
    """
    tri, edge, far, cutoff = sets if sets is not None else ref_sets(ref_h)
    ch = cand_h[tri]
    inv = np.zeros_like(ch)
    fin = np.isfinite(ch)
    inv[fin] = 1.0 / np.maximum(ch[fin], 1.0)
    E = float(inv[edge].mean())
    E0 = float(inv.mean())
    m = float(np.median(ch[fin])) if fin.any() else np.inf
    sep = (~fin) | (ch >= m)
    S = float(sep[far].mean())
    S0 = float(sep.mean())
    au = auroc_far_near(ch, far, edge)
    return {
        "E": E, "E0_exact": E0, "E_star": chance_correct(E, E0),
        "S": S, "S0_exact": S0, "S_star": chance_correct(S, S0),
        "AUROC_far_vs_edge": au,
        "S_threshold_hops": m, "far_cutoff_hops": cutoff,
        "n_far": int(far.sum()), "n_edges": int(edge.sum()),
    }


def chance_correct(v: float, v0: float) -> float:
    return float((v - v0) / (1.0 - v0)) if v0 < 1.0 else float("nan")


def auroc_far_near(cand_pair_hops: np.ndarray, far: np.ndarray, near: np.ndarray) -> float:
    """P(h_Z(far pair) > h_Z(near pair)) + 0.5 P(tie); inf ranks highest.

    Near pairs are reference edges (h_X = 1); far pairs are the archived far set
    (h_X >= 90th percentile).  Mann-Whitney with mid-ranks, so integer hop ties
    contribute 1/2 and the permutation-null expectation is exactly 0.5.
    """
    a = cand_pair_hops[far]
    b = cand_pair_hops[near]
    big = np.nanmax(np.where(np.isfinite(cand_pair_hops), cand_pair_hops, np.nan)) + 1 if np.isfinite(cand_pair_hops).any() else 1.0
    a = np.where(np.isfinite(a), a, big)
    b = np.where(np.isfinite(b), b, big)
    vals = np.concatenate([a, b])
    uniq, inv, counts = np.unique(vals, return_inverse=True, return_counts=True)
    # mid-ranks for each unique value
    cum = np.cumsum(counts)
    mid = cum - (counts - 1) / 2.0
    ranks = mid[inv]
    ra = ranks[: len(a)].sum()
    u = ra - len(a) * (len(a) + 1) / 2.0
    return float(u / (len(a) * len(b)))


def perm_null_from_hops(ref_h: np.ndarray, cand_h: np.ndarray, n_perm: int, seed: int, sets=None) -> dict:
    """Monte Carlo row-permutation null using relabelled candidate hop matrices.

    Equivalent to rescoring candidate[perm] when the candidate kNN graph has no
    cutoff ties (validated against direct archived-scorer calls by callers).
    """
    sets = sets if sets is not None else ref_sets(ref_h)
    rng = np.random.default_rng(seed)
    out = {"E": [], "S": [], "AUROC_far_vs_edge": []}
    for _ in range(n_perm):
        p = rng.permutation(cand_h.shape[0])
        r = e_s_from_hops(ref_h, cand_h[np.ix_(p, p)], sets)
        for key in out:
            out[key].append(r[key])
    return {k: np.asarray(v) for k, v in out.items()}
