"""Reference-grounded dimensionality-reduction metrics.

This module is intentionally separate from the historical ``eval_lib``
metric battery.  The latter is kept for backwards-compatible benchmark
tables; this file implements the published co-ranking quantities directly
and adds graph-geodesic comparisons for trajectory claims.

The co-ranking definitions follow Lee & Verleysen and the ``dimRed``
implementation: ``Q_NX`` is the fraction of shared K-nearest neighbours,
``LCMC`` subtracts the random-overlap baseline, and ``Q_local``/``Q_global``
average the curve on either side of its maximum.  No labels are required.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import pearsonr, spearmanr
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsRegressor


def _as_array(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 3:
        raise ValueError("embedding must have shape (n>=3, d>=1)")
    if not np.isfinite(arr).all():
        raise ValueError("embedding contains NaN or infinite values")
    return arr


def _rank_matrix(x: np.ndarray) -> np.ndarray:
    """Return one-based neighbour ranks, with rank zero on the diagonal."""

    arr = _as_array(x)
    distances = pairwise_distances(arr)
    order = np.argsort(distances, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.int64)
    ranks[np.arange(arr.shape[0])[:, None], order] = np.arange(arr.shape[0])
    ranks[np.arange(arr.shape[0]), np.arange(arr.shape[0])] = 0
    # The self entry was rank zero; all non-self entries already have ranks
    # 1..n-1 because self is the unique zero-distance item up to stable ties.
    return ranks


def coranking_matrix(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Build the ``(n-1) × (n-1)`` ordered-pair co-ranking matrix."""

    high = _as_array(high)
    low = _as_array(low)
    if high.shape[0] != low.shape[0]:
        raise ValueError("high and low embeddings must have the same number of rows")
    n = high.shape[0]
    rh = _rank_matrix(high)
    rl = _rank_matrix(low)
    mask = (rh > 0) & (rl > 0)
    q = np.zeros((n - 1, n - 1), dtype=np.int64)
    np.add.at(q, (rh[mask] - 1, rl[mask] - 1), 1)
    return q


def coranking_metrics(high: np.ndarray, low: np.ndarray) -> dict[str, float]:
    """Return standard co-ranking quality summaries.

    ``qnx`` is not returned as a list to keep JSON output compact; the scalar
    summaries include the selected ``kmax``, the local/global means, and the
    log-weighted area under the random-corrected ``R_NX`` curve.
    """

    q = coranking_matrix(high, low)
    n = q.shape[0] + 1
    ks = np.arange(1, n - 1, dtype=float)
    cumulative = q.cumsum(axis=0).cumsum(axis=1)
    qnx = np.asarray([cumulative[int(k) - 1, int(k) - 1] / (k * n) for k in ks])
    lcmc = qnx - ks / (n - 1.0)
    kmax_index = int(np.argmax(lcmc))
    kmax = kmax_index + 1
    q_local = float(np.mean(qnx[:kmax]))
    q_global = float(np.mean(qnx[kmax:])) if kmax < len(qnx) else float(qnx[-1])

    # R_NX rescales Q_NX so that random overlap is zero and a perfect
    # embedding is one.  The final K=n-1 value is undefined and omitted.
    denom = (n - 1.0) - ks
    rnx = ((n - 1.0) * qnx - ks) / denom
    weights = 1.0 / ks
    auc = float(np.sum(weights * rnx) / np.sum(weights))
    return {
        "q_local": float(np.clip(q_local, 0.0, 1.0)),
        "q_global": float(np.clip(q_global, 0.0, 1.0)),
        "q_kmax": float(np.clip(qnx[kmax_index], 0.0, 1.0)),
        "lcmc_kmax": float(lcmc[kmax_index]),
        "kmax": float(kmax),
        "rnx_at_kmax": float(np.clip(rnx[kmax_index], -1.0, 1.0)),
        "auc_ln_rnx": float(np.clip(auc, -1.0, 1.0)),
    }


def _continuity(high: np.ndarray, low: np.ndarray, k: int) -> float:
    high = _as_array(high)
    low = _as_array(low)
    if high.shape[0] != low.shape[0]:
        raise ValueError("high and low embeddings must have the same number of rows")
    n = high.shape[0]
    kk = int(min(max(1, k), n - 2))
    rh = _rank_matrix(high)
    rl = _rank_matrix(low)
    high_neighbours = np.argsort(rh, axis=1)[:, 1 : kk + 1]
    penalty = 0.0
    for i in range(n):
        ranks = rl[i, high_neighbours[i]]
        missed = ranks[ranks > kk]
        penalty += float(np.sum(missed - kk))
    denominator = n * kk * (2 * n - 3 * kk - 1)
    if denominator <= 0:
        return float("nan")
    return float(np.clip(1.0 - 2.0 * penalty / denominator, 0.0, 1.0))


def standard_dre_metrics(high: np.ndarray, low: np.ndarray, k: int = 15) -> dict[str, float]:
    """Combine standard trustworthiness/continuity with co-ranking metrics."""

    high = _as_array(high)
    low = _as_array(low)
    if high.shape[0] != low.shape[0]:
        raise ValueError("high and low embeddings must have the same number of rows")
    kk = int(min(max(1, k), high.shape[0] - 2))
    tri = np.triu_indices(high.shape[0], k=1)
    dh = pairwise_distances(high)[tri]
    dl = pairwise_distances(low)[tri]
    co = coranking_metrics(high, low)
    return {
        "trustworthiness": float(trustworthiness(high, low, n_neighbors=kk)),
        "continuity": _continuity(high, low, kk),
        "distance_spearman": float(spearmanr(dh, dl).statistic),
        "distance_pearson": float(pearsonr(dh, dl).statistic),
        **co,
    }


def _weighted_knn_graph(x: np.ndarray, k: int) -> csr_matrix:
    arr = _as_array(x)
    kk = int(min(max(2, k), arr.shape[0] - 1))
    distances, indices = NearestNeighbors(n_neighbors=kk + 1).fit(arr).kneighbors(arr)
    rows = np.repeat(np.arange(arr.shape[0]), kk)
    graph = csr_matrix(
        (distances[:, 1:].ravel(), (rows, indices[:, 1:].ravel())),
        shape=(arr.shape[0], arr.shape[0]),
    )
    return graph.maximum(graph.T)


def _largest_component_fraction(graph: csr_matrix) -> float:
    n_components, labels = connected_components(graph, directed=False)
    if n_components == 1:
        return 1.0
    _, counts = np.unique(labels, return_counts=True)
    return float(counts.max() / graph.shape[0])


def graph_geodesic_metrics(
    high: np.ndarray,
    low: np.ndarray,
    pseudotime: Iterable[float] | None = None,
    k: int = 15,
) -> dict[str, float]:
    """Compare graph geodesics of a reference and candidate embedding.

    The reference graph is built from ``high`` (for example, PCA of the held-
    out expression matrix).  Pairwise geodesic correlation is computed only
    on pairs reachable in both graphs; component fractions are emitted so a
    fragmented candidate cannot hide behind pair filtering.  If pseudotime is
    supplied, the root is the minimum-reference-pseudotime cell and the
    scIB-style trajectory-conservation transform ``(rho+1)/2`` is reported.
    """

    high = _as_array(high)
    low = _as_array(low)
    if high.shape[0] != low.shape[0]:
        raise ValueError("high and low embeddings must have the same number of rows")
    gh = _weighted_knn_graph(high, k)
    gl = _weighted_knn_graph(low, k)
    dh = shortest_path(gh, directed=False)
    dl = shortest_path(gl, directed=False)
    tri = np.triu_indices(high.shape[0], k=1)
    both = np.isfinite(dh[tri]) & np.isfinite(dl[tri])
    if both.sum() < 3:
        pair_rho = float("nan")
    else:
        pair_rho = float(spearmanr(dh[tri][both], dl[tri][both]).statistic)
    out = {
        "reference_lcc_fraction": _largest_component_fraction(gh),
        "candidate_lcc_fraction": _largest_component_fraction(gl),
        "geodesic_pairwise_spearman": pair_rho,
        "geodesic_pair_count": float(both.sum()),
    }
    if pseudotime is not None:
        t = np.asarray(list(pseudotime), dtype=float).reshape(-1)
        if t.shape[0] != high.shape[0] or not np.isfinite(t).all():
            raise ValueError("pseudotime must be finite and aligned with embeddings")
        root = int(np.argmin(t))
        rh = dh[root]
        rl = dl[root]
        finite = np.isfinite(rh) & np.isfinite(rl)
        if finite.sum() < 3:
            rho = float("nan")
        else:
            rho = float(spearmanr(rh[finite], rl[finite]).statistic)
        out["root_geodesic_spearman"] = rho
        # Historical key retained for compatibility: local root-geodesic
        # Spearman rescaling, NOT the official scIB diffusion-pseudotime API.
        out["trajectory_conservation"] = float((rho + 1.0) / 2.0) if np.isfinite(rho) else float("nan")
        # A root correlation computed on the reachable subset can look
        # perfect even when a candidate graph drops an entire island.  Keep
        # the unpenalised score for transparency, and expose a conservative
        # score that charges the candidate's disconnected fraction.
        if np.isfinite(out["trajectory_conservation"]):
            out["trajectory_conservation_penalized"] = float(
                out["trajectory_conservation"] * out["candidate_lcc_fraction"]
            )
        else:
            out["trajectory_conservation_penalized"] = float("nan")
    return out


def knn_signal_r2(
    embedding: np.ndarray,
    signal: np.ndarray,
    n_neighbors: int = 15,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    """Cross-validated KNN predictability of an external cell-level signal.

    This is a supervised *diagnostic* rather than a training objective: the
    signal is never used to fit the embedding.  It is useful for lineage
    probabilities or marker scores when a continuous biological reference is
    available.  The reported score is the uniform multi-output :math:`R^2`
    averaged over folds, with the median included to expose outlier folds.
    """

    z = _as_array(embedding)
    y = np.asarray(signal, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2 or y.shape[0] != z.shape[0]:
        raise ValueError("signal must have shape (n, m) with the same n as embedding")
    if not np.isfinite(y).all():
        raise ValueError("signal contains NaN or infinite values")
    n = z.shape[0]
    folds = int(min(max(2, n_splits), n))
    k = int(min(max(1, n_neighbors), n - 1))
    scores: list[float] = []
    splitter = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    for train_idx, test_idx in splitter.split(z):
        model = KNeighborsRegressor(n_neighbors=min(k, len(train_idx)), weights="distance")
        model.fit(z[train_idx], y[train_idx])
        score = float(model.score(z[test_idx], y[test_idx]))
        scores.append(score)
    return {
        "mean_r2": float(np.mean(scores)),
        "median_r2": float(np.median(scores)),
        "min_r2": float(np.min(scores)),
        "max_r2": float(np.max(scores)),
        "n_splits": float(folds),
    }
