"""RAAG-v3 working revision: explicit identity, ties and absolute variation.

Historical ``local_raag.py`` and its outputs are not changed. This module is a
new, not-yet-numerically-validated measurement version. ``eligible`` denotes
operational eligibility only, never biological utility or perturbation stability.
The dense distance and distinct-pair calculations have quadratic cost.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral
from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist

RAAG_VERSION = "RAAG-v3-identity-compensated-20261009"
PROBABILITY_FLOOR = 1e-12
DENOMINATOR_GUARD = 1e-12
NEAR_UNIFORM_PERPLEXITY_FRACTION = 0.95
CellID = tuple[str, int]


def _array(values: np.ndarray, name: str) -> np.ndarray:
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError(f"{name} must have shape (n>=2, d>=1)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")
    return array


def validate_cell_ids(cell_ids: Sequence[CellID], n: int) -> tuple[CellID, ...]:
    if cell_ids is None or len(cell_ids) != n:
        raise ValueError("one explicit persistent ID per row is required")
    result = []
    for value in cell_ids:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError("IDs must be (namespace, integer index) pairs")
        namespace, index = value
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("ID namespace must be a nonempty string")
        if not isinstance(index, Integral) or isinstance(index, (bool, np.bool_)) or index < 0:
            raise ValueError("ID index must be a nonnegative integer")
        result.append((namespace, int(index)))
    if len(set(result)) != n:
        raise ValueError("persistent IDs must be unique within an object")
    return tuple(result)


def cell_ids_from_indices(namespace: str, test_idx: np.ndarray, n: int | None = None) -> tuple[CellID, ...]:
    """Computational identities, not reconstructed upstream biological barcodes."""
    indices = np.asarray(test_idx)
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise ValueError("test_idx must be a one-dimensional integer array")
    count = len(indices) if n is None else n
    return validate_cell_ids([(namespace, int(index)) for index in indices], count)


def _aligned_ids(reference_ids: Sequence[CellID], candidate_ids: Sequence[CellID], n: int) -> tuple[CellID, ...]:
    left, right = validate_cell_ids(reference_ids, n), validate_cell_ids(candidate_ids, n)
    if left != right:
        raise ValueError("ordered reference/candidate IDs differ; implicit alignment is forbidden")
    return left


@dataclass(frozen=True)
class GraphResult:
    graph: csr_matrix
    directed: csr_matrix
    metadata: dict[str, object]


def build_knn_graph(x: np.ndarray, k: int, *, cell_ids: Sequence[CellID]) -> GraphResult:
    """Full distance rows ensure complete cutoff ties precede ID selection."""
    original_dtype = str(np.asarray(x).dtype)
    array = _array(x, "embedding")
    ids = validate_cell_ids(cell_ids, len(array))
    if not isinstance(k, Integral) or isinstance(k, (bool, np.bool_)) or k < 1:
        raise ValueError("k must be a positive integer, not bool")
    effective_k = min(int(k), len(array) - 1)
    canonical = np.asarray(sorted(range(len(ids)), key=ids.__getitem__), dtype=int)
    id_rank = np.empty(len(ids), dtype=int)
    id_rank[canonical] = np.arange(len(ids))
    distances = cdist(array, array, metric="sqeuclidean")
    if not np.isfinite(distances).all():
        raise ValueError("distance calculation overflowed; rescale inputs explicitly")
    np.fill_diagonal(distances, np.inf)
    neighbours = np.empty((len(array), effective_k), dtype=int)
    cutoff_tie_rows, cutoff_tie_sizes, cutoff_gaps = 0, [], []
    for row in range(len(array)):
        order = np.lexsort((id_rank, distances[row]))
        neighbours[row] = order[:effective_k]
        cutoff = distances[row, order[effective_k - 1]]
        tie_size = int(np.count_nonzero(distances[row] == cutoff))
        cutoff_tie_sizes.append(tie_size)
        if effective_k < len(array) - 1 and distances[row, order[effective_k]] == cutoff:
            cutoff_tie_rows += 1
        if effective_k < len(array) - 1:
            cutoff_gaps.append(float(distances[row, order[effective_k]] - cutoff))
    directed = csr_matrix(
        (np.ones(neighbours.size, dtype=np.int8),
         (np.repeat(np.arange(len(array)), effective_k), neighbours.ravel())),
        shape=(len(array), len(array)),
    )
    graph = directed.maximum(directed.T)
    if directed.diagonal().any() or not np.all(np.diff(directed.indptr) == effective_k):
        raise RuntimeError("nonself/effective-k graph contract failed")
    constant = bool(np.all(array == array[0]))
    identity_bytes = json.dumps([ids[index] for index in canonical], separators=(",", ":")).encode()
    canonical_edges = np.ascontiguousarray(graph[canonical][:, canonical].toarray(), dtype=np.int8).tobytes()
    return GraphResult(graph, directed, {
        "graph_status": "noninformative_constant" if constant else "eligible",
        "usable_geometry": not constant,
        "requested_k": int(k), "effective_k": effective_k,
        "input_dtype": original_dtype, "distance_dtype": "float64",
        "distance": "squared_euclidean_full_rows",
        "self_rule": "explicit_diagonal_exclusion_with_validated_unique_IDs",
        "tie_rule": "distance_then_namespace_then_numeric_index_complete_cutoff_group",
        "symmetrisation": "undirected_union", "cutoff_tie_rows": cutoff_tie_rows,
        "maximum_cutoff_tie_size": max(cutoff_tie_sizes),
        "minimum_cutoff_gap_squared_distance": min(cutoff_gaps) if cutoff_gaps else None,
        "minimum_positive_cutoff_gap_squared_distance": min((gap for gap in cutoff_gaps if gap > 0), default=None),
        "zero_distance_nonself_pairs": int(np.count_nonzero(distances == 0) // 2),
        "canonical_edges_sha256": hashlib.sha256(identity_bytes + canonical_edges).hexdigest(),
        "tie_interpretation": "reproducibility bookkeeping, not unique geometry or near-tie stability",
    })


def knn_hop_graph(x: np.ndarray, k: int, *, cell_ids: Sequence[CellID]) -> csr_matrix:
    return build_knn_graph(x, k, cell_ids=cell_ids).graph


def _jensen_shannon(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Symmetric JSD in nats; fixed even-series branch avoids near-equality loss.

    For m=(p+q)/2 and t=(p-q)/(p+q), each coordinate is
    m * sum_{a>=1} t**(2a)/(2a*(2a-1)). The first six terms are used for
    |t|<=.01; elsewhere use the defining log expression with exact endpoints.
    This stability change is versioned and requires the independent oracle.
    """
    p, q = np.broadcast_arrays(np.asarray(p, float), np.asarray(q, float))
    total = p + q
    t = np.divide(p - q, total, out=np.zeros_like(total), where=total > 0)
    values = np.empty_like(t)
    small = np.abs(t) <= .01
    t2 = t[small] ** 2
    values[small] = sum(t2 ** power / (2 * power * (2 * power - 1)) for power in range(1, 7))
    other = t[~small]
    with np.errstate(divide="ignore", invalid="ignore"):
        left = np.where(other == -1, 0, (1 + other) * np.log1p(other))
        right = np.where(other == 1, 0, (1 - other) * np.log1p(-other))
    values[~small] = .5 * (left + right)
    return np.sum(.5 * total * values, axis=-1)


def to_simplex(latent: np.ndarray, *, already_simplex: bool) -> tuple[np.ndarray, int]:
    z = _array(latent, "latent")
    if not isinstance(already_simplex, (bool, np.bool_)):
        raise ValueError("native allocation semantics require an explicit boolean")
    if already_simplex:
        if np.any(z < 0) or not np.allclose(z.sum(axis=1), 1, atol=1e-4, rtol=0):
            raise ValueError("declared native allocation must be nonnegative and row-stochastic")
        floored = int(np.count_nonzero(z < PROBABILITY_FLOOR))
        probabilities = np.maximum(z, PROBABILITY_FLOOR)
    else:
        floored = 0
        probabilities = np.exp(z - z.max(axis=1, keepdims=True))
    # Accurate row totals matter when a nearly-unit coordinate accompanies
    # tiny tails: ordinary reduction error can dominate their absolute range.
    totals = np.asarray([math.fsum(row.tolist()) for row in probabilities])[:, None]
    return probabilities / totals, floored


def _absolute_diagnostics(p: np.ndarray) -> dict[str, float | int]:
    pair_js, pair_tv = [], []
    for row in range(len(p) - 1):
        pair_js.append(float(np.sum(_jensen_shannon(p[row], p[row + 1:]))))
        pair_tv.append(float(np.sum(.5 * np.abs(p[row] - p[row + 1:]).sum(axis=1))))
    pairs = len(p) * (len(p) - 1) // 2
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.where(p > 0, p * np.log(p), 0).sum(axis=1)
    perplexity = math.fsum(np.exp(entropy).tolist()) / len(p)
    return {
        "pair_js": math.fsum(pair_js) / pairs,
        "mean_pair_tv": math.fsum(pair_tv) / pairs,
        "max_coordinate_range": float(np.max(np.ptp(p, axis=0))),
        "mean_perplexity": perplexity,
        "perplexity_fraction": perplexity / p.shape[1], "distinct_pairs": pairs,
    }


def _allocation_from_graph(z: np.ndarray, reference_graph: GraphResult, *, native: bool, return_probabilities: bool) -> dict[str, object]:
    probabilities, flooring_entries = to_simplex(z, already_simplex=native)
    native = bool(native)
    absolute = _absolute_diagnostics(probabilities)
    usable = bool(reference_graph.metadata["usable_geometry"])
    edge_js = None
    if usable:
        src, dst = reference_graph.graph.nonzero()
        upper = src < dst
        divergences = _jensen_shannon(probabilities[src[upper]], probabilities[dst[upper]])
        edge_js = math.fsum(divergences.tolist()) / len(divergences)
    pair_js = float(absolute["pair_js"])
    degenerate = pair_js <= DENOMINATOR_GUARD
    near_uniform = float(absolute["perplexity_fraction"]) > NEAR_UNIFORM_PERPLEXITY_FRACTION
    ratio = None if edge_js is None or pair_js == 0 else 1 - edge_js / pair_js
    raw = None if edge_js is None else 0.0 if degenerate else float(np.clip(ratio, 0, 1))
    status = "unsupported_reference" if not usable else "degenerate" if degenerate else "near_uniform" if near_uniform else "eligible"
    result: dict[str, object] = {
        "raag_version": RAAG_VERSION,
        "allocation_is_native_simplex": native,
        "allocation_identity_source": "caller_declared",
        "allocation_dimension": probabilities.shape[1] if native else None,
        "allocation_status": status if native else "non_native",
        "allocation_coherence": raw if native and status == "eligible" else None,
        "allocation_coherence_raw": raw if native else None,
        "allocation_relative_coherence_unclipped": ratio if native else None,
        "allocation_edge_js": edge_js if native else None,
        "allocation_pair_js": pair_js if native else None,
        "allocation_mean_pair_tv": absolute["mean_pair_tv"] if native else None,
        "allocation_max_coordinate_range": absolute["max_coordinate_range"] if native else None,
        "allocation_mean_perplexity": absolute["mean_perplexity"] if native else None,
        "allocation_perplexity_fraction": absolute["perplexity_fraction"] if native else None,
        "allocation_degenerate": degenerate if native else False,
        "allocation_near_uniform": near_uniform if native else False,
        "coherence_estimator": "exact_unordered_distinct_pairs",
        "coherence_denominator_pairs": absolute["distinct_pairs"],
        "flooring_entries": flooring_entries,
        "status_interpretation": "operational eligibility; no biological usefulness threshold",
    }
    if not native:
        result.update({
            "softmax_coordinate_status": status,
            "softmax_coordinate_coherence": raw if status == "eligible" else None,
            "softmax_coordinate_coherence_raw": raw,
            "softmax_coordinate_relative_coherence_unclipped": ratio,
            "softmax_coordinate_dimension": probabilities.shape[1],
            **{f"softmax_coordinate_{name}": value for name, value in absolute.items()},
            "softmax_coordinate_edge_js": edge_js,
        })
    if return_probabilities:
        result["input_probabilities" if native else "input_coordinates"] = np.asarray(z, float).tolist()
        result["measured_probabilities"] = probabilities.tolist()
    return result


def allocation_diagnostics(reference: np.ndarray, latent: np.ndarray, *, reference_ids: Sequence[CellID], allocation_ids: Sequence[CellID], k: int = 15, already_simplex: bool, return_probabilities: bool = False) -> dict[str, object]:
    x, z = _array(reference, "reference"), _array(latent, "latent")
    if len(x) != len(z):
        raise ValueError("reference/allocation row counts differ")
    ids = _aligned_ids(reference_ids, allocation_ids, len(x))
    graph = build_knn_graph(reference, k, cell_ids=ids)
    out = _allocation_from_graph(z, graph, native=already_simplex, return_probabilities=return_probabilities)
    out["reference_graph"] = graph.metadata
    return out


def _geometry(reference: GraphResult, candidate: GraphResult, far_quantile: float) -> dict[str, object]:
    names = ("reachability_survival", "edge_survival", "separation_survival")
    result: dict[str, object] = {name: None for name in names}
    if not .5 <= far_quantile < 1:
        raise ValueError("far_quantile must lie in [.5,1)")
    if not reference.metadata["usable_geometry"]:
        return {**result, "geometry_status": "reference_noninformative"}
    if not candidate.metadata["usable_geometry"]:
        return {**result, "geometry_status": "candidate_noninformative"}
    ref_hops = shortest_path(reference.graph, directed=False, unweighted=True)
    cand_hops = shortest_path(candidate.graph, directed=False, unweighted=True)
    tri = np.triu_indices(reference.graph.shape[0], 1)
    rh, ch = ref_hops[tri], cand_hops[tri]
    connected = np.isfinite(rh)
    result["n_reference_connected_pairs"] = int(np.count_nonzero(connected))
    if np.count_nonzero(connected) < 10:
        return {**result, "geometry_status": "insufficient_reference_pairs"}
    result["reachability_survival"] = float(np.mean(np.isfinite(ch[connected])))
    src, dst = reference.graph.nonzero()
    keep = src < dst
    edge_hops = cand_hops[src[keep], dst[keep]]
    reciprocal = np.divide(1, edge_hops, out=np.zeros_like(edge_hops), where=np.isfinite(edge_hops))
    result.update({"edge_survival": math.fsum(reciprocal.tolist()) / len(reciprocal),
                   "n_reference_edges": len(reciprocal),
                   "edge_severed_fraction": float(np.mean(~np.isfinite(edge_hops))),
                   "edge_preserved_fraction": float(np.mean(edge_hops == 1))})
    cutoff = float(np.quantile(rh[connected], far_quantile))
    far = connected & (rh >= cutoff)
    result["n_far_pairs"] = int(np.count_nonzero(far))
    if np.count_nonzero(far) < 10:
        return {**result, "geometry_status": "insufficient_far_pairs"}
    finite_candidate = ch[np.isfinite(ch)]
    threshold = float(np.median(finite_candidate))
    result.update({"separation_survival": float(np.mean(ch[far] >= threshold)),
                   "separation_far_cutoff_hops": cutoff,
                   "separation_threshold_hops": threshold,
                   "geometry_status": "eligible"})
    return result


def raag_battery(reference: np.ndarray, latent: np.ndarray, *, reference_ids: Sequence[CellID], candidate_ids: Sequence[CellID], k: int = 15, far_quantile: float = .9, already_simplex: bool) -> dict[str, object]:
    x, z = _array(reference, "reference"), _array(latent, "latent")
    if len(x) != len(z):
        raise ValueError("reference/candidate row counts differ")
    ids = _aligned_ids(reference_ids, candidate_ids, len(x))
    reference_graph = build_knn_graph(reference, k, cell_ids=ids)
    candidate_graph = build_knn_graph(latent, k, cell_ids=ids)
    return {
        "raag_version": RAAG_VERSION,
        "ordered_ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
        "reference_graph": reference_graph.metadata, "candidate_graph": candidate_graph.metadata,
        **_geometry(reference_graph, candidate_graph, far_quantile),
        **_allocation_from_graph(z, reference_graph, native=already_simplex, return_probabilities=False),
    }
