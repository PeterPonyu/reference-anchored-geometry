"""Independent, small-fixture RAAG-v3 oracle; standard-library imports only.

Graph construction deliberately uses dense Python squared distances and a full
sort, hop distances use breadth-first search, and allocation diagnostics use
80-digit Decimal arithmetic.  No production scorer, NumPy, SciPy, or sklearn
code is imported or called.  Public diagnostic values are floats for assertions;
the measured simplex and Jensen--Shannon calculation stay Decimal internally.

This is validation source, not a compute-gate receipt.  Creating or AST-parsing
this file does not establish numerical correctness or authorize execution.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal, localcontext
from hashlib import sha256
import json
import math
from numbers import Integral
from typing import Iterable, Sequence


CellID = tuple[str, int]
DECIMAL_PRECISION = 80
PROBABILITY_FLOOR = 1e-12
DENOMINATOR_THRESHOLD = 1e-12
NEAR_UNIFORM_FRACTION = 0.95
MINIMUM_PAIRS = 10
GEOMETRY_KEYS = (
    "reachability_survival",
    "n_reference_connected_pairs",
    "edge_survival",
    "edge_severed_fraction",
    "edge_preserved_fraction",
    "n_reference_edges",
    "separation_survival",
    "separation_far_cutoff_hops",
    "separation_threshold_hops",
    "n_far_pairs",
)


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not a boolean")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _namespace(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("CellID namespace must be a nonempty string")
    return value


def validate_cell_ids(cell_ids: Iterable[CellID], n: int) -> tuple[CellID, ...]:
    """Validate logical identities without sorting or aligning their rows."""
    try:
        supplied = tuple(cell_ids)
    except TypeError as error:
        raise ValueError("cell_ids must be an ordered iterable of CellID tuples") from error
    if len(supplied) != n:
        raise ValueError("cell_ids count must equal the number of rows")
    result = []
    for identity in supplied:
        if not isinstance(identity, (tuple, list)) or len(identity) != 2:
            raise ValueError("each CellID must be a (namespace, index) tuple")
        result.append((_namespace(identity[0]), _integer(identity[1], "CellID index", minimum=0)))
    if len(set(result)) != n:
        raise ValueError("cell_ids must be unique")
    return tuple(result)


def cell_ids_from_indices(
    namespace: str, test_idx: Iterable[int], n: int | None = None
) -> tuple[CellID, ...]:
    namespace = _namespace(namespace)
    if getattr(test_idx, "ndim", 1) != 1:
        raise ValueError("test_idx must be one-dimensional")
    dtype = getattr(test_idx, "dtype", None)
    if dtype is not None and getattr(dtype, "kind", None) not in ("i", "u"):
        raise TypeError("test_idx must have integer dtype")
    try:
        indices = tuple(_integer(value, "test_idx entry", minimum=0) for value in test_idx)
    except TypeError as error:
        raise TypeError("test_idx must be a one-dimensional integer sequence") from error
    if n is not None and _integer(n, "n", minimum=0) != len(indices):
        raise ValueError("test_idx count does not match n")
    if len(set(indices)) != len(indices):
        raise ValueError("test_idx entries must be unique")
    return tuple((namespace, index) for index in indices)


def _rows(x: Iterable[Iterable[float]], name: str) -> tuple[tuple[float, ...], ...]:
    if getattr(getattr(x, "dtype", None), "kind", None) == "c":
        raise ValueError(f"{name} must be real-valued")
    if getattr(x, "ndim", 2) != 2:
        raise ValueError(f"{name} must have shape (n>=2, d>=1)")
    try:
        rows = tuple(tuple(float(value) for value in row) for row in x)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite rectangular numeric matrix") from error
    if len(rows) < 2 or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f"{name} must have shape (n>=2, d>=1)")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError(f"{name} contains nonfinite coordinates")
    return rows


def _native_bool(value: object) -> bool:
    # Explicit scalar bool, never numeric truthiness; no NumPy import.
    if isinstance(value, bool):
        return value
    kind = type(value)
    if kind.__module__ == "numpy" and kind.__name__ in ("bool", "bool_"):
        return bool(value)
    raise TypeError("already_simplex must be an explicit boolean scalar")


def _aligned_ids(reference_ids: Iterable[CellID], other_ids: Iterable[CellID], n: int):
    reference_ids = validate_cell_ids(reference_ids, n)
    other_ids = validate_cell_ids(other_ids, n)
    if reference_ids != other_ids:
        raise ValueError("reference and recipient identities must be exactly equal in order")
    return reference_ids


def _constant(rows: Sequence[Sequence[float]]) -> bool:
    return all(tuple(row) == tuple(rows[0]) for row in rows[1:])


def canonical_edge_hash(
    graph: Sequence[Sequence[int]], cell_ids: Sequence[CellID]
) -> str:
    """Oracle's own canonical JSON encoding of undirected logical-ID edges.

    The production contract promises a canonical edge hash but does not freeze
    its byte encoding.  Tests compare its invariance and the underlying ID edge
    set, not this oracle-specific serialization to a production digest.
    """
    edges = sorted(
        tuple(sorted((cell_ids[i], cell_ids[j])))
        for i, neighbours in enumerate(graph)
        for j in neighbours
        if i < j
    )
    return sha256(json.dumps(edges, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OracleGraph:
    graph: tuple[tuple[int, ...], ...]
    directed: tuple[tuple[int, ...], ...]
    cell_ids: tuple[CellID, ...]
    squared_distances: tuple[tuple[float, ...], ...]
    metadata: dict[str, object]


def build_knn_graph(x, k: int, *, cell_ids: Iterable[CellID]) -> OracleGraph:
    """Exact non-self neighbours, sorted by (dense squared distance, CellID)."""
    rows = _rows(x, "embedding")
    k = _integer(k, "k", minimum=1)
    ids = validate_cell_ids(cell_ids, len(rows))
    effective_k = min(k, len(rows) - 1)
    distances = tuple(
        tuple(sum((left - right) * (left - right) for left, right in zip(a, b)) for b in rows)
        for a in rows
    )
    if any(not math.isfinite(value) for row in distances for value in row):
        raise ValueError("distance calculation overflowed; rescale inputs explicitly")
    directed = []
    cutoff_tie_rows = 0
    boundary_tie_rows = 0
    for i, row in enumerate(distances):
        ordered = sorted((j for j in range(len(rows)) if j != i), key=lambda j: (row[j], ids[j]))
        selected = tuple(ordered[:effective_k])
        directed.append(selected)
        cutoff = row[selected[-1]]
        cutoff_tie_rows += sum(row[j] == cutoff for j in ordered) > 1
        boundary_tie_rows += effective_k < len(ordered) and row[ordered[effective_k]] == cutoff
    union = [set(neighbours) for neighbours in directed]
    for i, neighbours in enumerate(directed):
        for j in neighbours:
            union[j].add(i)
    graph = tuple(tuple(sorted(neighbours)) for neighbours in union)
    constant = _constant(rows)
    return OracleGraph(
        graph=graph,
        directed=tuple(directed),
        cell_ids=ids,
        squared_distances=distances,
        metadata={
            "effective_k": effective_k,
            "graph_status": "noninformative_constant" if constant else "eligible",
            "usable_geometry": not constant,
            "cutoff_tie_rows": cutoff_tie_rows,
            "cutoff_boundary_tie_rows": boundary_tie_rows,
            "input_dtype": str(getattr(x, "dtype", "python_numeric")),
            "distance_dtype": "float64",
            "canonical_edge_hash": canonical_edge_hash(graph, ids),
        },
    )


def oracle_graph(x, k: int, cell_ids: Iterable[CellID]):
    """Driver API: return (directed, union) as row-index adjacency tuples.

    Each directed row has exactly effective-k nonself indices; each union row
    contains its sorted undirected neighbours.  Use build_knn_graph when graph
    support, tie metadata, distances, or logical-ID hashes are also needed.
    """
    graph = build_knn_graph(x, k, cell_ids=cell_ids)
    return graph.directed, graph.graph


def hop_distances(graph: Sequence[Sequence[int]]) -> tuple[tuple[float, ...], ...]:
    result = []
    for source in range(len(graph)):
        distances = [math.inf] * len(graph)
        distances[source] = 0.0
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for neighbour in graph[node]:
                if math.isinf(distances[neighbour]):
                    distances[neighbour] = distances[node] + 1.0
                    queue.append(neighbour)
        result.append(tuple(distances))
    return tuple(result)


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def raag_battery(
    reference, latent, *, reference_ids, candidate_ids, k=15,
    far_quantile=0.9, already_simplex: bool,
) -> dict[str, object]:
    """Original hop metrics with independent geometry-eligibility decisions."""
    already_simplex = _native_bool(already_simplex)
    reference_rows = _rows(reference, "reference")
    candidate_rows = _rows(latent, "latent")
    if len(reference_rows) != len(candidate_rows):
        raise ValueError("reference and candidate row counts differ")
    ids = _aligned_ids(reference_ids, candidate_ids, len(reference_rows))
    if not math.isfinite(float(far_quantile)) or not 0.5 <= far_quantile < 1.0:
        raise ValueError("far_quantile must lie in [0.5, 1.0)")
    ref_graph = build_knn_graph(reference, k, cell_ids=ids)
    candidate_graph = build_knn_graph(latent, k, cell_ids=ids)
    result = {key: None for key in GEOMETRY_KEYS}
    result.update({
        "geometry_status": "eligible",
        "reference_graph_metadata": ref_graph.metadata,
        "candidate_graph_metadata": candidate_graph.metadata,
    })
    if not ref_graph.metadata["usable_geometry"]:
        result["geometry_status"] = "reference_noninformative"
        return result
    if not candidate_graph.metadata["usable_geometry"]:
        result["geometry_status"] = "candidate_noninformative"
        return result
    ref_hops = hop_distances(ref_graph.graph)
    candidate_hops = hop_distances(candidate_graph.graph)
    pairs = [(i, j) for i in range(len(ids)) for j in range(i + 1, len(ids))]
    connected = [(i, j) for i, j in pairs if math.isfinite(ref_hops[i][j])]
    result["n_reference_connected_pairs"] = len(connected)
    if len(connected) < MINIMUM_PAIRS:
        result["geometry_status"] = "insufficient_reference_pairs"
        return result
    edges = [(i, j) for i, neighbours in enumerate(ref_graph.graph) for j in neighbours if i < j]
    edge_hops = [candidate_hops[i][j] for i, j in edges]
    result.update({
        "reachability_survival": sum(math.isfinite(candidate_hops[i][j]) for i, j in connected) / len(connected),
        "n_reference_connected_pairs": float(len(connected)),
        "edge_survival": sum(1.0 / h if math.isfinite(h) else 0.0 for h in edge_hops) / len(edges),
        "edge_severed_fraction": sum(math.isinf(h) for h in edge_hops) / len(edges),
        "edge_preserved_fraction": sum(h == 1.0 for h in edge_hops) / len(edges),
        "n_reference_edges": float(len(edges)),
    })
    cutoff = _quantile([ref_hops[i][j] for i, j in connected], far_quantile)
    far = [(i, j) for i, j in connected if ref_hops[i][j] >= cutoff]
    result["n_far_pairs"] = len(far)
    if len(far) < MINIMUM_PAIRS:
        result["geometry_status"] = "insufficient_far_pairs"
        return result
    candidate_finite = [candidate_hops[i][j] for i, j in pairs if math.isfinite(candidate_hops[i][j])]
    result["n_far_pairs"] = float(len(far))
    result["separation_far_cutoff_hops"] = cutoff
    if not candidate_finite:
        result["separation_survival"] = 0.0
        return result
    threshold = _quantile(candidate_finite, 0.5)
    result["separation_threshold_hops"] = threshold
    result["separation_survival"] = sum(
        math.isinf(candidate_hops[i][j]) or candidate_hops[i][j] >= threshold for i, j in far
    ) / len(far)
    return result


def _decimal(value: float | Decimal) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal.from_float(float(value))


def jensen_shannon_decimal(p, q) -> Decimal:
    """Classical KL-to-mixture definition; no production series or kernels."""
    if len(p) != len(q) or len(p) == 0:
        raise ValueError("JSD arguments must have equal nonzero dimension")
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        result = Decimal(0)
        for left, right in zip(p, q):
            left, right = _decimal(left), _decimal(right)
            if not left.is_finite() or not right.is_finite() or left < 0 or right < 0:
                raise ValueError("JSD arguments must be finite and nonnegative")
            mixture = (left + right) / 2
            if left:
                result += left * (left / mixture).ln() / 2
            if right:
                result += right * (right / mixture).ln() / 2
        return +result


def decimal_jsd(p, q) -> Decimal:
    """Driver API: defining JSD in nats at Decimal precision 80.

    Delivered floats are converted with Decimal.from_float.  Existing Decimal
    coordinates are retained, allowing the driver to check the ideal rational
    scaling identity separately from floored/normalised delivered inputs.
    """
    return jensen_shannon_decimal(p, q)


def measured_simplex_decimal(latent, *, already_simplex: bool):
    """Return Decimal input/measured rows and the native flooring count."""
    already_simplex = _native_bool(already_simplex)
    rows = _rows(latent, "latent")
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        floor = Decimal.from_float(PROBABILITY_FLOOR)
        input_rows, measured_rows = [], []
        flooring_entries = 0
        for row in rows:
            if already_simplex:
                if any(value < 0 for value in row) or abs(sum(row) - 1.0) > 1e-4:
                    raise ValueError("native rows must be nonnegative and sum to one within atol=1e-4, rtol=0")
                probabilities = tuple(Decimal.from_float(value) for value in row)
                floored = tuple(max(value, floor) for value in probabilities)
                flooring_entries += sum(value < floor for value in probabilities)
            else:
                coordinates = tuple(Decimal.from_float(value) for value in row)
                maximum = max(coordinates)
                exponentials = tuple((value - maximum).exp() for value in coordinates)
                total = sum(exponentials, Decimal(0))
                probabilities = tuple(value / total for value in exponentials)
                floored = probabilities
            total = sum(floored, Decimal(0))
            input_rows.append(probabilities)
            measured_rows.append(tuple(value / total for value in floored))
        return tuple(input_rows), tuple(measured_rows), flooring_entries


def decimal_normalize(p):
    """Driver API: (input_decimal_rows, measured_decimal_rows, floor_count).

    Inputs are delivered native floats, converted exactly by Decimal.from_float.
    Entries below Decimal.from_float(float(1e-12)) are floored and each row is
    normalised in precision 80.  Input rows are preserved before that operation;
    the count counts strictly-below-floor entries, not equality or roundoff.
    """
    return measured_simplex_decimal(p, already_simplex=True)


def _allocation_summary(probabilities, union, *, supported_reference: bool):
    if type(supported_reference) is not bool:
        raise TypeError("supported_reference must be a native bool")
    n = len(probabilities)
    if len(union) != n:
        raise ValueError("union adjacency and probability rows must have equal counts")
    neighbours = []
    for i, row in enumerate(union):
        row = tuple(_integer(j, "union neighbour", minimum=0) for j in row)
        if len(set(row)) != len(row) or any(j >= n or j == i for j in row):
            raise ValueError("union adjacency must be unique, nonself and in range")
        neighbours.append(set(row))
    if any(i not in neighbours[j] for i, row in enumerate(neighbours) for j in row):
        raise ValueError("union adjacency must be symmetric")
    edges = [(i, j) for i, row in enumerate(neighbours) for j in sorted(row) if i < j]
    supported = supported_reference and bool(edges)
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        pair_js = sum((jensen_shannon_decimal(probabilities[i], probabilities[j]) for i, j in pairs), Decimal(0)) / len(pairs)
        pair_tv = sum(
            (sum((abs(left - right) for left, right in zip(probabilities[i], probabilities[j])), Decimal(0)) / 2 for i, j in pairs),
            Decimal(0),
        ) / len(pairs)
        dimension = len(probabilities[0])
        maximum_range = max(
            max(row[j] for row in probabilities) - min(row[j] for row in probabilities) for j in range(dimension)
        )
        perplexity = sum(
            ((-sum((value * value.ln() for value in row if value), Decimal(0))).exp() for row in probabilities),
            Decimal(0),
        ) / n
        fraction = perplexity / dimension
        edge_js = (
            sum((jensen_shannon_decimal(probabilities[i], probabilities[j]) for i, j in edges), Decimal(0)) / len(edges)
            if supported else None
        )
        degenerate = pair_js <= Decimal.from_float(DENOMINATOR_THRESHOLD)
        near_uniform = fraction > Decimal.from_float(NEAR_UNIFORM_FRACTION)
        relative = Decimal(1) - edge_js / pair_js if edge_js is not None and pair_js > 0 else None
        raw = (Decimal(0) if degenerate else max(Decimal(0), min(Decimal(1), relative))) if supported else None
        status = "unsupported_reference" if not supported else ("degenerate" if degenerate else ("near_uniform" if near_uniform else "eligible"))
        return {
            "status": status,
            "degenerate": degenerate,
            "near_uniform": near_uniform,
            "coherence": raw if status == "eligible" else None,
            "raw_coherence": raw,
            "relative_ratio": relative,
            "edge_js": edge_js,
            "pair_js": pair_js,
            "mean_pair_tv": pair_tv,
            "max_coordinate_range": maximum_range,
            "mean_perplexity": perplexity,
            "perplexity_fraction": fraction,
        }


def _float_summary(summary):
    return {key: float(value) if isinstance(value, Decimal) else value for key, value in summary.items()}


def oracle_allocation(p, union, *, supported_reference: bool = True) -> dict[str, object]:
    """Driver API: native measured-input diagnostics, without production imports.

    union is the row-index adjacency returned by oracle_graph.  For constant
    reference coordinates the caller must pass supported_reference=False: the
    deterministic constant kNN graph alone cannot disclose geometric support.
    Scalars/lists retain a float convenience view; decimal_summary and
    decimal_measured_probabilities retain the precision-80 values for the
    driver report and tolerance comparison without an intermediate float cast.
    """
    inputs, measured, flooring_entries = decimal_normalize(p)
    decimal_summary = _allocation_summary(measured, union, supported_reference=supported_reference)
    result = _float_summary(decimal_summary)
    result.update({
        "input_probabilities": [[float(value) for value in row] for row in inputs],
        "measured_probabilities": [[float(value) for value in row] for row in measured],
        "flooring_entries": flooring_entries,
        "decimal_summary": decimal_summary,
        "decimal_measured_probabilities": measured,
    })
    return result


def allocation_diagnostics(
    reference, latent, *, reference_ids, allocation_ids, k=15,
    already_simplex: bool, return_probabilities=False,
) -> dict[str, object]:
    """Scorer-shaped convenience API over the same independent definitions."""
    already_simplex = _native_bool(already_simplex)
    reference_rows = _rows(reference, "reference")
    latent_rows = _rows(latent, "latent")
    if len(reference_rows) != len(latent_rows):
        raise ValueError("reference and allocation row counts differ")
    ids = _aligned_ids(reference_ids, allocation_ids, len(reference_rows))
    reference_graph = build_knn_graph(reference, k, cell_ids=ids)
    inputs, probabilities, flooring_entries = measured_simplex_decimal(latent, already_simplex=already_simplex)
    summary = _float_summary(_allocation_summary(
        probabilities, reference_graph.graph,
        supported_reference=bool(reference_graph.metadata["usable_geometry"]),
    ))
    prefix = "allocation_" if already_simplex else "softmax_coordinate_"
    names = {"raw_coherence": "coherence_raw", "relative_ratio": "relative_coherence_unclipped"}
    result = {
        "allocation_status": summary["status"] if already_simplex else "non_native",
        "flooring_entries": flooring_entries,
    }
    result.update({prefix + names.get(key, key): value for key, value in summary.items() if key != "status"})
    if not already_simplex:
        result["allocation_coherence"] = None
        result["allocation_coherence_raw"] = None
        result["softmax_coordinate_dimension"] = len(probabilities[0])
        result["softmax_coordinate_perplexity"] = summary["mean_perplexity"]
    if return_probabilities:
        result["input_probabilities"] = [[float(value) for value in row] for row in inputs]
        result["measured_probabilities"] = [[float(value) for value in row] for row in probabilities]
    return result
