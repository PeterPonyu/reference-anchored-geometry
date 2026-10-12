"""Reference-Anchored Allocation Geometry (RAAG).

A purpose-built battery for allocation models -- topic models on the simplex
and mixture models with soft responsibilities -- evaluated against an external
expression reference.

Scope
-----
These diagnostics compare coordinate geometry to a selected expression
reference. They do not identify biological truth. Local rank metrics can be
weakly sensitive to a graph disconnection in the controlled Y fixture; the
matched-noise calibration reports continuous changes, not universal failure.
Official scIB metrics are separately evaluated using their annotation inputs.

The geometry quantities here are deterministic, label-free, and anchored to the
reference graph rather than to a clustering of it:

``reachability_survival``
    Do cells connected in the reference stay connected in the candidate?
    Detects tearing, which is what the rank-local family misses.  It is
    measured over pairs rather than edges: a cut that severs two percent of
    edges makes two thirds of pairs unreachable, and only the pair-level view
    shows that.  Calibration caught the edge-level version scoring a shattered
    manifold *above* a faithful one, for the same averaging reason
    trustworthiness does.

``edge_survival``
    How far apart reference neighbours drift, in hops.  This is a local
    distortion measure and explicitly *not* a tear detector; it is kept
    because it separates noise and permutation, which reachability cannot.

``separation_survival``
    Do cells that are far apart in the reference stay far apart?  Detects
    collapse, the complementary failure.

``allocation_coherence``
    Does the allocation itself vary smoothly along the reference manifold,
    measured with Jensen-Shannon divergence on the simplex rather than
    Euclidean distance?  This is the model-native question, and Euclidean
    metrics answer it with the wrong geometry.

    Coherence is a ratio of two divergences, so it stays numerically high
    when every row is near-uniform and both divergences are tiny.  The
    ``allocation_near_uniform`` flag (mean row perplexity above
    ``NEAR_UNIFORM_PERPLEXITY_FRACTION`` of the component count) marks that
    case; such cells print ``n.u.`` rather than a coherence value.

Version 2 uses an exact distinct-pair JSD denominator and reports non-native
softmax coordinates separately from allocations. Legacy sampled estimates remain
available through ``legacy_allocation_coherence``.

The quantities are reported separately.  Collapsing them into one number would
reproduce exactly the failure this battery exists to avoid: a single score
cannot distinguish a torn embedding from a merged one.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors

__all__ = [
    "NEAR_UNIFORM_PERPLEXITY_FRACTION",
    "knn_hop_graph",
    "reachability_survival",
    "edge_survival",
    "separation_survival",
    "allocation_coherence",
    "legacy_allocation_coherence",
    "exact_pair_js_mean",
    "RAAG_VERSION",
    "raag_battery",
]

#: Mean row perplexity, as a fraction of the component count, above which an
#: allocation is near-uniform and its coherence is not quotable.  Shared with
#: ``audit.score_sweep`` so the scorer and the battery cannot drift apart.
NEAR_UNIFORM_PERPLEXITY_FRACTION = 0.95
RAAG_VERSION = "RAAG-v2-exact-20260921"
LEGACY_RAAG_VERSION = "RAAG-v1-sampled-20260907"


def _as_array(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 5:
        raise ValueError(f"{name} must have shape (n>=5, d>=1)")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def knn_hop_graph(x: np.ndarray, k: int) -> csr_matrix:
    """Symmetric unweighted kNN graph.

    Hops rather than distances are used throughout the battery so that no
    score can be moved by rescaling an embedding.  Symmetrising with a union
    keeps the graph undirected without inventing edges between mutual
    non-neighbours.
    """

    arr = _as_array(x, "embedding")
    kk = int(min(max(1, k), arr.shape[0] - 1))
    _, indices = NearestNeighbors(n_neighbors=kk + 1).fit(arr).kneighbors(arr)
    rows = np.repeat(np.arange(arr.shape[0]), kk)
    cols = indices[:, 1:].ravel()
    graph = csr_matrix(
        (np.ones(rows.size, dtype=np.int8), (rows, cols)),
        shape=(arr.shape[0], arr.shape[0]),
    )
    return graph.maximum(graph.T)


def _hop_distances(graph: csr_matrix) -> np.ndarray:
    return shortest_path(graph, method="D", unweighted=True, directed=False)


def reachability_survival(
    reference: np.ndarray,
    candidate: np.ndarray,
    k: int = 15,
) -> dict[str, float]:
    """Fraction of reference-connected pairs that stay connected.

    This is the tear detector.  Averaging over *edges* hides a cut, because a
    translation that splits a Y manifold into three pieces severs only the few
    edges at the junctions while leaving every within-arm edge intact; the
    edge-level mean then rates the shattered embedding above the faithful one.
    Counting pairs instead makes the same cut cost two thirds of the score.

    Pairs unreachable in the reference are excluded, so a fragmented reference
    cannot make a candidate look good by lowering the bar.
    """

    reference = _as_array(reference, "reference")
    candidate = _as_array(candidate, "candidate")
    if reference.shape[0] != candidate.shape[0]:
        raise ValueError("reference and candidate must have the same number of rows")

    ref_hops = _hop_distances(knn_hop_graph(reference, k))
    cand_hops = _hop_distances(knn_hop_graph(candidate, k))
    tri = np.triu_indices(reference.shape[0], k=1)
    ref_ok = np.isfinite(ref_hops[tri])
    if ref_ok.sum() < 10:
        raise ValueError("reference graph is too fragmented to score reachability")
    cand_ok = np.isfinite(cand_hops[tri])
    return {
        "reachability_survival": float(np.mean(cand_ok[ref_ok])),
        "n_reference_connected_pairs": float(ref_ok.sum()),
    }


def edge_survival(
    reference: np.ndarray,
    candidate: np.ndarray,
    k: int = 15,
) -> dict[str, float]:
    """Local distortion of reference adjacency, in hops.

    Each reference edge contributes ``1 / hop_candidate``, so an edge that is
    still an edge scores 1, one that needs three hops scores 1/3, and one
    whose endpoints landed in different components scores 0.  The reciprocal
    is deliberate: it saturates, so a candidate cannot buy back a severed edge
    by making other edges shorter than they already are.
    """

    reference = _as_array(reference, "reference")
    candidate = _as_array(candidate, "candidate")
    if reference.shape[0] != candidate.shape[0]:
        raise ValueError("reference and candidate must have the same number of rows")

    g_ref = knn_hop_graph(reference, k)
    g_cand = knn_hop_graph(candidate, k)
    hops = _hop_distances(g_cand)

    src, dst = g_ref.nonzero()
    upper = src < dst
    src, dst = src[upper], dst[upper]
    if src.size == 0:
        raise ValueError("reference graph has no edges")

    h = hops[src, dst]
    finite = np.isfinite(h)
    survival = np.zeros(h.shape, dtype=float)
    survival[finite] = 1.0 / np.maximum(h[finite], 1.0)
    return {
        "edge_survival": float(np.mean(survival)),
        "edge_severed_fraction": float(np.mean(~finite)),
        "edge_preserved_fraction": float(np.mean(finite & (h <= 1.0))),
        "n_reference_edges": float(src.size),
    }


def separation_survival(
    reference: np.ndarray,
    candidate: np.ndarray,
    k: int = 15,
    far_quantile: float = 0.9,
) -> dict[str, float]:
    """Whether cells far apart in the reference stay far apart.

    The far set is the upper ``far_quantile`` of finite reference hop
    distances.  A pair survives if its candidate hop distance is at least the
    candidate's own median hop distance, which keeps the comparison free of
    any absolute scale.  Pairs that become unreachable count as survived: an
    over-separated candidate is penalised by ``edge_survival``, not here.
    """

    reference = _as_array(reference, "reference")
    candidate = _as_array(candidate, "candidate")
    if reference.shape[0] != candidate.shape[0]:
        raise ValueError("reference and candidate must have the same number of rows")
    if not 0.5 <= far_quantile < 1.0:
        raise ValueError("far_quantile must lie in [0.5, 1.0)")

    ref_hops = _hop_distances(knn_hop_graph(reference, k))
    cand_hops = _hop_distances(knn_hop_graph(candidate, k))

    tri = np.triu_indices(reference.shape[0], k=1)
    rh = ref_hops[tri]
    ch = cand_hops[tri]
    ref_finite = np.isfinite(rh)
    if ref_finite.sum() < 10:
        raise ValueError("reference graph is too fragmented to define a far set")

    cutoff = float(np.quantile(rh[ref_finite], far_quantile))
    far = ref_finite & (rh >= cutoff)
    if far.sum() < 10:
        raise ValueError("far set is too small; lower far_quantile")

    cand_finite = np.isfinite(ch)
    if cand_finite.sum() == 0:
        return {"separation_survival": 0.0, "n_far_pairs": float(far.sum())}
    threshold = float(np.median(ch[cand_finite]))

    far_candidate = ch[far]
    survived = ~np.isfinite(far_candidate) | (far_candidate >= threshold)
    return {
        "separation_survival": float(np.mean(survived)),
        "separation_far_cutoff_hops": cutoff,
        "separation_threshold_hops": threshold,
        "n_far_pairs": float(far.sum()),
    }


def _jensen_shannon(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Row-wise Jensen-Shannon divergence in nats, for simplex rows."""

    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = np.where(p > 0, p * (np.log(p) - np.log(m)), 0.0).sum(axis=1)
        kl_qm = np.where(q > 0, q * (np.log(q) - np.log(m)), 0.0).sum(axis=1)
    return 0.5 * (kl_pm + kl_qm)


def to_simplex(latent: np.ndarray, already_simplex: bool | None = None) -> np.ndarray:
    """Return a row-stochastic view of an allocation latent.

    Topic models already emit points on the simplex.  Mixture models emit an
    unconstrained latent, which is mapped with a softmax so the same divergence
    applies; that mapping is a convenience, not a claim that the coordinates
    were probabilities to begin with.
    """

    # No minimum row count here: this is a row-wise transform, not a graph
    # construction, and it is useful on single rows.
    z = np.asarray(latent, dtype=float)
    if z.ndim != 2 or z.shape[1] < 1:
        raise ValueError("latent must have shape (n, d>=1)")
    if not np.isfinite(z).all():
        raise ValueError("latent contains NaN or infinite values")
    if already_simplex is None:
        row_sums = z.sum(axis=1)
        already_simplex = bool(
            np.all(z >= -1e-9) and np.allclose(row_sums, 1.0, atol=1e-4)
        )
    if already_simplex:
        p = np.clip(z, 1e-12, None)
    else:
        shifted = z - z.max(axis=1, keepdims=True)
        p = np.exp(shifted)
    return p / p.sum(axis=1, keepdims=True)


def legacy_allocation_coherence(
    reference: np.ndarray,
    latent: np.ndarray,
    k: int = 15,
    n_random_pairs: int = 20000,
    random_state: int = 42,
    already_simplex: bool | None = None,
) -> dict[str, float]:
    """Contrast of allocation divergence across reference edges vs random pairs.

    A value near one means neighbouring cells share an allocation while distant
    cells do not.  Zero means the allocation carries no more information about
    the reference manifold than a random pairing would, which is what a
    posterior-collapsed or fully diffuse representation looks like.

    Coherence is a ratio, so a near-uniform allocation can still score high;
    ``allocation_near_uniform`` flags that case from the mean row perplexity
    and callers should not quote coherence when it is set.
    """

    reference = _as_array(reference, "reference")
    p = to_simplex(latent, already_simplex=already_simplex)
    if reference.shape[0] != p.shape[0]:
        raise ValueError("reference and latent must have the same number of rows")

    # Perplexity is the effective number of components a row spreads over;
    # its mean as a fraction of K is what the near-uniform flag reads.
    with np.errstate(divide="ignore", invalid="ignore"):
        row_entropy = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=1)
    mean_perplexity = float(np.mean(np.exp(row_entropy)))
    perplexity_fraction = mean_perplexity / p.shape[1]
    near_uniform = {
        "allocation_mean_perplexity": mean_perplexity,
        "allocation_perplexity_fraction": float(perplexity_fraction),
        "allocation_near_uniform": bool(
            perplexity_fraction > NEAR_UNIFORM_PERPLEXITY_FRACTION
        ),
    }

    src, dst = knn_hop_graph(reference, k).nonzero()
    upper = src < dst
    src, dst = src[upper], dst[upper]
    if src.size == 0:
        raise ValueError("reference graph has no edges")
    edge_js = _jensen_shannon(p[src], p[dst])

    rng = np.random.default_rng(random_state)
    n = p.shape[0]
    size = int(min(n_random_pairs, n * (n - 1) // 2))
    a = rng.integers(0, n, size=size)
    b = rng.integers(0, n, size=size)
    keep = a != b
    random_js = _jensen_shannon(p[a[keep]], p[b[keep]])

    baseline = float(np.mean(random_js))
    if baseline <= 1e-12:
        # Every cell carries the same allocation, so there is no contrast to
        # measure.  Reporting zero is correct and keeps the score bounded.
        return {
            "allocation_coherence": 0.0,
            "allocation_edge_js": float(np.mean(edge_js)),
            "allocation_random_js": baseline,
            "allocation_degenerate": True,
            **near_uniform,
        }
    return {
        "allocation_coherence": float(
            np.clip(1.0 - float(np.mean(edge_js)) / baseline, 0.0, 1.0)
        ),
        "allocation_edge_js": float(np.mean(edge_js)),
        "allocation_random_js": baseline,
        "allocation_degenerate": False,
        **near_uniform,
    }


def exact_pair_js_mean(p: np.ndarray) -> float:
    """Exact mean JSD of all unordered distinct simplex pairs, bounded memory.

    The denominator depends on cell identities, not sampled positional indices.
    Floating summation can change at machine precision after joint reordering.
    """
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or len(p) < 2 or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("finite nonnegative simplex rows required")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("simplex rows must sum to one")
    total = 0.0
    for i in range(len(p) - 1):
        left = np.broadcast_to(p[i], p[i + 1:].shape)
        total += float(_jensen_shannon(left, p[i + 1:]).sum())
    return total / (len(p) * (len(p) - 1) // 2)


def allocation_coherence(
    reference: np.ndarray,
    latent: np.ndarray,
    k: int = 15,
    n_random_pairs: int = 20000,
    random_state: int = 42,
    already_simplex: bool | None = None,
) -> dict[str, object]:
    """Version 2: exact denominator and explicit native-allocation eligibility.

    Legacy RNG arguments remain accepted for call compatibility; v2 does not
    sample. ``legacy_allocation_coherence`` reproduces the archived estimator.
    A non-native latent is never promoted to an allocation by softmax: its
    transformed-coordinate diagnostic has separate keys and its own dimension.
    Guarded native raw values remain available for calibration, while the
    quotable ``allocation_coherence`` is NaN with an explicit status.
    """
    reference = _as_array(reference, "reference")
    z = _as_array(latent, "latent")
    if reference.shape[0] != z.shape[0]:
        raise ValueError("reference and latent must have the same number of rows")
    is_stochastic = bool(np.all(z >= -1e-9) and np.allclose(z.sum(axis=1), 1.0, atol=1e-4))
    native = is_stochastic if already_simplex is None else bool(already_simplex)
    if native and not is_stochastic:
        raise ValueError("declared native allocation is not row-stochastic")
    p = to_simplex(z, already_simplex=native)
    src, dst = knn_hop_graph(reference, k).nonzero()
    upper = src < dst
    edge_js = float(np.mean(_jensen_shannon(p[src[upper]], p[dst[upper]])))
    baseline = exact_pair_js_mean(p)
    mean_perplexity = float(np.mean(np.exp(-np.sum(p * np.log(p), axis=1))))
    fraction = mean_perplexity / p.shape[1]
    degenerate = bool(baseline <= 1e-12)
    near_uniform = bool(fraction > NEAR_UNIFORM_PERPLEXITY_FRACTION)
    raw = 0.0 if degenerate else float(np.clip(1.0 - edge_js / baseline, 0.0, 1.0))
    result: dict[str, object] = {
        "raag_version": RAAG_VERSION,
        "coherence_estimator": "exact_unordered_distinct_pairs",
        "coherence_denominator_pairs": len(p) * (len(p) - 1) // 2,
        "allocation_is_native_simplex": native,
        "allocation_identity_source": "numeric_inference" if already_simplex is None else "caller_declared",
        "allocation_dimension": int(p.shape[1]) if native else None,
        "allocation_status": "non_native",
        "allocation_coherence": float("nan"),
        "allocation_coherence_raw": float("nan"),
        "allocation_mean_perplexity": float("nan"),
        "allocation_perplexity_mean": float("nan"),
        "allocation_perplexity_fraction": float("nan"),
        "allocation_edge_js": float("nan"),
        "allocation_random_js": float("nan"),
        "allocation_degenerate": False,
        "allocation_near_uniform": False,
    }
    if native:
        status = "degenerate" if degenerate else ("near_uniform" if near_uniform else "valid")
        result.update({
            "allocation_status": status,
            "allocation_coherence": raw if status == "valid" else float("nan"),
            "allocation_coherence_raw": raw,
            "allocation_mean_perplexity": mean_perplexity,
            "allocation_perplexity_mean": mean_perplexity,
            "allocation_perplexity_fraction": fraction,
            "allocation_edge_js": edge_js,
            "allocation_random_js": baseline,
            "allocation_degenerate": degenerate,
            "allocation_near_uniform": near_uniform,
        })
    else:
        result.update({
            "softmax_coordinate_coherence": raw,
            "softmax_coordinate_perplexity": mean_perplexity,
            "softmax_coordinate_dimension": int(p.shape[1]),
            "softmax_coordinate_perplexity_fraction": fraction,
            "softmax_coordinate_edge_js": edge_js,
            "softmax_coordinate_pair_js": baseline,
            "softmax_coordinate_degenerate": degenerate,
            "softmax_coordinate_near_uniform": near_uniform,
        })
    return result


def raag_battery(
    reference: np.ndarray,
    latent: np.ndarray,
    k: int = 15,
    far_quantile: float = 0.9,
    random_state: int = 42,
    already_simplex: bool | None = None,
) -> dict[str, object]:
    """Run unchanged geometry and v2 allocation diagnostics with status metadata."""

    out: dict[str, object] = {}
    out.update(reachability_survival(reference, latent, k=k))
    out.update(edge_survival(reference, latent, k=k))
    out.update(separation_survival(reference, latent, k=k, far_quantile=far_quantile))
    out.update(
        allocation_coherence(
            reference,
            latent,
            k=k,
            random_state=random_state,
            already_simplex=already_simplex,
        )
    )
    return out


def summarize(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    """Mean of each RAAG quantity across a collection of runs."""

    rows = list(rows)
    keys = (
        "reachability_survival",
        "edge_survival",
        "separation_survival",
        "allocation_coherence",
    )
    return {
        key: float(np.mean([r[key] for r in rows if key in r]))
        for key in keys
        if any(key in r for r in rows)
    }
