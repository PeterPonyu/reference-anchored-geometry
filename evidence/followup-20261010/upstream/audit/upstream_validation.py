"""Compare metric implementations on a matched-noise synthetic fixture.

Canonical, project-local and upstream implementations retain distinct identities.
Continuous responses describe this fixture; direction checks are regression
checks, not universal topology tests or statistical-power certificates.
The official scIB APIs use supplied labels/pseudotime and a candidate graph.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Callable

import numpy as np

from .reference_metrics import graph_geodesic_metrics, standard_dre_metrics

# ---------------------------------------------------------------------------
# Ground truth and controlled distortions
# ---------------------------------------------------------------------------


def make_reference_manifold(n_per_arm: int = 150, seed: int = 42) -> dict[str, np.ndarray]:
    """Return a connected Y manifold with pseudotime and branch labels.

    The trunk runs from ``t=0`` to ``t=0.5``; two arms then separate while
    pseudotime continues to ``t=1``.  Branch labels are exposed because the
    upstream ``scib`` connectivity metric is defined per label group.
    """

    n_per_arm = int(n_per_arm)
    if n_per_arm < 20:
        raise ValueError("n_per_arm must be at least 20")
    rng = np.random.default_rng(seed)

    trunk_t = np.linspace(0.0, 0.5, n_per_arm, endpoint=False)
    arm_t = np.linspace(0.0, 1.0, n_per_arm)
    trunk = np.c_[trunk_t, np.zeros_like(trunk_t)]
    arm_a = np.c_[0.5 + 0.5 * arm_t, 0.55 * arm_t]
    arm_b = np.c_[0.5 + 0.5 * arm_t, -0.55 * arm_t]

    coords = np.vstack([trunk, arm_a, arm_b])
    coords = coords + rng.normal(scale=0.004, size=coords.shape)
    pseudotime = np.r_[trunk_t, 0.5 + 0.5 * arm_t, 0.5 + 0.5 * arm_t]
    branch = np.r_[
        np.zeros(n_per_arm, dtype=int),
        np.ones(n_per_arm, dtype=int),
        np.full(n_per_arm, 2, dtype=int),
    ]
    return {"coords": coords, "pseudotime": pseudotime, "branch": branch}


def make_candidates_legacy(
    reference: dict[str, np.ndarray], seed: int = 42
) -> dict[str, dict[str, object]]:
    """Return candidate embeddings ordered from faithful to destroyed.

    ``expected`` records the qualitative verdict any competent topology metric
    must reach.  ``shatter`` keeps every local neighbourhood intact but breaks
    the manifold into three pieces, and ``collapse`` keeps the manifold
    connected while deleting the bifurcation.  Together they separate metrics
    that track topology from metrics that track local density or a single
    dominant axis.
    """

    rng = np.random.default_rng(seed)
    coords = reference["coords"]
    branch = reference["branch"]
    n = coords.shape[0]

    faithful = coords + rng.normal(scale=0.01, size=coords.shape)

    # Applied to ``faithful`` rather than ``coords`` so the only difference
    # from the faithful candidate is the similarity transform itself.
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    rotated = faithful @ rotation.T * 3.7 + np.array([12.0, -4.0])

    # A 2.5 unit translation is roughly 350x the local point spacing, so the
    # kNN graph is decisively split into three components while the panel
    # still renders legibly at equal aspect.
    shatter = coords.copy()
    shatter[branch == 1] += np.array([0.0, 2.5])
    shatter[branch == 2] += np.array([0.0, -2.5])

    collapse = np.c_[coords[:, 0], 0.002 * rng.normal(size=n)]

    noisy = coords + rng.normal(scale=0.18, size=coords.shape)

    permuted = coords[rng.permutation(n)]

    return {
        "faithful": {
            "embedding": faithful,
            "expected": "high",
            "note": "isotropic jitter only; topology and branching preserved",
        },
        "rotated_scaled": {
            "embedding": rotated,
            "expected": "high",
            "note": "rigid rotation plus uniform scale; must be scale invariant",
        },
        "noisy": {
            "embedding": noisy,
            "expected": "medium",
            "note": "heavy jitter blurs the fork but keeps global ordering",
        },
        "collapse": {
            "embedding": collapse,
            "expected": "low",
            "note": "bifurcation deleted by projecting onto the trunk axis",
        },
        "shatter": {
            "embedding": shatter,
            "expected": "low",
            "note": "arms translated apart; three disconnected components",
        },
        "permuted": {
            "embedding": permuted,
            "expected": "floor",
            "note": "row permutation destroys every correspondence",
        },
    }


def make_candidates(
    reference: dict[str, np.ndarray], seed: int = 42,
    shatter_amplitude: float = 2.5, collapse_scale: float = 0.0,
    noise_sd: float = 0.18,
) -> dict[str, dict[str, object]]:
    """Matched-noise v2 ladder; every intervention starts from faithful.

    The archived fixture is available explicitly as ``make_candidates_legacy``.
    The reference is a noisy 2D Y, and faithful adds small coordinate jitter;
    it is not an isometric copy. Removing this jitter in just one damaged arm
    confounds the intended intervention and is forbidden in this v2 fixture.
    """
    rng = np.random.default_rng(seed)
    faithful = reference["coords"] + rng.normal(scale=0.01, size=reference["coords"].shape)
    branch = reference["branch"]
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    shatter = faithful.copy()
    shatter[branch == 1] += [0.0, shatter_amplitude]
    shatter[branch == 2] += [0.0, -shatter_amplitude]
    collapse = faithful.copy()
    collapse[:, 1] *= collapse_scale
    candidates = {
        "faithful": faithful,
        "rotated_scaled": faithful @ rotation.T * 3.7 + [12.0, -4.0],
        "noisy": faithful + rng.normal(scale=noise_sd, size=faithful.shape),
        "collapse": collapse,
        "shatter": shatter,
        "permuted": faithful[rng.permutation(len(faithful))],
    }
    return {name: {"embedding": arr, "expected": "continuous response; no power certificate",
                   "note": "v2 matched faithful noise; " + name}
            for name, arr in candidates.items()}


def official_scib_metrics(low: np.ndarray, labels: np.ndarray,
                          pseudotime: np.ndarray, k: int = 15,
                          random_state: int = 42) -> dict[str, object]:
    """Call both public scIB APIs; retain metric-specific errors and warnings.

    Input pseudotime and labels are known synthetic annotations. Candidate
    neighbors are computed by Scanpy. The trajectory API constructs diffusion
    pseudotime internally; the local root-geodesic surrogate is never used here.
    """
    import anndata as ad
    import pandas as pd
    import scanpy as sc
    from scib.metrics import graph_connectivity, trajectory_conservation

    obs = pd.DataFrame({"branch": pd.Categorical(labels.astype(str)),
                        "known_pseudotime": np.asarray(pseudotime, float)},
                       index=[f"c{i}" for i in range(len(low))])
    pre = ad.AnnData(X=np.zeros((len(low), 1), dtype=np.float32), obs=obs.copy())
    post = ad.AnnData(X=np.zeros((len(low), 1), dtype=np.float32), obs=obs.copy())
    post.obsm["X_emb"] = np.asarray(low, dtype=np.float32)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        sc.pp.neighbors(post, n_neighbors=k, use_rep="X_emb", random_state=random_state)
    result: dict[str, object] = {"neighbor_warnings": sorted({str(w.message) for w in captured})}
    for name, call in (
        ("scib_graph_connectivity", lambda: graph_connectivity(post.copy(), label_key="branch")),
        ("scib_trajectory_conservation", lambda: trajectory_conservation(
            pre.copy(), post.copy(), label_key="branch", pseudotime_key="known_pseudotime", batch_key=None)),
    ):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            try:
                value = float(call())
                result[name] = value if np.isfinite(value) else None
                result[name + "_status"] = "ok" if np.isfinite(value) else "nonfinite"
            except Exception as exc:
                result[name] = None
                result[name + "_status"] = "error"
                result[name + "_error"] = f"{type(exc).__name__}: {exc}"
        result[name + "_warnings"] = sorted({str(w.message) for w in captured})
    return result


# ---------------------------------------------------------------------------
# Metric stacks
# ---------------------------------------------------------------------------


def _legacy_metrics(high: np.ndarray, low: np.ndarray, k: int) -> dict[str, float]:
    """Historical ``eval_lib`` battery values."""

    from eval_lib.metrics.dre import evaluate_dimensionality_reduction
    from eval_lib.metrics.drex import evaluate_extended_dimensionality_reduction
    from eval_lib.metrics.lse import evaluate_single_cell_latent_space
    from eval_lib.metrics.lsex import evaluate_extended_latent_space

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dre = evaluate_dimensionality_reduction(high, low, k=k, verbose=False)
        drex = evaluate_extended_dimensionality_reduction(
            high, low, n_neighbors=k, verbose=False
        )
        lse = evaluate_single_cell_latent_space(low, data_type="trajectory", verbose=False)
        lsex = evaluate_extended_latent_space(low, n_neighbors=k, verbose=False)
    return {
        "legacy_DRE_q_local": float(dre.get("Q_local", np.nan)),
        "legacy_DRE_q_global": float(dre.get("Q_global", np.nan)),
        "legacy_DREX_trustworthiness": float(drex.get("trustworthiness", np.nan)),
        "legacy_DREX_continuity": float(drex.get("continuity", np.nan)),
        "legacy_DREX_overall": float(drex.get("overall_quality", np.nan)),
        "legacy_LSE_directionality": float(lse.get("trajectory_directionality", np.nan)),
        "legacy_LSE_overall": float(lse.get("overall_quality", np.nan)),
        "legacy_LSEX_two_hop": float(lsex.get("two_hop_connectivity", np.nan)),
        "legacy_LSEX_overall": float(lsex.get("overall_quality", np.nan)),
    }


def _audit_metrics(
    high: np.ndarray, low: np.ndarray, pseudotime: np.ndarray, k: int
) -> dict[str, float]:
    """Values from the audit's own canonical re-implementation."""

    dre = standard_dre_metrics(high, low, k=k)
    geo = graph_geodesic_metrics(high, low, pseudotime=pseudotime, k=k)
    return {
        "audit_trustworthiness": dre["trustworthiness"],
        "audit_continuity": dre["continuity"],
        "audit_q_local": dre["q_local"],
        "audit_q_global": dre["q_global"],
        "audit_auc_ln_rnx": dre["auc_ln_rnx"],
        "audit_distance_spearman": dre["distance_spearman"],
        "audit_candidate_lcc": geo["candidate_lcc_fraction"],
        "audit_geodesic_spearman": geo["geodesic_pairwise_spearman"],
        "audit_trajectory_conservation": geo.get("trajectory_conservation", float("nan")),
        "audit_trajectory_conservation_penalized": geo.get(
            "trajectory_conservation_penalized", float("nan")
        ),
    }


def _zadu_metrics(high: np.ndarray, low: np.ndarray, k: int, seed: int) -> dict[str, float]:
    """Upstream ``zadu`` values (Jeon et al., IEEE TVCG)."""

    from zadu.measures import local_continuity_meta_criteria as lcmc_mod
    from zadu.measures import spearman_rho as spearman_mod
    from zadu.measures import steadiness_cohesiveness as sc_mod
    from zadu.measures import trustworthiness_continuity as tc_mod

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tc = tc_mod.measure(high, low, k=k)
        lcmc = lcmc_mod.measure(high, low, k=k)
        rho = spearman_mod.measure(high, low)
        sc = sc_mod.measure(high, low, random_state=seed, n_jobs=1)
    return {
        "zadu_trustworthiness": float(tc["trustworthiness"]),
        "zadu_continuity": float(tc["continuity"]),
        "zadu_lcmc": float(lcmc["lcmc"]),
        "zadu_spearman_rho": float(rho["spearman_rho"]),
        "zadu_steadiness": float(sc["steadiness"]),
        "zadu_cohesiveness": float(sc["cohesiveness"]),
    }


def _scib_graph_connectivity(low: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Upstream ``scib`` graph connectivity on the candidate embedding.

    For every label group this is the fraction of that group sitting in the
    largest connected component of the induced kNN subgraph, averaged over
    groups.  It is the published quantity the project's ``two_hop`` score was
    informally standing in for.
    """

    import anndata as ad
    import pandas as pd
    import scanpy as sc
    from scib.metrics import graph_connectivity

    adata = ad.AnnData(
        X=np.zeros((low.shape[0], 1), dtype=np.float32),
        obs=pd.DataFrame(
            {"branch": pd.Categorical([f"branch_{int(v)}" for v in labels])},
            index=[f"c{i}" for i in range(low.shape[0])],
        ),
    )
    adata.obsm["X_emb"] = np.asarray(low, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.pp.neighbors(adata, n_neighbors=k, use_rep="X_emb")
        return float(graph_connectivity(adata, label_key="branch"))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

#: Quantities that all three stacks claim to compute, mapped to their keys.
AGREEMENT_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("trustworthiness", "legacy_DREX_trustworthiness", "zadu_trustworthiness"),
    ("trustworthiness", "audit_trustworthiness", "zadu_trustworthiness"),
    ("continuity", "legacy_DREX_continuity", "zadu_continuity"),
    ("continuity", "audit_continuity", "zadu_continuity"),
    ("distance_spearman", "audit_distance_spearman", "zadu_spearman_rho"),
)


def evaluate_ladder(
    n_per_arm: int = 150, seed: int = 42, k: int = 15
) -> dict[str, object]:
    """Run every metric stack on every candidate and summarise the outcome."""

    reference = make_reference_manifold(n_per_arm=n_per_arm, seed=seed)
    high = reference["coords"]
    pseudotime = reference["pseudotime"]
    branch = reference["branch"]
    candidates = make_candidates(reference, seed=seed)

    rows: list[dict[str, object]] = []
    for name, payload in candidates.items():
        low = np.asarray(payload["embedding"], dtype=float)
        row: dict[str, object] = {
            "candidate": name,
            "expected": payload["expected"],
            "note": payload["note"],
            "n": int(low.shape[0]),
            "k": int(k),
        }
        row.update(_legacy_metrics(high, low, k))
        row.update(_audit_metrics(high, low, pseudotime, k))
        row.update(_zadu_metrics(high, low, k, seed))
        row.update(official_scib_metrics(low, branch, pseudotime, k, seed))
        rows.append(row)

    return {
        "rows": rows,
        "agreement": _summarise_agreement(rows),
        "discrimination": _summarise_discrimination(rows),
        "provenance": {
            "reference": "noisy 2D Y manifold with known synthetic pseudotime and 3 branch labels",
            "fixture_version": "matched-noise-v2",
            "audit_trajectory_conservation_identity": "local root-geodesic surrogate, not official scIB",
            "n_per_arm": int(n_per_arm),
            "seed": int(seed),
            "k": int(k),
            "upstream": _upstream_versions(),
        },
    }


def _upstream_versions() -> dict[str, str]:
    import importlib.metadata as md

    out: dict[str, str] = {}
    for pkg in ("zadu", "scib", "scanpy", "scikit-learn", "numpy", "scipy"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:  # pragma: no cover
            out[pkg] = "not-installed"
    return out


def _summarise_agreement(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Report how far each local implementation sits from the upstream value."""

    out: list[dict[str, object]] = []
    for quantity, local_key, upstream_key in AGREEMENT_PAIRS:
        deltas = []
        for row in rows:
            local = row.get(local_key)
            upstream = row.get(upstream_key)
            if isinstance(local, float) and isinstance(upstream, float):
                if np.isfinite(local) and np.isfinite(upstream):
                    deltas.append(abs(local - upstream))
        out.append(
            {
                "quantity": quantity,
                "local": local_key,
                "upstream": upstream_key,
                "n_compared": len(deltas),
                "max_abs_delta": float(np.max(deltas)) if deltas else float("nan"),
                "mean_abs_delta": float(np.mean(deltas)) if deltas else float("nan"),
            }
        )
    return out


#: Metrics evaluated for topology sensitivity, with orientation.
_DISCRIMINATION_KEYS = (
    "legacy_LSEX_two_hop",
    "legacy_LSEX_overall",
    "legacy_LSE_directionality",
    "legacy_LSE_overall",
    "legacy_DRE_q_local",
    "legacy_DREX_trustworthiness",
    "legacy_DREX_continuity",
    "audit_trustworthiness",
    "audit_continuity",
    "audit_q_local",
    "audit_candidate_lcc",
    "audit_geodesic_spearman",
    "audit_trajectory_conservation",
    "audit_trajectory_conservation_penalized",
    "zadu_trustworthiness",
    "zadu_continuity",
    "zadu_lcmc",
    "zadu_steadiness",
    "zadu_cohesiveness",
    "scib_graph_connectivity",
    "scib_trajectory_conservation",
)


def _summarise_discrimination(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Report continuous changes and legacy directional regression checks.

    A negative change is not a demonstrated useful effect or power estimate.
    Historical boolean keys remain compatible, explicitly qualified by scope.
    """

    by_name = {str(row["candidate"]): row for row in rows}
    faithful = by_name["faithful"]
    out: list[dict[str, object]] = []
    for key in _DISCRIMINATION_KEYS:
        base = faithful.get(key)
        if not isinstance(base, float) or not np.isfinite(base):
            continue
        record: dict[str, object] = {"metric": key, "faithful": float(base)}
        verdicts = []
        for adversary in ("collapse", "shatter", "permuted"):
            value = by_name[adversary].get(key)
            value = float(value) if isinstance(value, float) else float("nan")
            record[adversary] = value
            record[f"{adversary}_delta"] = value - base
            record[f"{adversary}_relative_delta"] = (value - base) / abs(base) if base else float("nan")
            detected = bool(np.isfinite(value) and value < base)
            record[f"{adversary}_penalised"] = detected
            verdicts.append(detected)
        # Scale invariance is a separate correctness requirement: a rigid
        # rotation with uniform scaling must not change a geometry score.
        rotated = by_name["rotated_scaled"].get(key)
        rotated = float(rotated) if isinstance(rotated, float) else float("nan")
        record["rotated_scaled"] = rotated
        record["scale_invariant"] = bool(
            np.isfinite(rotated) and abs(rotated - base) <= 0.05
        )
        record["passes_topology_test"] = bool(all(verdicts))
        record["direction_check_scope"] = "legacy key: all damaged scores lower on this fixture only"
        out.append(record)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="output JSON path")
    parser.add_argument("--n-per-arm", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=15)
    args = parser.parse_args(argv)

    result = evaluate_ladder(n_per_arm=args.n_per_arm, seed=args.seed, k=args.k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.out}")
    print("\nagreement with upstream (max |delta| across candidates):")
    for item in result["agreement"]:
        print(
            f"  {item['local']:<32s} vs {item['upstream']:<24s} "
            f"max={item['max_abs_delta']:.6f}"
        )
    print("\nfixture response (direction is not detection power):")
    for item in result["discrimination"]:
        flag = "all lower" if item["passes_topology_test"] else "mixed direction"
        print(
            f"  [{flag}] {item['metric']:<42s} "
            f"faithful={item['faithful']:.3f} "
            f"collapse={item['collapse']:.3f} "
            f"shatter={item['shatter']:.3f} "
            f"scale_inv={item['scale_invariant']}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
