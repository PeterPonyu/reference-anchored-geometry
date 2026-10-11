#!/usr/bin/env python3
"""A1. Reachability degeneracy under a connected reference, and a reference-aware alternative.

Claim (proved in the manuscript Methods): if the reference k-NN graph is connected then
P_X is every unordered pair and

    R(X, Z) = |{i<j : h_Z(i,j) < inf}| / C(n,2) = sum_c C(n_c, 2) / C(n, 2),

where n_c are the candidate component sizes.  R then depends on Z only.

Numerical checks use the archived reachability_survival() on
  (a) all 36 calibration fixture candidates (3 seeds x 12) at k=15,
  (b) the 24 scale objects at k in {5,15,30} x reference rank {10,25,50},
  (c) the regenerated diversity arrays (both references, 5 candidates).
The proposed alternative is pair-counting precision/recall/F1 and ARI between
reference and candidate component partitions; it is evaluated on a fixture
whose reference is genuinely fragmented.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (OUT, PAPER, UPSTREAM, component_formula_R, components, dump,  # noqa: E402
                    edge_survival, partition_pair_scores, reachability_survival, separation_survival)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(UPSTREAM))
from audit.upstream_validation import make_candidates, make_reference_manifold  # noqa: E402


def check(ref, cand, k):
    r = reachability_survival(ref, cand, k=k)
    ref_lab = components(ref, k)
    cand_lab = components(cand, k)
    formula = component_formula_R(cand_lab)
    return {
        "R_archived_fn": r["reachability_survival"],
        "R_component_formula": formula,
        "abs_diff": abs(r["reachability_survival"] - formula),
        "ref_connected": bool(len(np.unique(ref_lab)) == 1),
        "n_ref_components": int(len(np.unique(ref_lab))),
        "n_cand_components": int(len(np.unique(cand_lab))),
        "cand_lcc_fraction": float(np.bincount(cand_lab).max() / len(cand_lab)),
    }


def fixture_rows():
    cal = json.loads((PAPER / "evidence/measurement-inventory/calibration.json").read_text())
    stored = {(r["seed"], r["candidate"]): r for r in cal["rows"]}
    rows = []
    for seed in (7, 17, 42):
        a = np.load(PAPER / f"evidence/measurement-inventory/calibration_arrays/seed{seed}.npz")
        ref = a["coords"]
        for name in a.files:
            if name in ("coords", "pseudotime", "branch"):
                continue
            c = check(ref, a[name], 15)
            st = stored[(seed, name)]["stacks"]
            c.update(group="fixture", object=f"seed{seed}", candidate=name, k=15, rank=2,
                     R_stored=st["raag"]["values"]["reachability_survival"],
                     audit_candidate_lcc_stored=st["audit"]["values"]["audit_candidate_lcc"])
            rows.append(c)
    return rows


def scale_rows():
    meta = json.loads((PAPER / "evidence/scale-correspondence/inputs.json").read_text())
    res = json.loads((PAPER / "evidence/scale-correspondence/results.json").read_text())["scale"]
    stored = {(r["id"], r["k"], r["rank"]): r["reachability_survival"] for r in res}
    rows = []
    for item in meta:
        a = np.load(PAPER / "evidence/scale-correspondence" / item["file"])
        x, z = a["reference"], a["latent"]
        for k in (5, 15, 30):
            for rank in (10, 25, 50):
                c = check(x[:, :rank], z, k)
                c.update(group="scale24", object=item["id"], candidate="latent", k=k, rank=rank,
                         R_stored=stored[(item["id"], k, rank)], n=int(len(z)))
                rows.append(c)
    return rows


def diversity_rows():
    rows = []
    folder = OUT / "arrays" / "diversity"
    for path in sorted(folder.glob("*.npz")):
        a = np.load(path)
        refs = {"expression": a["pca"]}
        if "tissue" in a.files:
            refs["tissue"] = a["tissue"]
        for refname, ref in refs.items():
            for cand in ("pca", "scvi", "umap", "random", "permuted_pca"):
                c = check(ref, a[cand], 15)
                c.update(partition_pair_scores(components(ref, 15), components(a[cand], 15)))
                c.update(group="diversity", object=path.stem, candidate=cand, reference=refname, k=15, n=int(len(ref)))
                rows.append(c)
    return rows


def fragmented_fixture():
    """Reference with three genuinely disjoint pieces: the Y fixture with arms
    translated by +/-9 (the archived shatter_9 geometry applied to the reference).
    Candidates probe the failure modes R can and cannot see."""
    out = []
    for seed in (7, 17, 42):
        base = make_reference_manifold(150, seed)
        branch = base["branch"]
        clean_y = base["coords"]
        ref = clean_y.copy()
        ref[branch == 1] += [0.0, 9.0]
        ref[branch == 2] += [0.0, -9.0]
        # Independent candidate noise (seed offset) so the faithful copy is not
        # a deterministic rescaling of the reference jitter.
        rng = np.random.default_rng(seed + 100_000)
        faithful = ref + rng.normal(scale=0.01, size=ref.shape)
        merged = clean_y + rng.normal(scale=0.01, size=ref.shape)  # arms re-attached: 1 component
        over = faithful.copy()
        trunk = branch == 0
        over[trunk & (base["pseudotime"] < 0.25)] += [-9.0, 0.0]  # split trunk into two pieces
        permuted = faithful[rng.permutation(len(faithful))]
        merge_two = faithful.copy()
        merge_two[branch == 1] -= [0.0, 9.0]  # arm a re-attached to trunk, arm b still detached
        cands = {"faithful": faithful, "merged_all": merged, "merged_two": merge_two,
                 "over_fragmented": over, "row_permuted": permuted}
        for name, z in cands.items():
            r = reachability_survival(ref, z, k=15)
            e = edge_survival(ref, z, k=15)
            s = separation_survival(ref, z, k=15)
            p = partition_pair_scores(components(ref, 15), components(z, 15))
            out.append({"seed": seed, "candidate": name,
                        "R_archived": r["reachability_survival"],
                        "n_reference_connected_pairs": r["n_reference_connected_pairs"],
                        "E": e["edge_survival"], "S": s["separation_survival"], **p})
    return out


def main():
    rows = fixture_rows() + scale_rows() + diversity_rows()
    df = pd.DataFrame(rows)
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "results" / "a1_reachability_identity.csv", index=False)
    frag = pd.DataFrame(fragmented_fixture())
    frag.to_csv(OUT / "results" / "a1_fragmented_reference_fixture.csv", index=False)

    summary = {}
    for g, sub in df.groupby("group"):
        conn = sub[sub.ref_connected]
        summary[g] = {
            "n_scored": int(len(sub)),
            "n_reference_connected": int(sub.ref_connected.sum()),
            "max_abs_diff_R_vs_component_formula_connected_ref": float(conn.abs_diff.max()) if len(conn) else None,
            "n_R_below_1": int((sub.R_archived_fn < 1).sum()),
            "R_values_below_1": sorted(set(np.round(sub.loc[sub.R_archived_fn < 1, "R_archived_fn"], 6).tolist())),
        }
        if "R_stored" in sub and sub["R_stored"].notna().any():
            summary[g]["max_abs_diff_vs_stored_R"] = float((sub.R_archived_fn - sub.R_stored).abs().max())
        if "audit_candidate_lcc_stored" in sub:
            summary[g]["max_abs_diff_R_minus_stored_LCC"] = float((sub.R_archived_fn - sub.audit_candidate_lcc_stored).abs().max())
    summary["fragmented_fixture_mean_over_seeds"] = frag.groupby("candidate")[
        ["R_archived", "pair_precision", "pair_f1", "component_ari", "E", "S", "n_ref_components", "n_cand_components"]
    ].mean().round(4).to_dict(orient="index")
    dump(OUT / "results" / "a1_summary.json", summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
