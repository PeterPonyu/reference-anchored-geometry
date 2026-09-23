#!/usr/bin/env python3
"""Frozen paired-grid contrast and exact-denominator native permutation audit."""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"
import hashlib
import json
from pathlib import Path

import numpy as np
from local_raag import knn_hop_graph, to_simplex, _jensen_shannon

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evidence/scale-correspondence"
OUT = ROOT / "evidence/paired-grid-permutation"
METRICS = ("reachability_survival", "edge_survival", "separation_survival")
GRID = tuple((k, d) for k in (5, 15, 30) for d in (10, 25, 50))
BASE = (15, 50)
N_PERM = 999
RNG_SEED = 20260922


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    meta = json.loads((SOURCE / "inputs.json").read_text())
    source = json.loads((SOURCE / "results.json").read_text())
    index = {(r["dataset"], r["seed"], r["native"], r["k"], r["rank"]): r for r in source["scale"]}
    assert len(meta) == 24 and len(index) == 216 and len(GRID) == 9
    for item in meta:
        assert digest(SOURCE / item["file"]) == item["sha256"]
    assert len({(m["dataset"], m["seed"], m["native"]) for m in meta}) == 24
    pairs = []
    for dataset in ("setty", "dentate", "endo", "lung"):
        for seed in (0, 1, 2):
            for metric in METRICS:
                vals = []
                for k, d in GRID:
                    native = index[(dataset, seed, True, k, d)][metric]
                    continuous = index[(dataset, seed, False, k, d)][metric]
                    assert np.isfinite(native) and np.isfinite(continuous)
                    vals.append(dict(k=k, d=d, native=native, continuous=continuous, difference=native-continuous))
                baseline = next(v["difference"] for v in vals if (v["k"], v["d"]) == BASE)
                signs = [int(np.sign(v["difference"])) for v in vals]
                pairs.append(dict(dataset=dataset, seed=seed, metric=metric, baseline_difference=baseline,
                                  minimum_difference=min(v["difference"] for v in vals),
                                  maximum_difference=max(v["difference"] for v in vals),
                                  strict_reversals=sum(s*int(np.sign(baseline)) == -1 for s in signs),
                                  ties=sum(s == 0 for s in signs),
                                  settings=vals))
    # Separate, newly scoped 12-object test; does not revise historical 144-run null.
    perm_rows = []
    for item in meta:
        if not item["native"]:
            continue
        arrays = np.load(SOURCE / item["file"])
        x, z = arrays["reference"], arrays["latent"]
        p = to_simplex(z, already_simplex=True)
        n = len(p)
        graph = knn_hop_graph(x, k=15)
        src, dst = graph.nonzero()
        mask = src < dst
        src, dst = src[mask], dst[mask]
        js = np.zeros((n, n), dtype=np.float64)
        for i in range(n-1):
            js[i, i+1:] = _jensen_shannon(np.broadcast_to(p[i], p[i+1:].shape), p[i+1:])
        js += js.T
        denominator = float(js[np.triu_indices(n, 1)].mean())
        assert denominator > 1e-12
        def score(order):
            return float(np.clip(1-js[order[src], order[dst]].mean()/denominator, 0, 1))
        observed = score(np.arange(n))
        recorded = index[(item["dataset"], item["seed"], True, 15, 50)]["allocation_coherence"]
        assert abs(observed-recorded) < 1e-12, (item["id"], observed, recorded)
        rng = np.random.default_rng(RNG_SEED)
        null = np.array([score(rng.permutation(n)) for _ in range(N_PERM)])
        exceed = int(np.count_nonzero(null >= observed))
        perm_rows.append(dict(id=item["id"], dataset=item["dataset"], seed=item["seed"],
                              observed=observed, null_min=float(null.min()), null_median=float(np.median(null)),
                              null_max=float(null.max()), null_ge_observed=exceed,
                              monte_carlo_p=(exceed+1)/(N_PERM+1), n_permutations=N_PERM,
                              n_cells=n, n_reference_edges=len(src), input_sha256=item["sha256"]))
    summary = {}
    for metric in METRICS:
        selected = [r for r in pairs if r["metric"] == metric]
        summary[metric] = dict(n_pairs=len(selected), n_baseline_ties=sum(r["baseline_difference"] == 0 for r in selected),
                               n_pairs_with_reversal=sum(r["strict_reversals"] > 0 for r in selected),
                               n_reversed_settings=sum(r["strict_reversals"] for r in selected),
                               median_within_pair_difference_range=float(np.median([r["maximum_difference"]-r["minimum_difference"] for r in selected])))
    OUT.mkdir(exist_ok=True)
    result = dict(protocol="PROTOCOL.md", source_results_sha256=digest(SOURCE / "results.json"),
                  source_inputs_sha256=digest(SOURCE / "inputs.json"), scorer_sha256=digest(ROOT / "scripts/local_raag.py"),
                  driver_sha256=digest(Path(__file__)), permutation_seed=RNG_SEED,
                  permutation_definition="native allocation rows reassigned with NumPy Generator.permutation; reference fixed; one-sided observed >= null with +1 correction; ties counted as null >= observed",
                  grid_summary=summary, pairs=pairs, native_permutations=perm_rows)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(grid_summary=summary, native_p_values=[(r["id"], r["monte_carlo_p"]) for r in perm_rows]), indent=2))


if __name__ == "__main__":
    main()
