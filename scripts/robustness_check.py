#!/usr/bin/env python3
"""Independent fixed-object checks; no import from the archived RAAG scorer."""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.special import xlogy

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evidence/scale-correspondence"
OUT = ROOT / "evidence/robustness-audit"
METRICS = ("reachability_survival", "edge_survival", "separation_survival")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def graph(x, k):
    distances = cdist(np.asarray(x, float), np.asarray(x, float))
    np.fill_diagonal(distances, np.inf)
    order = np.argsort(distances, axis=1, kind="stable")
    boundary_ties = int(np.count_nonzero(np.take_along_axis(distances, order[:, k-1:k], 1) ==
                                        np.take_along_axis(distances, order[:, k:k+1], 1)))
    adjacency = np.zeros(distances.shape, bool)
    adjacency[np.arange(len(x))[:, None], order[:, :k]] = True
    adjacency |= adjacency.T
    return adjacency, boundary_ties


def score_geometry(x, z, k):
    gx, xties = graph(x, k)
    gz, zties = graph(z, k)
    hx = shortest_path(csr_matrix(gx), directed=False, unweighted=True)
    hz = shortest_path(csr_matrix(gz), directed=False, unweighted=True)
    tri = np.triu_indices(len(x), 1)
    rh, ch = hx[tri], hz[tri]
    ref_ok = np.isfinite(rh)
    far = ref_ok & (rh >= np.quantile(rh[ref_ok], .9))
    edges = np.triu(gx, 1)
    values = {
        "reachability_survival": float(np.mean(np.isfinite(ch[ref_ok]))),
        "edge_survival": float(np.mean(1/hz[edges])),
        "separation_survival": float(np.mean(ch[far] >= np.median(ch[np.isfinite(ch)]))),
    }
    return values, gx, xties, zties


def pairwise_js(z):
    p = np.clip(np.asarray(z, float), 1e-12, None)
    p /= p.sum(axis=1, keepdims=True)
    js = np.empty((len(p), len(p)))
    for i in range(len(p)):
        mid = (p[i] + p)/2
        js[i] = .5*np.sum(xlogy(p[i], p[i]/mid) + xlogy(p, p/mid), axis=1)
    return js


def main():
    inputs = json.loads((SOURCE/"inputs.json").read_text())
    stored = json.loads((SOURCE/"results.json").read_text())
    original = json.loads((ROOT/"evidence/paired-grid-permutation/results.json").read_text())
    lookup = {(r["id"], r["k"], r["rank"]): r for r in stored["scale"]}
    comparisons, permutations = [], []
    for item in inputs:
        path = SOURCE/item["file"]
        assert sha(path) == item["sha256"], path
        with np.load(path) as data:
            x, z = data["reference"], data["latent"]
        for k in (5, 15, 30):
            for rank in (10, 25, 50):
                independent, gx, xties, zties = score_geometry(x[:, :rank], z, k)
                frozen = lookup[(item["id"], k, rank)]
                errors = {name: abs(independent[name]-frozen[name]) for name in METRICS}
                assert max(errors.values()) < 1e-12, (item["id"], k, rank, errors)
                comparisons.append(dict(id=item["id"], k=k, rank=rank, errors=errors,
                                        reference_boundary_ties=xties, candidate_boundary_ties=zties))
        if item["native"]:
            gx, _ = graph(x, 15)
            src, dst = np.where(np.triu(gx, 1))
            js = pairwise_js(z)
            denominator = float(js[np.triu_indices(len(z), 1)].mean())
            observed = float(np.clip(1-js[src, dst].mean()/denominator, 0, 1))
            expected = lookup[(item["id"], 15, 50)]["allocation_coherence"]
            assert abs(observed-expected) < 1e-12
            streams = []
            for seed in (20260922, 104729):
                rng = np.random.default_rng(seed)
                null = []
                for _ in range(999):
                    order = rng.permutation(len(z))
                    null.append(float(np.clip(1-js[order[src], order[dst]].mean()/denominator, 0, 1)))
                tail = int(np.count_nonzero(np.asarray(null) >= observed))
                streams.append(dict(seed=seed, draws=999, exceedances=tail,
                                    monte_carlo_p=(tail+1)/1000, maximum=max(null)))
            previous = next(r for r in original["native_permutations"] if r["id"] == item["id"])
            assert streams[0]["exceedances"] == previous["null_ge_observed"]
            assert abs(streams[0]["maximum"]-previous["null_max"]) < 1e-12
            permutations.append(dict(id=item["id"], observed=observed,
                                     exact_denominator=denominator,
                                     observed_absolute_error=abs(observed-expected), streams=streams))
        print(item["id"], "independent grid verified", flush=True)
    ps = np.asarray([p["streams"][0]["monte_carlo_p"] for p in permutations])
    order = np.argsort(ps, kind="stable")
    holm = np.minimum(1, np.maximum.accumulate((len(ps)-np.arange(len(ps)))*ps[order]))
    for index, value in zip(order, holm):
        permutations[index]["posthoc_holm_p"] = float(value)
    result = dict(scope="post hoc numerical robustness on 24 fixed computational objects; no biological replication",
                  protocol="PROTOCOL.md", independent_method="SciPy cdist with explicit self-exclusion and stable sort; dense graph construction; scipy.special.xlogy JSD; no archived scorer import",
                  driver_sha256=sha(Path(__file__)), input_manifest_sha256=sha(SOURCE/"inputs.json"),
                  grid_comparisons=comparisons, native_permutations=permutations,
                  summary=dict(grid_computations=len(comparisons),
                               maximum_geometry_absolute_error=max(v for r in comparisons for v in r["errors"].values()),
                               boundary_ties=sum(r["reference_boundary_ties"]+r["candidate_boundary_ties"] for r in comparisons),
                               maximum_coherence_absolute_error=max(r["observed_absolute_error"] for r in permutations),
                               new_stream_exceedances=sum(r["streams"][1]["exceedances"] for r in permutations),
                               posthoc_holm_p_range=[float(holm.min()),float(holm.max())]))
    OUT.mkdir(exist_ok=True)
    (OUT/"results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
