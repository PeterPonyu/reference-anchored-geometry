#!/usr/bin/env python3
"""Stress the archived neighbour implementation on an exact-tie fixture."""
import json
from pathlib import Path
import numpy as np
from local_raag import knn_hop_graph, raag_battery

ROOT = Path(__file__).resolve().parents[1]
n = 20
x = np.random.default_rng(91).normal(size=(n, 3))
z = np.zeros((n, 3))
g = knn_hop_graph(z, k=5)
baseline = raag_battery(x, z, k=5, already_simplex=False)
metrics = ("reachability_survival", "edge_survival", "separation_survival")
rows = []
for seed in (0, 17, 42):
    order = np.random.default_rng(seed).permutation(n)
    moved = raag_battery(x[order], z[order], k=5, already_simplex=False)
    rows.append(dict(seed=seed, changes={name: moved[name]-baseline[name] for name in metrics}))
result = dict(scope="post hoc synthetic exact-coordinate-tie stress test; not part of the stored 24-object experiment",
              n_rows=n, candidate="twenty identical three-dimensional zero rows", k=5,
              archived_self_loops=int(g.diagonal().sum()),
              baseline={name:baseline[name] for name in metrics}, joint_reorder=rows,
              interpretation="Frozen neighbour routine removes the first returned rank rather than the identity explicitly. Exact duplicates expose a self-exclusion assumption and row-order-sensitive graph choices. The archived scorer is retained unchanged for reproducibility; general tie handling is not validated by the tie-free fixed-object controls.")
(ROOT/"evidence/robustness-audit/tied-graph-probe.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
