#!/usr/bin/env python3
"""Regenerate the scored diversity-extension arrays (they were never saved).

The archived run (scripts/diversity_extension.py) saved only scores and later a
2-D view export (object_views.csv; PCA columns 1-2 only, scVI a refit).  The
chance-corrected analysis needs the full scored arrays, so this script imports
the archived loaders and ``representations`` unchanged and re-runs them.  PCA,
UMAP, random projection and the PCA permutation are seeded and expected to be
reproduced; scVI GPU training may not be bit-reproducible.  Every regenerated
candidate is re-scored with the archived scorer and compared with the archived
results.json so any discrepancy is visible.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, PAPER, SCRIPTS, clean, dump, edge_survival, reachability_survival, separation_survival  # noqa: E402

import numpy as np  # noqa: E402

import diversity_extension as dx  # noqa: E402  (archived module, imported only)

ARR = OUT / "arrays" / "diversity"
K = 15


def geometry(reference, candidate):
    out = {}
    out.update(reachability_survival(reference, candidate, k=K))
    out.update(edge_survival(reference, candidate, k=K))
    out.update(separation_survival(reference, candidate, k=K))
    return {k: float(v) for k, v in out.items()}


def loaders():
    SR = dx.SPATIAL_ROOT
    return [
        ("bulb", dx.as_spatial(dx.load_mob)),
        ("pfc_151673", dx.as_spatial(dx.load_dlpfc)),
        ("metastasis", dx.as_spatial(dx.load_bclm)),
        ("pfc_151507", lambda: dx.load_visium(
            "pfc_151507",
            SR / "Banksy/data/DLPFC/151507/151507_raw_feature_bc_matrix.h5",
            SR / "Banksy/data/DLPFC/151507/tissue_positions_list.txt")),
        ("lymph", lambda: dx.load_visium(
            "lymph",
            SR / "SEDR_analyses/data/Human_Lymph_Node/filtered_feature_bc_matrix.h5",
            SR / "SEDR_analyses/data/Human_Lymph_Node/spatial/tissue_positions_list.csv")),
        ("breast", lambda: dx.load_visium(
            "breast",
            SR / "SEDR_analyses/data/BRCA1/V1_Human_Breast_Cancer_Block_A_Section_1/filtered_feature_bc_matrix.h5",
            SR / "SEDR_analyses/data/BRCA1/V1_Human_Breast_Cancer_Block_A_Section_1/spatial/tissue_positions_list.csv")),
        ("ovary", lambda: dx.load_visium(
            "ovary",
            SR / "SEDR_analyses/data/Human_Ovarian_Cancer/filtered_feature_bc_matrix.h5",
            SR / "SEDR_analyses/data/Human_Ovarian_Cancer/spatial/tissue_positions_list.csv")),
        ("pbmc3k", dx.load_pbmc),
    ]


def main():
    import scanpy as sc
    sc.settings.verbosity = 0
    ARR.mkdir(parents=True, exist_ok=True)
    archived = {o["object"]: o for o in json.loads((PAPER / "evidence/diversity-20261009/results.json").read_text())["objects"]}
    report = []
    for name, loader in loaders():
        t0 = time.time()
        adata, coords = loader()
        names_before = np.asarray(adata.obs_names.astype(str))
        bundle, meta = dx.representations(adata)
        keep = meta.pop("keep")
        obs_names = names_before[keep]
        payload = {k: np.asarray(v, dtype=float) for k, v in bundle.items()}
        payload["obs_names"] = obs_names.astype("U64")
        if coords is not None:
            payload["tissue"] = np.asarray(coords, dtype=float)[keep]
        np.savez_compressed(ARR / f"{name}.npz", **payload)
        cmp = {"object": name, "meta": meta, "n": int(len(obs_names)), "diffs": {}}
        refs = {"expression_reference": payload["pca"]}
        if "tissue" in payload:
            refs["spatial_reference"] = payload["tissue"]
        for refname, ref in refs.items():
            for cand in ("pca", "scvi", "umap", "random", "permuted_pca"):
                new = geometry(ref, payload[cand])
                old = archived[name][refname][cand]
                cmp["diffs"][f"{refname}/{cand}"] = {
                    m: {"archived": old[m], "regenerated": new[m], "diff": new[m] - old[m]}
                    for m in ("reachability_survival", "edge_survival", "separation_survival")
                }
        cmp["seconds"] = time.time() - t0
        report.append(cmp)
        maxd = {c: max(abs(v["diff"]) for key, d in cmp["diffs"].items() if key.endswith("/" + c) for v in d.values())
                for c in ("pca", "scvi", "umap", "random", "permuted_pca")}
        print(name, cmp["n"], "max|diff| by candidate", {k: f"{v:.2e}" for k, v in maxd.items()}, f"{cmp['seconds']:.0f}s", flush=True)
    dump(OUT / "results" / "regen_diversity_check.json", {
        "note": "Regenerated with archived diversity_extension.representations; diffs = regenerated - archived results.json",
        "objects": report,
    })


if __name__ == "__main__":
    main()
