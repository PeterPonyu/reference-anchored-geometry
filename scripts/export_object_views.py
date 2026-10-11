#!/usr/bin/env python3
"""Export the plotted coordinates for the diversity objects.

PCA and UMAP follow the frozen diversity preprocessing. scVI is refit with the
same seed and budget only for the specimens drawn as embeddings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from diversity_extension import (  # noqa: E402
    SPATIAL_ROOT,
    as_spatial,
    load_bclm,
    load_dlpfc,
    load_mob,
    load_pbmc,
    load_visium,
    representations,
)

OUT = ROOT / "evidence" / "diversity-20261009" / "object_views.csv"


def views_only(adata):
    sc.pp.filter_genes(adata, min_cells=3)
    keep = np.asarray(adata.X.sum(axis=1)).ravel() > 0
    adata = adata[keep].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    n_top = int(min(2000, adata.n_vars))
    if n_top < adata.n_vars:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor="seurat")
        adata = adata[:, adata.var["highly_variable"].to_numpy()].copy()
    sc.pp.scale(adata, max_value=10)
    n_comps = int(min(50, adata.n_obs - 1, adata.n_vars - 1))
    sc.tl.pca(adata, n_comps=n_comps, random_state=42, svd_solver="arpack")
    pca = np.asarray(adata.obsm["X_pca"], dtype=float)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=n_comps, random_state=0)
    sc.tl.umap(adata, random_state=0, n_components=2)
    umap = np.asarray(adata.obsm["X_umap"], dtype=float)
    return keep, pca[:, 0], pca[:, :2], umap


def rows_for(object_name, coords, pc1, pca, umap, latent=None, tissue_xy=True):
    frames = []
    if coords is not None:
        x_index, y_index = (0, 1) if tissue_xy else (1, 0)
        frames.append(("tissue", coords[:, x_index], coords[:, y_index]))
    frames.append(("pca", pca[:, 0], pca[:, 1]))
    frames.append(("umap", umap[:, 0], umap[:, 1]))
    if latent is not None:
        frames.append(("scvi", latent[:, 0], latent[:, 1]))
    pieces = []
    for view, x, y in frames:
        pieces.append(pd.DataFrame({
            "object": object_name,
            "view": view,
            "x": x,
            "y": y,
            "pc1": pc1,
        }))
    return pieces


def main():
    sc.settings.verbosity = 0
    pieces = []
    spatial = {
        "bulb": as_spatial(load_mob),
        "pfc_151673": as_spatial(load_dlpfc),
        "metastasis": as_spatial(load_bclm),
        "lymph": lambda: load_visium(
            "lymph",
            SPATIAL_ROOT / "SEDR_analyses/data/Human_Lymph_Node/filtered_feature_bc_matrix.h5",
            SPATIAL_ROOT / "SEDR_analyses/data/Human_Lymph_Node/spatial/tissue_positions_list.csv",
        ),
        "breast": lambda: load_visium(
            "breast",
            SPATIAL_ROOT / "SEDR_analyses/data/BRCA1/V1_Human_Breast_Cancer_Block_A_Section_1/filtered_feature_bc_matrix.h5",
            SPATIAL_ROOT / "SEDR_analyses/data/BRCA1/V1_Human_Breast_Cancer_Block_A_Section_1/spatial/tissue_positions_list.csv",
        ),
        "ovary": lambda: load_visium(
            "ovary",
            SPATIAL_ROOT / "SEDR_analyses/data/Human_Ovarian_Cancer/filtered_feature_bc_matrix.h5",
            SPATIAL_ROOT / "SEDR_analyses/data/Human_Ovarian_Cancer/spatial/tissue_positions_list.csv",
        ),
    }
    gallery = {"breast", "bulb"}
    for name, loader in spatial.items():
        adata, coords = loader()
        if name in gallery:
            bundle, meta = representations(adata)
            coords = np.asarray(coords, dtype=float)[meta["keep"]]
            pc1 = bundle["pca"][:, 0]
            pieces.extend(rows_for(
                name, coords, pc1, bundle["pca"][:, :2], bundle["umap"], bundle["scvi"][:, :2],
                tissue_xy=(name == "bulb"),
            ))
        else:
            keep, pc1, pca, umap = views_only(adata)
            pieces.extend(rows_for(
                name, np.asarray(coords, dtype=float)[keep], pc1, pca, umap,
                tissue_xy=(name == "bulb"),
            ))
        print(name, "views", pieces[-1].shape[0], flush=True)
    adata, _coords = load_pbmc()
    bundle, meta = representations(adata)
    pieces.extend(rows_for("pbmc3k", None, bundle["pca"][:, 0], bundle["pca"][:, :2], bundle["umap"], bundle["scvi"][:, :2]))
    print("pbmc3k", meta["n_spots"], flush=True)
    frame = pd.concat(pieces, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    print("wrote", OUT, frame.shape, flush=True)


if __name__ == "__main__":
    main()
