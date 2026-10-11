#!/usr/bin/env python3
"""Diversity extension declared in evidence/diversity-20261009/PROTOCOL.md."""

from __future__ import annotations

import json
import os
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import scvi
import torch
from anndata import AnnData

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from local_raag import edge_survival, reachability_survival, separation_survival  # noqa: E402
from spatial_reference_case import load_bclm, load_dlpfc, load_mob  # noqa: E402

OUT = ROOT / "evidence" / "diversity-20261009"
K = 15
SPATIAL_ROOT = Path(os.environ.get("RAAG_SPATIAL_EXTERNAL", ROOT / "data" / "spatial-external"))


def geometry(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    out.update(reachability_survival(reference, candidate, k=K))
    out.update(edge_survival(reference, candidate, k=K))
    out.update(separation_survival(reference, candidate, k=K))
    return {key: float(value) for key, value in out.items()}


def representations(adata: AnnData) -> tuple[dict[str, np.ndarray], dict]:
    sc.pp.filter_genes(adata, min_cells=3)
    keep = np.asarray(adata.X.sum(axis=1)).ravel() > 0
    adata = adata[keep].copy()
    counts = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    n_top = int(min(2000, adata.n_vars))
    if n_top < adata.n_vars:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor="seurat")
        hvg = adata.var["highly_variable"].to_numpy()
        adata = adata[:, hvg].copy()
        counts = counts[:, hvg]
    else:
        hvg = np.ones(adata.n_vars, dtype=bool)
    view = adata.copy()
    sc.pp.scale(view, max_value=10)
    scaled = np.asarray(view.X.todense() if hasattr(view.X, "todense") else view.X, dtype=float)
    n_comps = int(min(50, view.n_obs - 1, view.n_vars - 1))
    sc.tl.pca(view, n_comps=n_comps, random_state=42, svd_solver="arpack")
    pca = np.asarray(view.obsm["X_pca"], dtype=float)
    sc.pp.neighbors(view, n_neighbors=15, n_pcs=n_comps, random_state=0)
    sc.tl.umap(view, random_state=0, n_components=2)
    umap = np.asarray(view.obsm["X_umap"], dtype=float)
    weights = np.random.default_rng(0).normal(size=(scaled.shape[1], 10))
    random_projection = scaled @ weights
    scvi.settings.seed = 0
    torch.manual_seed(0)
    model_view = adata.copy()
    model_view.X = counts
    scvi.model.SCVI.setup_anndata(model_view)
    model = scvi.model.SCVI(model_view, n_latent=10, n_layers=1, gene_likelihood="nb")
    model.train(
        max_epochs=200,
        accelerator="gpu",
        devices=1,
        batch_size=int(min(128, max(32, model_view.n_obs // 8))),
        early_stopping=True,
    )
    latent = np.asarray(model.get_latent_representation(), dtype=float)
    order = np.random.default_rng(0).permutation(pca.shape[0])
    bundle = {
        "pca": pca,
        "scvi": latent,
        "umap": umap,
        "random": random_projection,
        "permuted_pca": pca[order],
    }
    meta = {
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "pca_components": n_comps,
        "scvi_epochs_ran": int(len(model.history["elbo_train"])),
        "keep": keep,
    }
    return bundle, meta


def read_positions(path: Path) -> pd.DataFrame:
    sample = path.read_text(errors="replace").splitlines()[0]
    header = "barcode" in sample.lower() or "in_tissue" in sample.lower()
    if header:
        frame = pd.read_csv(path)
        frame = frame.rename(columns={frame.columns[0]: "barcode"})
    else:
        frame = pd.read_csv(
            path, header=None,
            names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"],
        )
    frame["barcode"] = frame["barcode"].astype(str)
    row_col = "pxl_row_in_fullres" if "pxl_row_in_fullres" in frame.columns else "pxl_row"
    col_col = "pxl_col_in_fullres" if "pxl_col_in_fullres" in frame.columns else "pxl_col"
    frame = frame.set_index("barcode")
    return frame.rename(columns={row_col: "y", col_col: "x"})[["in_tissue", "y", "x"]]


def load_visium(name: str, h5: Path, positions: Path) -> tuple[AnnData, np.ndarray]:
    adata = sc.read_10x_h5(h5)
    adata.var_names_make_unique()
    pos = read_positions(positions)
    names = pd.Index(adata.obs_names.astype(str))
    if not names.isin(pos.index).any():
        names = names.str.replace(r"-\d+$", "", regex=True)
        pos.index = pos.index.str.replace(r"-\d+$", "", regex=True)
        adata.obs_names = names
    if not names.isin(pos.index).all():
        raise RuntimeError(f"{name} barcodes do not match positions")
    tissue = pos.loc[names]
    keep = tissue["in_tissue"].to_numpy() == 1
    adata = adata[keep].copy()
    coords = tissue.loc[keep, ["y", "x"]].to_numpy(dtype=float)
    return adata, coords


def load_pbmc() -> tuple[AnnData, None]:
    destination = OUT / "pbmc3k_matrix"
    if not (destination / "matrix.mtx").exists():
        destination.mkdir(parents=True, exist_ok=True)
        archive = Path(os.environ.get("RAAG_PBMC3K_ARCHIVE", ROOT / "data" / "pbmc3k_filtered_gene_bc_matrices.tar.gz"))
        with tarfile.open(archive) as handle:
            for member in handle.getmembers():
                if member.name.endswith(("matrix.mtx", "genes.tsv", "barcodes.tsv")):
                    member.name = Path(member.name).name
                    handle.extract(member, destination)
    adata = sc.read_10x_mtx(destination, var_names="gene_symbols", make_unique=True)
    return adata, None


def score_bundle(reference: np.ndarray, bundle: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {name: geometry(reference, values) for name, values in bundle.items()}


def as_spatial(loader):
    def inner():
        adata, coords, _provenance = loader()
        return adata, coords
    return inner


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sc.settings.verbosity = 0
    spatial_objects = [
        ("bulb", as_spatial(load_mob)),
        ("pfc_151673", as_spatial(load_dlpfc)),
        ("metastasis", as_spatial(load_bclm)),
        ("pfc_151507", lambda: load_visium(
            "pfc_151507",
            SPATIAL_ROOT / "Banksy/data/DLPFC/151507/151507_raw_feature_bc_matrix.h5",
            SPATIAL_ROOT / "Banksy/data/DLPFC/151507/tissue_positions_list.txt",
        )),
        ("lymph", lambda: load_visium(
            "lymph",
            SPATIAL_ROOT / "SEDR_analyses/data/Human_Lymph_Node/filtered_feature_bc_matrix.h5",
            SPATIAL_ROOT / "SEDR_analyses/data/Human_Lymph_Node/spatial/tissue_positions_list.csv",
        )),
        ("breast", lambda: load_visium(
            "breast",
            SPATIAL_ROOT / "SEDR_analyses/data/BRCA1/V1_Human_Breast_Cancer_Block_A_Section_1/filtered_feature_bc_matrix.h5",
            SPATIAL_ROOT / "SEDR_analyses/data/BRCA1/V1_Human_Breast_Cancer_Block_A_Section_1/spatial/tissue_positions_list.csv",
        )),
        ("ovary", lambda: load_visium(
            "ovary",
            SPATIAL_ROOT / "SEDR_analyses/data/Human_Ovarian_Cancer/filtered_feature_bc_matrix.h5",
            SPATIAL_ROOT / "SEDR_analyses/data/Human_Ovarian_Cancer/spatial/tissue_positions_list.csv",
        )),
    ]
    rows = []
    failures = []
    for name, loader in spatial_objects:
        try:
            adata, coords = loader()
            bundle, meta = representations(adata)
            coords = np.asarray(coords, dtype=float)[meta.pop("keep")]
            identity = geometry(coords, coords)
            if identity["edge_survival"] < 0.999:
                raise RuntimeError(f"identity check failed: {identity}")
            rows.append({
                "object": name,
                "kind": "spatial",
                "preprocessing": meta,
                "spatial_reference": score_bundle(coords, bundle),
                "expression_reference": score_bundle(bundle["pca"], bundle),
            })
            print(name, meta["n_spots"], "ok", flush=True)
        except Exception as exc:
            failures.append({"object": name, "error": f"{type(exc).__name__}: {exc}"})
            print(name, "FAILED", exc, flush=True)
    try:
        adata, _coords = load_pbmc()
        bundle, meta = representations(adata)
        meta.pop("keep", None)
        rows.append({
            "object": "pbmc3k",
            "kind": "dissociated",
            "preprocessing": meta,
            "expression_reference": score_bundle(bundle["pca"], bundle),
        })
        print("pbmc3k", meta["n_spots"], "ok", flush=True)
    except Exception as exc:
        failures.append({"object": "pbmc3k", "error": f"{type(exc).__name__}: {exc}"})
        print("pbmc3k FAILED", exc, flush=True)
    payload = {"protocol": "evidence/diversity-20261009/PROTOCOL.md", "k": K, "objects": rows, "failures": failures}
    (OUT / "results.json").write_text(json.dumps(payload, indent=2))
    print("wrote", OUT / "results.json", "failures", len(failures), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
