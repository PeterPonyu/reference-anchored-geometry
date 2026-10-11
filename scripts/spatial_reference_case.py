#!/usr/bin/env python3
"""Spatial-coordinate reference case for RAAG geometry scores.

The contract is evidence/spatial-reference-20261009/PROTOCOL.md.
Scores are written after that contract; datasets are not dropped.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
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

DATA = Path(os.environ.get("RAAG_SPATIAL_DATA", ROOT / "data" / "spatial"))
OUT = ROOT / "evidence" / "spatial-reference-20261009"
K = 15


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counts_matrix(x) -> np.ndarray | scipy.sparse.spmatrix:
    if hasattr(x, "toarray"):
        return x.tocsr()
    return np.asarray(x)


def prepare_expression(
    adata: AnnData, coords: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    sc.pp.filter_genes(adata, min_cells=3)
    keep = np.asarray(adata.X.sum(axis=1)).ravel() > 0
    adata = adata[keep].copy()
    coords = np.asarray(coords, dtype=float)[keep]
    adata.layers["counts"] = _counts_matrix(adata.X)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    n_top = int(min(2000, adata.n_vars))
    if n_top < adata.n_vars:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor="seurat")
        adata = adata[:, adata.var["highly_variable"]].copy()
    else:
        n_top = int(adata.n_vars)
    pca_view = adata.copy()
    sc.pp.scale(pca_view, max_value=10)
    n_comps = int(min(50, pca_view.n_obs - 1, pca_view.n_vars - 1))
    sc.tl.pca(pca_view, n_comps=n_comps, random_state=42, svd_solver="arpack")
    pca = np.asarray(pca_view.obsm["X_pca"], dtype=float)

    scvi.settings.seed = 0
    torch.manual_seed(0)
    model_view = adata.copy()
    model_view.X = model_view.layers["counts"].copy()
    scvi.model.SCVI.setup_anndata(model_view)
    model = scvi.model.SCVI(model_view, n_latent=10, n_layers=1, gene_likelihood="nb")
    batch_size = int(min(128, max(32, model_view.n_obs // 8)))
    model.train(
        max_epochs=200,
        accelerator="gpu",
        devices=1,
        batch_size=batch_size,
        early_stopping=True,
    )
    latent = np.asarray(model.get_latent_representation(), dtype=float)
    epochs = int(len(model.history["elbo_train"]))
    meta = {
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_hvg_requested": n_top,
        "pca_components": n_comps,
        "scvi_latent": 10,
        "scvi_epochs_ran": epochs,
        "scvi_batch_size": batch_size,
        "scvi_seed": 0,
        "pca_seed": 42,
        "device": torch.cuda.get_device_name(0),
    }
    return pca, latent, coords, meta


def geometry(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    out.update(reachability_survival(reference, candidate, k=K))
    out.update(edge_survival(reference, candidate, k=K))
    out.update(separation_survival(reference, candidate, k=K))
    return {key: float(value) for key, value in out.items()}


def load_mob() -> tuple[AnnData, np.ndarray, dict]:
    expr_path = DATA / "st_stahl2016_mob_rep11" / "Rep11_MOB_0_expr.txt"
    loc_path = DATA / "st_stahl2016_mob_rep11" / "Rep11_MOB_0_location.txt"
    expr = pd.read_csv(expr_path, sep=r"\s+", index_col=0)
    loc = pd.read_csv(loc_path, sep=r"\s+")
    loc = loc.set_index("ID")
    shared = [spot for spot in loc.index if spot in expr.columns]
    if len(shared) < 5:
        raise RuntimeError(f"MOB alignment failed: {len(shared)} shared spots")
    matrix = expr.loc[:, shared].to_numpy(dtype=float).T
    coords = loc.loc[shared, ["x", "y"]].to_numpy(dtype=float)
    adata = AnnData(matrix)
    adata.obs_names = shared
    return adata, coords, {"files": {str(expr_path): sha256(expr_path), str(loc_path): sha256(loc_path)}}


def load_dlpfc() -> tuple[AnnData, np.ndarray, dict]:
    h5_path = DATA / "dlpfc_libd_151673" / "151673_filtered_feature_bc_matrix.h5"
    pos_path = DATA / "dlpfc_libd_151673" / "tissue_positions_list.txt"
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()
    pos = pd.read_csv(pos_path, header=None, names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"])
    pos = pos.set_index("barcode")
    names = pd.Index(adata.obs_names.astype(str))
    if not names.isin(pos.index).all():
        stripped = names.str.replace(r"-\d+$", "", regex=True)
        if not stripped.isin(pos.index.str.replace(r"-\d+$", "", regex=True)).all():
            raise RuntimeError("DLPFC barcodes do not match tissue positions")
        pos.index = pos.index.str.replace(r"-\d+$", "", regex=True)
        names = stripped
        adata.obs_names = names
    tissue = pos.loc[names]
    keep = tissue["in_tissue"].to_numpy() == 1
    adata = adata[keep].copy()
    coords = tissue.loc[keep, ["pxl_row", "pxl_col"]].to_numpy(dtype=float)
    return adata, coords, {"files": {str(h5_path): sha256(h5_path), str(pos_path): sha256(pos_path)}}


def load_bclm() -> tuple[AnnData, np.ndarray, dict]:
    folder = DATA / "visium_bclm_A_0069226"
    paths = {
        "matrix": folder / "A_0069226_matrix.mtx.gz",
        "barcodes": folder / "A_0069226_barcodes.tsv.gz",
        "features": folder / "A_0069226_features.tsv.gz",
        "positions": folder / "A_0069226_tissue_positions.csv",
    }
    matrix = scipy.io.mmread(paths["matrix"]).tocsr()
    barcodes = pd.read_csv(paths["barcodes"], header=None, sep="\t")[0].astype(str)
    features = pd.read_csv(paths["features"], header=None, sep="\t")
    positions = pd.read_csv(paths["positions"]).set_index("barcode")
    if matrix.shape[0] == len(features) and matrix.shape[1] == len(barcodes):
        matrix = matrix.T.tocsr()
    elif not (matrix.shape[0] == len(barcodes) and matrix.shape[1] == len(features)):
        raise RuntimeError(f"BCLM matrix shape {matrix.shape} does not match {len(barcodes)} barcodes and {len(features)} features")
    if len(barcodes) != len(set(barcodes)):
        raise RuntimeError("BCLM barcodes are not unique")
    tissue = positions.loc[barcodes]
    keep = tissue["in_tissue"].to_numpy() == 1
    adata = AnnData(matrix[keep])
    adata.obs_names = barcodes.to_numpy()[keep]
    adata.var_names = features.iloc[:, 0].astype(str).to_numpy()
    adata.var_names_make_unique()
    coords = tissue.loc[keep, ["pxl_row_in_fullres", "pxl_col_in_fullres"]].to_numpy(dtype=float)
    return adata, coords, {"files": {str(path): sha256(path) for path in paths.values()}}


def score_object(name: str, loader) -> dict:
    adata, coords, provenance = loader()
    pca, latent, coords, meta = prepare_expression(adata, coords)
    if pca.shape[0] != coords.shape[0] or latent.shape[0] != coords.shape[0]:
        raise RuntimeError(f"{name} lost spot alignment")
    permutation = np.random.default_rng(0).permutation(pca.shape[0])
    candidates = {
        "spatial_identity": coords,
        "expression_pca": pca,
        "expression_scvi": latent,
        "pca_row_permutation": pca[permutation],
    }
    scores = {label: geometry(coords, values) for label, values in candidates.items()}
    identity = scores["spatial_identity"]
    if identity["reachability_survival"] < 0.999 or identity["edge_survival"] < 0.999:
        raise RuntimeError(f"{name} identity control failed: {identity}")
    return {
        "object": name,
        "provenance": provenance,
        "preprocessing": meta,
        "k": K,
        "scores": scores,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sc.settings.verbosity = 1
    objects = [
        ("stahl2016_mob_rep11", load_mob),
        ("maynard2021_dlpfc_151673", load_dlpfc),
        ("bclm_visium_A_0069226", load_bclm),
    ]
    rows = []
    failures = []
    for name, loader in objects:
        try:
            rows.append(score_object(name, loader))
            print(name, "ok", rows[-1]["preprocessing"], flush=True)
        except Exception as exc:  # retained, not dropped
            failures.append({"object": name, "error": f"{type(exc).__name__}: {exc}"})
            print(name, "FAILED", exc, flush=True)
    payload = {
        "protocol": "evidence/spatial-reference-20261009/PROTOCOL.md",
        "k": K,
        "native_coherence": "not_quoted_continuous_candidates",
        "objects": rows,
        "failures": failures,
    }
    destination = OUT / "results.json"
    destination.write_text(json.dumps(payload, indent=2))
    print("wrote", destination, flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
