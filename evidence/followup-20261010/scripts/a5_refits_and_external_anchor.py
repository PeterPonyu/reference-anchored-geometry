#!/usr/bin/env python3
"""A5. Fit-to-fit variability (scVI, UMAP; seeds 0-4) and a DLPFC layer anchor.

Refits: bulb, pfc_151673, pfc_151507.  Preprocessing mirrors the archived
diversity_extension.representations line for line, with only the scVI seed
(scvi.settings.seed and torch.manual_seed) and the UMAP random_state varied;
the PCA and its 15-NN graph keep their archived seeds.  Seed 0 is checked
against the regenerated arrays (which reproduce the archived scores).
Scores (archived scorer, k=15): E, S plus exact permutation-null E0/S0
(validated against 50-draw Monte Carlo in A2), E*, S*, AUROC.

External anchor (DLPFC only; labels: layer_guesses_151673.csv and the Banksy
barcode_level_layer_map.tsv for 151507): for each candidate, k=15 neighbour
layer purity and ARI of KMeans(7) against layers; compared with tissue- and
expression-reference E* across candidates.  n candidates is small; reported
descriptively.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, dump, e_s_from_hops, edge_survival, hops, ref_sets, separation_survival  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import diversity_extension as dx  # noqa: E402

RES = OUT / "results"
SEEDS = range(5)
LAYER_151673 = Path(os.environ.get("RAAG_SPATIAL_DATA", dx.ROOT / "data" / "spatial")) / "dlpfc_libd_151673/layer_guesses_151673.csv"
LAYER_MAP = dx.SPATIAL_ROOT / "Banksy/data/DLPFC/barcode_level_layer_map.tsv"


def prep(adata):
    import scanpy as sc
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
    view = adata.copy()
    sc.pp.scale(view, max_value=10)
    n_comps = int(min(50, view.n_obs - 1, view.n_vars - 1))
    sc.tl.pca(view, n_comps=n_comps, random_state=42, svd_solver="arpack")
    sc.pp.neighbors(view, n_neighbors=15, n_pcs=n_comps, random_state=0)
    return adata, counts, view, keep


def umap_seed(view, seed):
    import scanpy as sc
    v = view.copy()
    sc.tl.umap(v, random_state=seed, n_components=2)
    return np.asarray(v.obsm["X_umap"], float)


def scvi_seed(adata, counts, seed):
    import scvi
    import torch
    scvi.settings.seed = seed
    torch.manual_seed(seed)
    mv = adata.copy()
    mv.X = counts
    scvi.model.SCVI.setup_anndata(mv)
    model = scvi.model.SCVI(mv, n_latent=10, n_layers=1, gene_likelihood="nb")
    model.train(max_epochs=200, accelerator="gpu", devices=1,
                batch_size=int(min(128, max(32, mv.n_obs // 8))), early_stopping=True)
    return np.asarray(model.get_latent_representation(), float), int(len(model.history["elbo_train"]))


def score(ref, z, rh=None, sets=None):
    rh = hops(ref, 15) if rh is None else rh
    sets = ref_sets(rh) if sets is None else sets
    ex = e_s_from_hops(rh, hops(z, 15), sets)
    e = edge_survival(ref, z, k=15)["edge_survival"]
    s = separation_survival(ref, z, k=15)["separation_survival"]
    assert abs(e - ex["E"]) < 1e-12 and abs(s - ex["S"]) < 1e-12
    return {k: ex[k] for k in ("E", "E0_exact", "E_star", "S", "S0_exact", "S_star", "AUROC_far_vs_edge")}


def layers_for(name, obs_names):
    if name == "pfc_151673":
        lab = pd.read_csv(LAYER_151673)
        m = dict(zip(lab.spot_name.astype(str), lab.layer.astype(str)))
    else:
        lab = pd.read_csv(LAYER_MAP, sep="\t", header=None, names=["barcode", "sample", "layer"])
        lab = lab[lab["sample"].astype(str) == "151507"]
        m = dict(zip(lab.barcode.astype(str), lab.layer.astype(str)))
    out = []
    for b in obs_names:
        out.append(m.get(b, m.get(b + "-1", m.get(b.split("-")[0] + "-1"))))
    return np.asarray([x if x is not None else "NA" for x in out])


def layer_metrics(z, labels):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.neighbors import NearestNeighbors
    ok = labels != "NA"
    zz, ll = z[ok], labels[ok]
    _, idx = NearestNeighbors(n_neighbors=16).fit(zz).kneighbors(zz)
    purity = float((ll[idx[:, 1:]] == ll[:, None]).mean())
    km = KMeans(n_clusters=len(np.unique(ll)), n_init=10, random_state=0).fit_predict(zz)
    return {"layer_knn_purity": purity, "layer_kmeans_ari": float(adjusted_rand_score(ll, km)), "n_labelled": int(ok.sum())}


def main():
    import scanpy as sc
    sc.settings.verbosity = 0
    RES.mkdir(parents=True, exist_ok=True)
    loaders = {"bulb": dx.as_spatial(dx.load_mob), "pfc_151673": dx.as_spatial(dx.load_dlpfc),
               "pfc_151507": lambda: dx.load_visium(
                   "pfc_151507", dx.SPATIAL_ROOT / "Banksy/data/DLPFC/151507/151507_raw_feature_bc_matrix.h5",
                   dx.SPATIAL_ROOT / "Banksy/data/DLPFC/151507/tissue_positions_list.txt")}
    rows, checks = [], []
    for name, loader in loaders.items():
        t0 = time.time()
        adata, coords = loader()
        reg = np.load(OUT / "arrays/diversity" / f"{name}.npz")
        adata2, counts, view, keep = prep(adata)
        tissue = np.asarray(coords, float)[keep]
        pca = np.asarray(view.obsm["X_pca"], float)
        checks.append(dict(object=name, pca_max_abs_diff_vs_regenerated=float(np.abs(pca - reg["pca"]).max()),
                           tissue_equal=bool(np.array_equal(tissue, reg["tissue"]))))
        refs = {"tissue": tissue, "expression": pca}
        rhs = {r: hops(x, 15) for r, x in refs.items()}
        sets = {r: ref_sets(h) for r, h in rhs.items()}
        labels = layers_for(name, reg["obs_names"]) if name.startswith("pfc") else None
        cands = {("pca", 0): pca, ("random", 0): reg["random"], ("permuted_pca", 0): reg["permuted_pca"]}
        for s in SEEDS:
            cands[("umap", s)] = umap_seed(view, s)
            z, ep = scvi_seed(adata2, counts, s)
            cands[("scvi", s)] = z
            if s == 0:
                checks.append(dict(object=name, seed0_umap_max_abs_diff=float(np.abs(cands[("umap", 0)] - reg["umap"]).max()),
                                   seed0_scvi_max_abs_diff=float(np.abs(z - reg["scvi"]).max())))
        for (method, s), z in cands.items():
            row = dict(object=name, method=method, seed=s)
            for r in refs:
                for key, v in score(refs[r], z, rhs[r], sets[r]).items():
                    row[f"{r}_{key}"] = v
            if labels is not None:
                row.update(layer_metrics(z, labels))
            rows.append(row)
        print(name, f"{time.time()-t0:.0f}s", checks[-1], flush=True)
        pd.DataFrame(rows).to_csv(RES / "a5_refit_scores.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "a5_refit_scores.csv", index=False)
    summ = {"checks": checks, "seed_variability": [], "external_anchor": []}
    for name, g in df.groupby("object"):
        for r in ("tissue", "expression"):
            for metric in ("E", "E_star", "S", "S_star", "AUROC_far_vs_edge"):
                col = f"{r}_{metric}"
                sd = {m: float(g[g.method == m][col].std(ddof=1)) for m in ("scvi", "umap")}
                rng_ = {m: [float(g[g.method == m][col].min()), float(g[g.method == m][col].max())] for m in ("scvi", "umap")}
                pca_v = float(g[g.method == "pca"][col].iloc[0])
                summ["seed_variability"].append(dict(object=name, reference=r, metric=col, sd_scvi=sd["scvi"], sd_umap=sd["umap"],
                                                     range_scvi=rng_["scvi"], range_umap=rng_["umap"], pca=pca_v,
                                                     mean_scvi=float(g[g.method == "scvi"][col].mean()),
                                                     mean_umap=float(g[g.method == "umap"][col].mean()),
                                                     random=float(g[g.method == "random"][col].iloc[0])))
        if "layer_knn_purity" in g and g.layer_knn_purity.notna().any():
            from scipy.stats import spearmanr
            h = g[g.method != "permuted_pca"]
            for col in ("tissue_E_star", "expression_E_star", "tissue_S_star", "tissue_AUROC_far_vs_edge", "tissue_E"):
                summ["external_anchor"].append(dict(
                    object=name, score=col, n_candidates=len(h),
                    spearman_vs_layer_purity=float(spearmanr(h[col], h.layer_knn_purity).statistic),
                    spearman_vs_layer_ari=float(spearmanr(h[col], h.layer_kmeans_ari).statistic)))
    dump(RES / "a5_summary.json", summ)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
