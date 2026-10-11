#!/usr/bin/env python3
"""Write MANIFEST.json: SHA256 of every input read, every script and every output."""
from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, PAPER, UPSTREAM, dump, sha256  # noqa: E402

SPATIAL = Path(os.environ.get("RAAG_SPATIAL_DATA", PAPER / "data" / "spatial"))
SR = Path(os.environ.get("RAAG_SPATIAL_EXTERNAL", PAPER / "data" / "spatial-external"))


def label(path: Path) -> str:
    """Portable name for a recorded file: never an absolute workstation path."""
    for base, name in ((PAPER, ""), (SPATIAL, "spatial-data/"), (SR, "spatial-external/"), (UPSTREAM, "upstream/")):
        try:
            return name + str(Path(path).relative_to(base))
        except ValueError:
            continue
    return Path(path).name


def main():
    inputs = [
        PAPER / "scripts/local_raag.py", PAPER / "scripts/diversity_extension.py", PAPER / "scripts/spatial_reference_case.py",
        UPSTREAM / "audit/upstream_validation.py", UPSTREAM / "audit/reference_metrics.py",
        PAPER / "evidence/measurement-inventory/calibration.json",
        PAPER / "evidence/scale-correspondence/inputs.json", PAPER / "evidence/scale-correspondence/results.json",
        PAPER / "evidence/diversity-20261009/results.json", PAPER / "evidence/diversity-20261009/object_views.csv",
        *sorted((PAPER / "evidence/measurement-inventory/calibration_arrays").glob("*.npz")),
        *sorted((PAPER / "evidence/scale-correspondence/inputs").glob("*.npz")),
        *sorted((PAPER / "evidence/diversity-20261009/pbmc3k_matrix").glob("*")),
        SPATIAL / "st_stahl2016_mob_rep11/Rep11_MOB_0_expr.txt", SPATIAL / "st_stahl2016_mob_rep11/Rep11_MOB_0_location.txt",
        SPATIAL / "dlpfc_libd_151673/151673_filtered_feature_bc_matrix.h5", SPATIAL / "dlpfc_libd_151673/tissue_positions_list.txt",
        SPATIAL / "dlpfc_libd_151673/layer_guesses_151673.csv",
        *sorted((SPATIAL / "visium_bclm_A_0069226").glob("A_0069226_*")),
        SR / "Banksy/data/DLPFC/151507/151507_raw_feature_bc_matrix.h5", SR / "Banksy/data/DLPFC/151507/tissue_positions_list.txt",
        SR / "Banksy/data/DLPFC/barcode_level_layer_map.tsv",
        SR / "SEDR_analyses/data/Human_Lymph_Node/filtered_feature_bc_matrix.h5",
        SR / "SEDR_analyses/data/Human_Lymph_Node/spatial/tissue_positions_list.csv",
        SR / "SEDR_analyses/data/BRCA1/V1_Human_Breast_Cancer_Block_A_Section_1/filtered_feature_bc_matrix.h5",
        SR / "SEDR_analyses/data/BRCA1/V1_Human_Breast_Cancer_Block_A_Section_1/spatial/tissue_positions_list.csv",
        SR / "SEDR_analyses/data/Human_Ovarian_Cancer/filtered_feature_bc_matrix.h5",
        SR / "SEDR_analyses/data/Human_Ovarian_Cancer/spatial/tissue_positions_list.csv",
    ]
    rec = {"inputs": {}, "scripts": {}, "outputs": {}}
    for p in inputs:
        rec["inputs"][label(p)] = sha256(p) if p.exists() else "MISSING"
    for p in sorted((OUT / "scripts").glob("*.py")):
        rec["scripts"][str(p.relative_to(OUT))] = sha256(p)
    for sub in ("results", "arrays"):
        for p in sorted((OUT / sub).rglob("*")):
            if p.is_file():
                rec["outputs"][str(p.relative_to(OUT))] = sha256(p)
    if (OUT / "README.md").exists():
        rec["outputs"]["README.md"] = sha256(OUT / "README.md")
    rec["environment"] = {"python_version": platform.python_version(),
                          **{n: importlib.metadata.version(n) for n in
                             ("numpy", "scipy", "scikit-learn", "scanpy", "anndata", "umap-learn", "scvi-tools", "torch", "zadu", "pandas")}}
    dump(OUT / "MANIFEST.json", rec)
    print(len(rec["inputs"]), "inputs", sum(v == "MISSING" for v in rec["inputs"].values()), "missing;",
          len(rec["outputs"]), "outputs")


if __name__ == "__main__":
    main()
