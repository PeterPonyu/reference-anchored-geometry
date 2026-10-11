"""Where does the official scIB trajectory-conservation value come from on the synthetic fixture?

Reads the saved fixture arrays (three seeds, twelve candidates) and the saved official scores.

  1. Reproduces every saved official value (scib 1.1.7 + scanpy; neighbour graph from X_emb, k=15).
  2. Records the root cell the official heuristic selects and recomputes the same score after
     replacing it by the cell with the smallest known pseudotime (one-component graphs).
  3. Reports the within-branch Spearman correlation between known and diffusion pseudotime.
  4. Probes the identity control: number of diffusion components, neighbour count, extra jitter.

Run:  python trajectory_diagnosis.py     (about one minute)      Outputs go to ../results.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.dont_write_bytecode = True
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scib
from scib.metrics import trajectory as scib_trajectory
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[2]
ARRAYS = PAPER / "evidence/measurement-inventory/calibration_arrays"
CALIBRATION = PAPER / "evidence/measurement-inventory/calibration.json"
RESULTS = HERE.parent / "results"
SEEDS = (7, 17, 42)
NON_CANDIDATE = ("coords", "pseudotime", "branch")


def build(emb, branch, pseudotime, k=15):
    n = len(pseudotime)
    names = [f"c{i}" for i in range(n)]
    pre = ad.AnnData(np.zeros((n, 1)), obs=pd.DataFrame(index=names))
    pre.obs["branch"] = pd.Categorical(branch.astype(str))
    pre.obs["dpt_pseudotime"] = np.asarray(pseudotime, dtype=float)
    post = ad.AnnData(np.zeros((n, 1)), obs=pd.DataFrame(index=names))
    post.obs["branch"] = pre.obs["branch"].values
    post.obsm["X_emb"] = np.asarray(emb, dtype=float)
    sc.pp.neighbors(post, n_neighbors=k, use_rep="X_emb")
    return pre, post


def dpt_values(post, root, n_dcs=10):
    p = post.copy()
    sc.tl.diffmap(p)
    p.uns["iroot"] = int(root)
    sc.tl.dpt(p, n_dcs=n_dcs)
    return p.obs["dpt_pseudotime"].values


def branch_rho(t, branch, dpt):
    return [float(spearmanr(t[branch == b], dpt[branch == b]).statistic) for b in range(3)]


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    saved = {(r["seed"], r["candidate"]): r["stacks"]["scib_official"]["values"]["scib_trajectory_conservation"]
             for r in json.load(open(CALIBRATION))["rows"]}
    rows, branch_rows = [], []
    for seed in SEEDS:
        z = np.load(ARRAYS / f"seed{seed}.npz")
        t, branch = z["pseudotime"].astype(float), z["branch"]
        for cand in [c for c in z.files if c not in NON_CANDIDATE]:
            pre, post = build(z[cand], branch, t)
            official = float(scib.me.trajectory_conservation(pre, post.copy(), label_key="branch"))
            n_comp, labels = connected_components(post.obsp["connectivities"], directed=False)
            largest = float(np.bincount(labels).max() / len(labels))
            iroot, post_ti = scib_trajectory.get_root(pre, post.copy(), "branch", "dpt_pseudotime")
            root_row = int(post_ti.obs_names[iroot][1:])
            row = dict(seed=seed, candidate=cand, saved_official=saved.get((seed, cand), np.nan),
                       reproduced_official=official, n_components=int(n_comp),
                       largest_component_fraction=largest, root_known_pseudotime=float(t[root_row]),
                       root_pseudotime_percentile=float((t < t[root_row]).mean()))
            if n_comp == 1:
                chosen = dpt_values(post, root_row)
                true_root = dpt_values(post, int(np.argmin(t)))
                row["score_chosen_root"] = (1 + spearmanr(t, chosen).statistic) / 2
                row["score_true_root"] = (1 + spearmanr(t, true_root).statistic) / 2
                trunk, arm1, arm2 = branch_rho(t, branch, true_root)
                branch_rows.append(dict(seed=seed, candidate=cand, official=official, trunk=trunk, arm1=arm1,
                                        arm2=arm2, arms_reversed=bool(arm1 < 0 and arm2 < 0)))
            rows.append(row)
    diagnosis = pd.DataFrame(rows)
    diagnosis.to_csv(RESULTS / "trajectory_diagnosis.csv", index=False)
    branches = pd.DataFrame(branch_rows)
    branches.to_csv(RESULTS / "trajectory_branch_rho.csv", index=False)

    # Probes on the identity control and the faithful candidate.
    probes = []

    def probe(seed, cand, setting, value, emb, n_dcs=10, k=15, draw=0):
        z = np.load(ARRAYS / f"seed{seed}.npz")
        t, branch = z["pseudotime"].astype(float), z["branch"]
        _, post = build(emb(z), branch, t, k=k)
        d = dpt_values(post, int(np.argmin(t)), n_dcs=n_dcs)
        trunk, arm1, arm2 = branch_rho(t, branch, d)
        probes.append(dict(seed=seed, candidate=cand, setting=setting, value=value, draw=draw,
                           score=(1 + spearmanr(t, d).statistic) / 2, trunk=trunk, arm1=arm1, arm2=arm2))

    for seed in SEEDS:
        for n_dcs in (3, 4, 10):
            probe(seed, "identity", "diffusion_components", n_dcs, lambda z: z["identity"], n_dcs=n_dcs)
        for k in (15, 30):
            probe(seed, "identity", "neighbours", k, lambda z: z["identity"], k=k)
        for n_dcs in (3, 10):
            probe(seed, "faithful", "diffusion_components", n_dcs, lambda z: z["faithful"], n_dcs=n_dcs)
        for sd in (0.002, 0.006, 0.01):
            for draw in range(5):
                rng = np.random.default_rng(1000 * seed + draw)
                probe(seed, "identity", "extra_jitter_sd", sd,
                      lambda z, sd=sd, rng=rng: z["identity"] + rng.normal(0, sd, z["identity"].shape), draw=draw)
    probes = pd.DataFrame(probes)
    probes.to_csv(RESULTS / "trajectory_probes.csv", index=False)

    ident = diagnosis[diagnosis.candidate == "identity"]
    single = diagnosis[diagnosis.n_components == 1]
    reversed_rows = branches[branches.arms_reversed & ~branches.candidate.isin(["permuted"])]
    reversed_diag = diagnosis.merge(reversed_rows[["seed", "candidate"]], on=["seed", "candidate"])
    jitter = probes[(probes.setting == "extra_jitter_sd")]
    restored = {str(sd): [int((g.score > 0.9).sum()), int(len(g))] for sd, g in jitter.groupby("value")}
    summary = dict(
        versions=dict(scib=scib.__version__, scanpy=sc.__version__, anndata=ad.__version__,
                      numpy=np.__version__, pandas=pd.__version__),
        reproduction_max_abs_difference=float((diagnosis.reproduced_official - diagnosis.saved_official).abs().max()),
        identity=dict(
            official_scores=ident.reproduced_official.round(4).tolist(),
            root_known_pseudotime=ident.root_known_pseudotime.round(4).tolist(),
            max_abs_change_with_true_root=float((ident.score_chosen_root - ident.score_true_root).abs().max()),
            trunk_rho_range=[float(branches[branches.candidate == "identity"].trunk.min()),
                             float(branches[branches.candidate == "identity"].trunk.max())],
            arm_rho_range=[float(branches[branches.candidate == "identity"][["arm1", "arm2"]].min().min()),
                           float(branches[branches.candidate == "identity"][["arm1", "arm2"]].max().max())]),
        one_component_candidates=dict(
            max_abs_change_with_true_root=float((single.score_chosen_root - single.score_true_root).abs().max()),
            max_root_percentile=float(single.root_pseudotime_percentile.max())),
        arms_reversed=reversed_rows.assign(official=reversed_rows.official.round(4))[
            ["seed", "candidate", "official"]].to_dict("records"),
        arms_reversed_root=dict(
            n_cases=int(len(reversed_diag)),
            max_root_known_pseudotime=float(reversed_diag.root_known_pseudotime.max()),
            max_root_pseudotime_percentile=float(reversed_diag.root_pseudotime_percentile.max()),
            max_abs_change_with_true_root=float((reversed_diag.score_chosen_root - reversed_diag.score_true_root).abs().max()),
            trunk_rho_range=[float(reversed_rows.trunk.min()), float(reversed_rows.trunk.max())],
            arm_rho_range=[float(reversed_rows[["arm1", "arm2"]].min().min()),
                           float(reversed_rows[["arm1", "arm2"]].max().max())]),
        jitter_restored_draws=restored,
        arms_correct_candidates=sorted(set(branches[~branches.arms_reversed].candidate)),
        shatter=diagnosis[diagnosis.candidate.str.startswith("shatter")][
            ["seed", "candidate", "reproduced_official", "n_components", "largest_component_fraction"]].round(4)
            .to_dict("records"),
        probe_ranges={f"{s}|{v}": [float(g.score.min()), float(g.score.max())]
                      for (s, v), g in probes[probes.candidate == "identity"].groupby(["setting", "value"])},
        faithful_probe_ranges={str(v): [float(g.score.min()), float(g.score.max())]
                               for v, g in probes[probes.candidate == "faithful"].groupby("value")},
    )
    json.dump(summary, open(RESULTS / "trajectory_summary.json", "w"), indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
