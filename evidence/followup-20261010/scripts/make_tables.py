#!/usr/bin/env python3
"""Compact markdown tables (results/tables.md) summarising the saved follow-up results; reads only results/*."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT  # noqa: E402

import pandas as pd  # noqa: E402

R = OUT / "results"


def md(df, floatfmt=3):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floatfmt}g}" if abs(v) < 1e-2 and v != 0 else f"{v:.{floatfmt}f}")
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.astype(str).values]
    return "\n".join([head, sep, *rows])


def main():
    out = []
    s = pd.read_csv(R / "a3_graded_response_summary.csv")
    metrics = ["R", "n_cand_components", "E", "E_star", "AUROC", "geodesic_spearman", "trustworthiness", "LCMC",
               "distance_spearman", "Q_global", "AUC_RNX", "steadiness", "cohesiveness", "pair_precision"]
    for col, title in (("A_detect_3sd", "Detection amplitude (smallest grid shift with |paired mean change| > 3 SD over 20 seeds, same sign as full response)"),
                       ("A50_median", "A50: median over seeds of the shift giving 50% of the full (amplitude 9 / full merge) response"),
                       ("full_change_mean", "Full response: mean paired change faithful -> amplitude 9 (clusters: full merge)")):
        t = s[s.metric.isin(metrics)].pivot_table(index=["topology", "n_key", "k"], columns="metric", values=col)[metrics].reset_index()
        out.append(f"### {title}\n\n" + md(t) + "\n")
    t = s[s.metric.isin(["R", "E_star", "distance_spearman", "Q_global"])].pivot_table(
        index=["topology", "n_key", "k"], columns="metric", values="A50_seed_sd").reset_index()
    out.append("### Seed SD of A50\n\n" + md(t) + "\n")
    w = pd.read_csv(R / "a3_stored_comparators_wide.csv")
    cols = ["reachability_survival", "audit_candidate_lcc", "edge_survival", "separation_survival", "audit_trustworthiness", "zadu_lcmc",
            "audit_distance_spearman", "audit_geodesic_spearman", "audit_q_global", "audit_auc_ln_rnx", "zadu_steadiness", "zadu_cohesiveness",
            "audit_trajectory_conservation_penalized", "scib_trajectory_conservation", "scib_graph_connectivity"]
    order = ["identity", "faithful", "rotated_scaled", "shatter_0.5", "shatter_2.5", "shatter_9", "collapse_0.5", "collapse_0",
             "noise_0.06", "noise_0.18", "permuted", "historical_unmatched_shatter"]
    t = w[w.seed == 42].set_index("candidate").loc[order, cols].T.reset_index().rename(columns={"index": "metric (seed 42)"})
    out.append("### Stored comparator panel, calibration.json, seed 42\n\n" + md(t) + "\n")
    d = pd.read_csv(R / "a2_diversity.csv")
    for m in ("E", "E0", "E_star", "S", "S0", "S_star", "AUROC"):
        t = d.pivot_table(index=["reference", "object"], columns="candidate", values=m).reset_index()
        out.append(f"### Diversity {m} (k=15)\n\n" + md(t) + "\n")
    f = pd.read_csv(R / "a4_family_means.csv")
    t = f.pivot(index="id", columns="family", values="C_unclipped").reset_index()
    out.append("### A4 unclipped coherence by allocation family\n\n" + md(t) + "\n")
    b = pd.read_csv(R / "a4_bootstrap_differences.csv")
    out.append("### A4 native minus baseline (unclipped C), 95% cell-bootstrap interval\n\n" + md(b) + "\n")
    tb = pd.read_csv(R / "a4b_tempered_gmm.csv")
    out.append("### A4b perplexity-matched (tempered) GMM baselines\n\n" + md(tb[["id", "gmm", "C_native", "C_gmm_raw", "C_gmm_tempered", "native_pf", "pf_tempered"]]) + "\n")
    if (R / "a5_refit_scores.csv").exists():
        a5 = pd.read_csv(R / "a5_refit_scores.csv")
        keep = [c for c in ["object", "method", "seed", "tissue_E", "tissue_E_star", "tissue_S_star", "tissue_AUROC_far_vs_edge",
                            "expression_E", "expression_E_star", "expression_S_star", "layer_knn_purity", "layer_kmeans_ari"] if c in a5]
        out.append("### A5 refits (scVI/UMAP seeds 0-4) and layer metrics\n\n" + md(a5[keep]) + "\n")
    (R / "tables.md").write_text("\n".join(out))
    print("wrote", R / "tables.md")


if __name__ == "__main__":
    main()
