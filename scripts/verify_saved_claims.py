#!/usr/bin/env python3
"""Verify the43 canonical numeric claims using the unchanged published2.0.1 checks.

This reads frozen research records; it does not fit a model or modify them.
"""
from pathlib import Path
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
import json
import re
import numpy as np
import pandas as pd


def verify(root):
    ROOT = Path(root).resolve()
    RESULT = ROOT / 'evidence/submission-20261011/results'
    tex = (ROOT / 'paper.tex').read_text()
    errors, report = [], {}
    def fail(message):
        errors.append(message)
    ties = json.loads((RESULT / "tie_claims.json").read_text())
    traj = json.loads((RESULT / "trajectory_summary.json").read_text())
    def outward(values, places):
        """Minimum rounded down and maximum rounded up to the printed precision, so the printed interval contains every value."""
        q = Decimal(1).scaleb(-places)
        lo = Decimal(str(min(values))).quantize(q, rounding=ROUND_FLOOR)
        hi = Decimal(str(max(values))).quantize(q, rounding=ROUND_CEILING)
        return f"{lo:.{places}f}", f"{hi:.{places}f}"


    def rng(values, places):
        lo, hi = outward(values, places)
        return f"{lo}--{hi}"


    def bound(value, places):
        """A printed upper bound rounded up so that it is not below the value."""
        return f"{Decimal(str(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_CEILING):.{places}f}"


    rng_all = ties["largest_range_over_all_62_graphs"]
    shift = ties["largest_tie_complete_shift"]
    checks = [
        ("tie E range", rng_all["E"] <= 0.0015, "0.0015"), ("tie E* range", rng_all["E_star"] <= 0.0021, "0.0021"),
        ("tie R range", rng_all["R"] == 0, "reachability not at all"), ("tie S range", rng_all["S"] <= 0.013, "0.013"),
        ("tie S* range", rng_all["S_star"] <= 0.043, "0.043"), ("tie-complete E* shift", shift["E_star"] <= 0.0017, "0.0017"),
        ("tie-complete S* shift", shift["S_star"] <= 0.040, "0.040"),
        ("UMAP E* range", True, rng(ties["umap_Estar_range_all_graphs"], 4) + " for UMAP"),
        ("random E* range", True, rng(ties["random_Estar_range_all_graphs"], 4) + " for the random projection"),
        ("orderings hold in every graph", all(ties["held_in_every_graph"].values()), "Every ordering reported above held in every graph"),
        ("saved-order scores reproduced", all(v == 0 for v in ties["reproduction_max_abs_difference_vs_saved"].values()), "reproduce the stored scores exactly"),
        ("identity root", traj["identity"]["max_abs_change_with_true_root"] <= 0.0005 and max(traj["identity"]["root_known_pseudotime"]) <= 0.0067, "0.06 or less"),
        ("reversed-case root", traj["arms_reversed_root"]["max_root_known_pseudotime"] <= 0.0601 and traj["arms_reversed_root"]["max_abs_change_with_true_root"] <= 0.010, "0.010"),
        ("reversed cases", traj["arms_reversed_root"]["n_cases"] == 7, "seven cases in all"),
        ("trunk range", round(traj["arms_reversed_root"]["trunk_rho_range"][0], 3) == 0.977 and round(traj["arms_reversed_root"]["trunk_rho_range"][1], 3) == 0.996, "0.977--0.996"),
        ("arm range", round(traj["arms_reversed_root"]["arm_rho_range"][0], 3) == -0.907 and round(traj["arms_reversed_root"]["arm_rho_range"][1], 3) == -0.852, "$-0.852$ to $-0.907$"),
        ("jitter draws", traj["jitter_restored_draws"] == {"0.002": [0, 15], "0.006": [0, 15], "0.01": [5, 15]}, "five of 15 draws"),
        ("three diffusion components", (round(min(traj["probe_ranges"]["diffusion_components|3.0"]), 4), round(max(traj["probe_ranges"]["diffusion_components|3.0"]), 4)) == (0.9995, 0.9997), "0.9995--0.9997 in all three seeds"),
        ("30 neighbours", (round(min(traj["probe_ranges"]["neighbours|30.0"]), 4), round(max(traj["probe_ranges"]["neighbours|30.0"]), 4)) == (0.9986, 0.9995), "0.9986--0.9995"),
        ("shatter components", all(r["n_components"] == 3 and abs(r["largest_component_fraction"] - 1 / 3) < 1e-3 for r in traj["shatter"]), "three components of one third"),
        ("reproduction", traj["reproduction_max_abs_difference"] < 1e-4, "scores only 0.383--0.391"),
    ]
    calibration = json.loads((ROOT / "evidence/measurement-inventory/calibration.json").read_text())
    official = {}
    for row in calibration["rows"]:
        official.setdefault(row["candidate"], []).append(row["stacks"]["scib_official"]["values"]["scib_trajectory_conservation"])


    def span(names, places=3):
        return rng([v for name in names for v in official[name]], places)


    checks += [
        ("official faithful range", True, span(["faithful"]) + " for faithful"),
        ("official matched shatter range", True, span(["shatter_0.5", "shatter_2.5", "shatter_9"]) + " for matched shatter"),
        ("official complete collapse range", True, "complete collapse scores " + span(["collapse_0"])),
        ("official identity range", True, "scores only " + span(["identity"])),
        ("official partial collapse values", True, "(" + ", ".join(f"{v:.3f}" for v in official["collapse_0.5"][:2]) + " and " + f"{official['collapse_0.5'][2]:.3f}" + ")"),
    ]
    checks += [("identity probe range", True, "where the 15-neighbour graph gave " + span(["identity"]) + ".")]
    follow = ROOT / "evidence/followup-20261010/results"
    div = pd.read_csv(follow / "a2_diversity.csv")
    tissue = div[div["reference"] == "tissue"]
    umap_t, random_t = tissue[tissue["candidate"] == "umap"], tissue[tissue["candidate"] == "random"]
    others_t = tissue[tissue["candidate"].isin(["pca", "scvi", "random"])]
    perm = pd.read_csv(follow / "a4_native_permutation_unclipped.csv")
    alloc = pd.read_csv(follow / "a4_coherence_allocations.csv")
    native = alloc[alloc["family"] == "native"]
    scale = json.loads((ROOT / "evidence/scale-correspondence/results.json").read_text())["scale"]
    by_id = {}
    for row in scale:
        if row["native"]:
            by_id.setdefault(row["id"], []).append(row["allocation_coherence_raw"])
    c_ranges = [max(v) - min(v) for v in by_id.values()]
    sections = json.loads((RESULT / "tie_sections.json").read_text())
    fractions = [100 * s["cutoff_tie_fraction"] for s in sections if s["cutoff_tie_fraction"] > 0]
    refit = json.loads((follow / "a5_summary.json").read_text())["seed_variability"]
    e_star_sd = [max(r["sd_scvi"], r["sd_umap"]) for r in refit if r["metric"] == "tissue_E_star"]
    bulb = [r for r in refit if r["object"] == "bulb" and r["metric"] == "tissue_E_star"][0]
    probe = json.loads((ROOT / "evidence/robustness-audit/tied-graph-probe.json").read_text())["joint_reorder"]
    d_e = max(abs(r["changes"].get("edge_survival", 0)) for r in probe)
    d_s = max(abs(r["changes"].get("separation_survival", 0)) for r in probe)
    lo, hi = outward(umap_t["E_star"], 3)
    lo_r, hi_r = outward(random_t["E_star"], 3)
    checks += [
        ("saved-order UMAP E*", True, f"($E^*$ between {lo} and {hi})"),
        ("saved-order random E*", True, f"with $E^*$ between {lo_r} and {hi_r}."),
        ("UMAP tissue E0", True, f"tissue $E_0$ {rng(umap_t[['E0', 'E0_exact']].to_numpy().ravel(), 3)} for UMAP"),
        ("other tissue E0", True, f"against {rng(others_t[['E0', 'E0_exact']].to_numpy().ravel(), 2)} for the others"),
        ("native coherence range", True, f"(range {rng(native['C_unclipped'], 3)})"),
        ("null z range", True, f"lay {rng(perm['z'], 1)} null standard deviations"),
        ("coherence range across objects", True, f"with range across objects {rng(c_ranges, 3)}"),
        ("tie fractions", True, f"equidistant for {rng(fractions, 1)}\\% of the spots"),
        ("refit SD bound (results)", True, f"SDs of at most {bound(max(e_star_sd), 4)},"),
        ("refit SD bound (discussion)", True, f"(standard deviation at most {bound(max(e_star_sd), 4)})"),
        ("bulb interval", True, f"their means and PCA lay within {rng([bulb['mean_scvi'], bulb['mean_umap'], bulb['pca']], 3)}."),
        ("tied probe edge bound", True, f"edge survival by up to {bound(d_e, 4)}"),
        ("tied probe separation bound", True, f"separation by up to {bound(d_s, 4)} in absolute value"),
    ]
    tail = json.loads((ROOT / "evidence/raag-v3/full-20261009-c/report.json").read_text())
    smooth = {row["epsilon_decimal"]: row["actual"] for row in tail["rows"] if row["control"] == "smooth"}
    first, last = smooth["1"], smooth["1E-10"]
    checks += [
        ("tail coherence", round(first["allocation_coherence"], 3) == round(last["allocation_coherence"], 3) == 0.964, "coherence 0.964 at both tail masses"),
        ("tail divergence", f"{first['allocation_pair_js']:.2e}" == "2.65e-02" and f"{last['allocation_pair_js']:.2e}" == "2.65e-12", "$2.65\\times10^{-2}$ to $2.65\\times10^{-12}$"),
        ("tail total variation", f"{first['allocation_mean_pair_tv']:.3f}" == "0.158" and f"{last['allocation_mean_pair_tv']:.2e}" == "1.58e-11", "from 0.158 to $1.58\\times10^{-11}$"),
    ]
    for label, ok, snippet in checks:
        if not ok:
            fail(f"number check failed: {label}")
        if snippet not in tex:
            fail(f"text does not contain {snippet!r} ({label})")
    report["number_checks"] = len(checks)

    report.update(errors=errors, status="PASS" if not errors else "FAIL")
    return report
