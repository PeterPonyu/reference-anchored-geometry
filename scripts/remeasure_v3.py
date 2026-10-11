#!/usr/bin/env python3
"""RAAG-v3 remeasurement (P2) of the archived fixed-object batteries.

Re-scores the 24 hash-verified objects used by the archived v2 scale/correspondence
protocol (216 scale settings, 180 correspondence perturbations, 48 controls) and
both 999-draw native row-order streams with the v3 scorer, then compares every
value with the archived v2 result under tolerances frozen in the admission
receipt *before* execution.  It never writes into historical evidence directories
and refuses an existing output directory.  No training, GPU, network or new data.

    python3 scripts/remeasure_v3.py --admission-receipt PATH --output-dir evidence/raag-v3/remeasure-unique
"""
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "2"
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

import local_raag_v3 as v3

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evidence/scale-correspondence"
GEOMETRY = ("reachability_survival", "edge_survival", "separation_survival")
EXACT_FIELDS = ("n_reference_connected_pairs", "n_reference_edges", "n_far_pairs",
                "separation_far_cutoff_hops", "separation_threshold_hops")
STREAM_SEEDS = (20260922, 104729)
N_PERM = 999


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def peak_rss_mib():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    return None


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def compact(result):
    """Keep scalar scores, statuses and graph tie/identity bookkeeping."""
    keep = {key: value for key, value in result.items()
            if not isinstance(value, dict) and key != "ordered_ids_sha256"}
    for side in ("reference_graph", "candidate_graph"):
        graph = result.get(side)
        if graph:
            for field in ("cutoff_tie_rows", "maximum_cutoff_tie_size", "zero_distance_nonself_pairs",
                          "canonical_edges_sha256", "graph_status"):
                keep[f"{side}.{field}"] = graph[field]
    return keep


def within(actual, expected, atol, rtol):
    if actual is None or expected is None:
        return actual is None and expected is None
    return abs(actual - expected) <= atol + rtol * abs(expected)


def verify_admission(path, output_dir):
    receipt = json.loads(Path(path).read_text())
    assert receipt["schema"] == "raag-v3-p2-remeasurement-admission-v1"
    for name, digest in receipt["bound_sha256"].items():
        assert sha(ROOT / name) == digest, f"admission hash mismatch: {name}"
    assert not Path(output_dir).exists(), "refusing an existing output directory"
    assert Path(output_dir).resolve().is_relative_to(ROOT / "evidence/raag-v3"), "output must stay under evidence/raag-v3"
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission-receipt", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    receipt = verify_admission(args.admission_receipt, ROOT / args.output_dir)
    tol = receipt["frozen_tolerances"]
    start = time.perf_counter()
    meta = json.loads((SOURCE / "inputs.json").read_text())
    archived = json.loads((SOURCE / "results.json").read_text())
    scale_index = {(r["id"], r["k"], r["rank"]): r for r in archived["scale"]}
    corrupt_index = {(r["id"], r["corruption_seed"], r["requested_fraction"]): r for r in archived["correspondence"]}
    control_index = {(r["id"], r["control"]): r for r in archived["controls"]}
    assert len(meta) == 24 and len(scale_index) == 216 and len(corrupt_index) == 180 and len(control_index) == 48

    scale, controls, corrupt, mismatches = [], [], [], []
    baselines, objects = {}, {}

    def check(kind, key, new, old, native):
        for name in GEOMETRY:
            if not within(new.get(name), old.get(name), tol["geometry_atol"], 0):
                mismatches.append(dict(kind=kind, key=key, field=name, v3=new.get(name), v2=old.get(name)))
        for name in EXACT_FIELDS:
            if new.get(name) != old.get(name):
                mismatches.append(dict(kind=kind, key=key, field=name, v3=new.get(name), v2=old.get(name)))
        if native:
            if not within(new.get("allocation_coherence_raw"), old.get("allocation_coherence_raw"),
                          tol["coherence_atol"], tol["coherence_rtol"]):
                mismatches.append(dict(kind=kind, key=key, field="allocation_coherence_raw",
                                       v3=new.get("allocation_coherence_raw"), v2=old.get("allocation_coherence_raw")))
            if bool(new["allocation_near_uniform"]) != bool(old["allocation_near_uniform"]):
                mismatches.append(dict(kind=kind, key=key, field="allocation_near_uniform"))
            if bool(new["allocation_degenerate"]) != bool(old["allocation_degenerate"]):
                mismatches.append(dict(kind=kind, key=key, field="allocation_degenerate"))

    for item in meta:
        path = SOURCE / item["file"]
        assert sha(path) == item["sha256"]
        with np.load(path) as data:
            x, z, test_idx = data["reference"], data["latent"], data["test_idx"]
        ids = v3.cell_ids_from_indices(item["dataset"], test_idx, len(x))
        objects[item["id"]] = (x, z, ids)
        native = bool(item["native"])
        for k in (5, 15, 30):
            for rank in (10, 25, 50):
                out = v3.raag_battery(x[:, :rank], z, reference_ids=ids, candidate_ids=ids, k=k, already_simplex=native)
                scale.append(dict(id=item["id"], dataset=item["dataset"], seed=item["seed"], native=native,
                                  k=k, rank=rank, **compact(out)))
                check("scale", f"{item['id']}|k{k}|d{rank}", out, scale_index[(item["id"], k, rank)], native)
                if (k, rank) == (15, 50):
                    baselines[item["id"]] = out
        base = baselines[item["id"]]
        # Rotation and rescaling: same orthogonal transform for every row, continuous output.
        q, _ = np.linalg.qr(np.random.default_rng(771).normal(size=(z.shape[1], z.shape[1])))
        rot = v3.raag_battery(x, 2.5 * (np.asarray(z, float) @ q), reference_ids=ids, candidate_ids=ids, k=15, already_simplex=False)
        rot_delta = {m: rot[m] - base[m] for m in GEOMETRY}
        controls.append(dict(id=item["id"], control="rotation_rescale", geometry_delta=rot_delta))
        # Joint reorder: rows and persistent IDs move together; graphs must be invariant.
        ix = np.random.default_rng(772).permutation(len(x))
        joint = v3.raag_battery(x[ix], z[ix], reference_ids=[ids[i] for i in ix], candidate_ids=[ids[i] for i in ix],
                                k=15, already_simplex=native)
        keys = GEOMETRY + (("allocation_coherence_raw",) if native else ())
        joint_delta = {m: joint[m] - base[m] for m in keys}
        graph_invariant = (joint["reference_graph"]["canonical_edges_sha256"] == base["reference_graph"]["canonical_edges_sha256"]
                           and joint["candidate_graph"]["canonical_edges_sha256"] == base["candidate_graph"]["canonical_edges_sha256"])
        controls.append(dict(id=item["id"], control="joint_row_permutation", delta=joint_delta,
                             canonical_graph_hashes_invariant=graph_invariant))
        for kind, new_delta in (("rotation_rescale", rot_delta), ("joint_row_permutation", joint_delta)):
            for name, value in new_delta.items():
                if abs(value) > tol["control_atol"]:
                    mismatches.append(dict(kind="control", key=f"{item['id']}|{kind}", field=name, v3=value, v2="expected 0"))
        if not graph_invariant:
            mismatches.append(dict(kind="control", key=f"{item['id']}|joint_row_permutation", field="canonical_graph_hash"))
        if native:
            for seed in (0, 17, 42):
                order = np.random.default_rng(seed).permutation(len(z))
                for fraction in (0, .1, .25, .5, 1):
                    n = int(fraction * len(z))
                    idx = np.arange(len(z)); chosen = order[:n]
                    if n >= 2:
                        idx[chosen] = np.roll(chosen, 1)
                    assert np.array_equal(np.sort(idx), np.arange(len(z)))
                    # Recipients keep their persistent IDs; donors' allocation rows are attached to them.
                    out = v3.raag_battery(x, z[idx], reference_ids=ids, candidate_ids=ids, k=15, already_simplex=True)
                    corrupt.append(dict(id=item["id"], dataset=item["dataset"], model_seed=item["seed"], corruption_seed=seed,
                                        requested_fraction=fraction, changed_fraction=float(np.mean(idx != np.arange(len(z)))),
                                        **compact(out)))
                    check("correspondence", f"{item['id']}|s{seed}|f{fraction}", out,
                          corrupt_index[(item["id"], seed, fraction)], True)
        print(item["id"], "done", flush=True)

    # Both 999-draw native row-order streams, scored from the v3 divergences.
    perms = []
    for item in meta:
        if not item["native"]:
            continue
        x, z, ids = objects[item["id"]]
        p, _ = v3.to_simplex(z, already_simplex=True)
        n = len(p)
        graph = v3.build_knn_graph(x, 15, cell_ids=ids).graph
        src, dst = graph.nonzero(); keep = src < dst; src, dst = src[keep], dst[keep]
        js = np.zeros((n, n))
        for i in range(n - 1):
            js[i, i + 1:] = v3._jensen_shannon(np.broadcast_to(p[i], p[i + 1:].shape), p[i + 1:])
        js += js.T
        denominator = math.fsum(js[np.triu_indices(n, 1)].tolist()) / (n * (n - 1) // 2)
        score = lambda order: float(np.clip(1 - js[order[src], order[dst]].mean() / denominator, 0, 1))
        observed = score(np.arange(n))
        battery = baselines[item["id"]]["allocation_coherence_raw"]
        assert abs(observed - battery) < tol["coherence_atol"], (item["id"], observed, battery)
        archived_perm = next(r for r in json.loads((ROOT / "evidence/paired-grid-permutation/results.json").read_text())["native_permutations"]
                             if r["id"] == item["id"])
        streams = []
        for seed in STREAM_SEEDS:
            rng = np.random.default_rng(seed)
            null = np.array([score(rng.permutation(n)) for _ in range(N_PERM)])
            exceed = int(np.count_nonzero(null >= observed))
            streams.append(dict(seed=seed, draws=N_PERM, exceedances=exceed, monte_carlo_p=(exceed + 1) / (N_PERM + 1),
                                null_min=float(null.min()), null_median=float(np.median(null)), null_max=float(null.max())))
        primary = streams[0]
        if primary["exceedances"] != archived_perm["null_ge_observed"]:
            mismatches.append(dict(kind="permutation", key=item["id"], field="exceedances",
                                   v3=primary["exceedances"], v2=archived_perm["null_ge_observed"]))
        if not within(primary["null_max"], archived_perm["null_max"], tol["coherence_atol"], tol["coherence_rtol"]):
            mismatches.append(dict(kind="permutation", key=item["id"], field="null_max", v3=primary["null_max"], v2=archived_perm["null_max"]))
        if not within(observed, archived_perm["observed"], tol["coherence_atol"], tol["coherence_rtol"]):
            mismatches.append(dict(kind="permutation", key=item["id"], field="observed", v3=observed, v2=archived_perm["observed"]))
        perms.append(dict(id=item["id"], observed=observed, exact_denominator=denominator, n_cells=n,
                          n_reference_edges=int(len(src)), streams=streams, archived_exceedances=archived_perm["null_ge_observed"]))
        print(item["id"], "streams done", flush=True)

    elapsed = time.perf_counter() - start
    rss = peak_rss_mib()
    statuses = {}
    for row in scale + corrupt:
        statuses[row["allocation_status"]] = statuses.get(row["allocation_status"], 0) + 1
    abs_diff = {}
    for row in scale:
        old = scale_index[(row["id"], row["k"], row["rank"])]
        for name in GEOMETRY + (("allocation_coherence_raw",) if row["native"] else ()):
            abs_diff[name] = max(abs_diff.get(name, 0.0), abs(row[name] - old[name]))
    for row in corrupt:
        old = corrupt_index[(row["id"], row["corruption_seed"], row["requested_fraction"])]
        for name in GEOMETRY + ("allocation_coherence_raw",):
            abs_diff[name] = max(abs_diff.get(name, 0.0), abs(row[name] - old[name]))
    summary = dict(
        schema="raag-v3-p2-remeasurement-report-v1", status="PASS_AGREES_WITH_ARCHIVED_V2" if not mismatches else "FAIL_DISAGREES_WITH_ARCHIVED_V2",
        n_scale=len(scale), n_correspondence=len(corrupt), n_controls=len(controls),
        n_batteries=len(scale) + len(corrupt) + len(controls), n_permutation_objects=len(perms),
        n_mismatches=len(mismatches), maximum_absolute_difference_vs_v2=abs_diff,
        allocation_status_counts_v3=statuses,
        cutoff_tie_rows_total=sum(r["reference_graph.cutoff_tie_rows"] + r["candidate_graph.cutoff_tie_rows"] for r in scale),
        zero_distance_nonself_pairs_total=sum(r["reference_graph.zero_distance_nonself_pairs"] + r["candidate_graph.zero_distance_nonself_pairs"] for r in scale),
        permutation_exceedances_v3=[[r["id"], r["streams"][0]["exceedances"], r["streams"][1]["exceedances"]] for r in perms],
        permutation_exceedances_archived=[[r["id"], r["archived_exceedances"]] for r in perms],
        elapsed_seconds=elapsed, peak_rss_mib=rss, scorer_version=v3.RAAG_VERSION,
        ceilings=receipt["ceilings"], within_ceilings=bool(elapsed <= receipt["ceilings"]["wall_seconds"] and rss <= receipt["ceilings"]["peak_rss_mib"]))
    out = ROOT / args.output_dir
    out.mkdir(parents=True)
    (out / "results.json").write_text(json.dumps(clean(dict(scale=scale, correspondence=corrupt, controls=controls,
                                                           permutations=perms, optimizer_updates=0, network_access=False,
                                                           driver_sha256=sha(__file__), scorer_sha256=sha(ROOT / "scripts/local_raag_v3.py"),
                                                           admission_receipt_sha256=sha(args.admission_receipt))), indent=2, allow_nan=False) + "\n")
    (out / "report.json").write_text(json.dumps(clean(dict(summary=summary, mismatches=mismatches)), indent=2, allow_nan=False) + "\n")
    print(json.dumps(clean(summary), indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
