#!/usr/bin/env python3
"""Admitted, separately routed P1 fixtures; source authored, not yet executed.

This runner does not train, load checkpoints, replay archived batteries or alter
historical results. An admission JSON is a recorded authorization, not something
this script grants itself. Do not create an APPROVED receipt without the actual
user/workflow decision and fresh resource checks required by the saved plan.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "validation/PROTOCOL_RAAG_V3.md", "scripts/local_raag_v3.py",
    "validation/raag_v3_oracle.py", "validation/test_raag_v3.py",
    "scripts/validate_raag_v3.py",
)
JSD_TOLERANCE = (1e-28, 1e-10)
TV_TOLERANCE = (2e-16, 1e-10)
PERPLEXITY_TOLERANCE = (1e-12, 1e-12)
RATIO_TOLERANCE = (1e-12, 1e-10)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def peak_rss_mib() -> float:
    """Post-exec process peak; Linux rusage can retain the launcher's peak."""
    status = Path('/proc/self/status')
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith('VmHWM:'):
                return float(line.split()[1]) / 1024
        raise RuntimeError('Linux process peak-RSS measurement is unavailable')
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def plain(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    return value


def read_admission(receipt_path: Path, pilot: bool) -> tuple[dict, dict[str, str]]:
    hashes = {name: digest(ROOT / name) for name in SOURCE_FILES}
    receipt = json.loads(receipt_path.read_text())
    mode = "pilot" if pilot else "full"
    if receipt.get("status") != "APPROVED" or receipt.get("task") != "raag_v3_p1_fixtures" or receipt.get("mode") != mode:
        raise ValueError("this exact fixture mode has no approved admission receipt")
    if receipt.get("source_sha256") != hashes:
        raise ValueError("admission receipt does not bind the current source/protocol bytes")
    if not receipt.get("authorization_basis") or not receipt.get("drain_resolution"):
        raise ValueError("record the actual authorization and drain-resolution decision")
    if receipt.get("numerical_library_threads") != 1:
        raise ValueError("P1 fixtures require one numerical-library thread")
    wall, rss = receipt.get("max_wall_seconds"), receipt.get("max_peak_rss_mib")
    if type(wall) is not int or not 1 <= wall <= (120 if pilot else 600):
        raise ValueError("invalid admitted wall budget")
    if type(rss) is not int or not 1 <= rss <= 512:
        raise ValueError("invalid admitted RSS ceiling")
    if not pilot:
        pilot_info = receipt.get("successful_pilot", {})
        pilot_path = Path(pilot_info.get("report", ""))
        pilot_base = (ROOT / "evidence/raag-v3").resolve()
        if not pilot_path.is_absolute() or not pilot_path.resolve().is_relative_to(pilot_base):
            raise ValueError("full admission must identify a separately routed pilot report")
        if digest(pilot_path) != pilot_info.get("sha256"):
            raise ValueError("pilot report identity differs")
        pilot_report = json.loads(pilot_path.read_text())
        if pilot_report.get("status") != "PASS_PILOT" or pilot_report.get("source_sha256") != hashes:
            raise ValueError("a successful pilot on these exact sources is required")
    return receipt, hashes


def output_directory(value: Path) -> Path:
    target = value if value.is_absolute() else ROOT / value
    base = ROOT / "evidence/raag-v3"
    if not target.resolve().is_relative_to(base.resolve()) or target.resolve() == base.resolve():
        raise ValueError("output must be a new subdirectory of evidence/raag-v3")
    if target.exists():
        raise FileExistsError("output already exists; historical/previous bytes will not be overwritten")
    for parent in (target, *target.parents):
        if parent.is_symlink():
            raise ValueError("symlink output paths are forbidden")
        if parent == ROOT:
            break
    target.mkdir(parents=True, exist_ok=False)
    return target


def load_module(name: str, file: Path):
    spec = importlib.util.spec_from_file_location(name, file)
    if spec is None or spec.loader is None:
        raise ImportError(str(file))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def close(actual, expected, tolerance, label: str):
    if actual is None or expected is None:
        if actual is not None or expected is not None:
            raise AssertionError(f"{label}: null/quotable mismatch")
        return
    atol, rtol = map(lambda value: Decimal(str(value)), tolerance)
    left = Decimal.from_float(float(actual))
    right = expected if isinstance(expected, Decimal) else Decimal.from_float(float(expected))
    if not left.is_finite() or not right.is_finite() or abs(left - right) > atol + rtol * abs(right):
        raise AssertionError(f"{label}: actual={actual}, independent={expected}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--admission-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt, hashes = read_admission(args.admission_receipt, args.pilot)
    out = output_directory(args.output_dir)
    started = time.monotonic()
    report = {"schema": "raag-v3-p1-fixture-report-v1", "status": "RUNNING", "mode": receipt["mode"],
              "source_sha256": hashes, "admission_sha256": digest(args.admission_receipt), "rows": [],
              "unit_suite": "NOT RUN", "ideal_scaling": "NOT RUN", "precision_probes": "NOT RUN",
              "optimizer_updates": 0, "historical_remeasurement": "NOT RUN"}

    def budget():
        if time.monotonic() - started > receipt["max_wall_seconds"]:
            raise TimeoutError("admitted wall ceiling exceeded")
        if peak_rss_mib() > receipt["max_peak_rss_mib"]:
            raise MemoryError("admitted peak-RSS ceiling exceeded")

    def timed_out(signum, frame):
        raise TimeoutError("admitted wall ceiling exceeded")

    signal.signal(signal.SIGALRM, timed_out)
    signal.setitimer(signal.ITIMER_REAL, receipt["max_wall_seconds"])
    try:
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[variable] = "1"
        import numpy as np
        import scipy
        sys.path.insert(0, str(ROOT / "scripts"))
        from local_raag_v3 import allocation_diagnostics, build_knn_graph, raag_battery
        oracle = load_module("raag_v3_oracle", ROOT / "validation/raag_v3_oracle.py")
        report["environment"] = {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "threads": 1}
        ids = [("raag-v3-tail-fixture", row) for row in range(32)]
        x = np.asarray([[row / 31] for row in range(32)], dtype=np.float64)
        directed, union = oracle.oracle_graph(x.tolist(), 5, ids)
        graph = build_knn_graph(x, 5, cell_ids=ids)
        if not np.array_equal(graph.directed.toarray(), dense_adjacency(np, directed)) or not np.array_equal(graph.graph.toarray(), dense_adjacency(np, union)):
            raise AssertionError("reference directed/union graph differs from independent oracle")
        report["reference"] = {"coordinates": x.tolist(), "ids": ids, "directed": directed, "union": union, "metadata": graph.metadata}

        with localcontext() as context:
            context.prec = 80
            q = [[Decimal((31-row)*(column+1)+row*(9-column))/Decimal(1395) for column in range(9)] for row in range(32)]
            dq = sum((oracle.decimal_jsd(q[left], q[right]) for left in range(32) for right in range(left+1, 32)), Decimal(0))/Decimal(496)
            primary = [Decimal(10) ** -power for power in range(15)] + [Decimal(0)]
            floor_and_guard = Decimal.from_float(float(1e-12))
            epsilon = sorted(set(primary + [factor*anchor for anchor in (45*floor_and_guard, 5*floor_and_guard, floor_and_guard/dq) for factor in (Decimal("0.5"), Decimal(1), Decimal(2))]), reverse=True)
            donor_map = [(13*row+7) % 32 for row in range(32)]
            controls = {"smooth": q, "shuffled": [q[row] for row in donor_map],
                        "alternating_vertices": [[Decimal(int(column == (0 if row % 2 == 0 else 8))) for column in range(9)] for row in range(32)],
                        "constant": [q[0] for _ in range(32)], "uniform": [[Decimal(1)/Decimal(9)]*9 for _ in range(32)]}
            report["design"] = {"unclipped_D_q": str(dq), "epsilon_decimal": list(map(str, epsilon)), "donor_map": donor_map}
            cases = [(name, eps, tails) for name, tails in controls.items() for eps in epsilon]
            if args.pilot:
                cases = [("smooth", eps, q) for eps in (Decimal(1), Decimal("1e-10"), Decimal(0))]
            else:
                stream = io.StringIO()
                tests = load_module("test_raag_v3", ROOT / "validation/test_raag_v3.py")
                result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
                report["unit_suite"] = {"status": "PASS" if result.wasSuccessful() else "FAIL", "run": result.testsRun,
                                        "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped), "log": stream.getvalue()}
                if not result.wasSuccessful() or result.skipped or not result.testsRun:
                    raise AssertionError("complete non-skipped unit suite required")
                scaling = []
                for eps in primary:
                    ideal = [[Decimal(1)-eps, *[eps*value for value in row]] for row in q]
                    measured = sum((oracle.decimal_jsd(ideal[left], ideal[right]) for left in range(32) for right in range(left+1, 32)), Decimal(0))/Decimal(496)
                    residual = abs(measured-eps*dq)
                    scaling.append({"epsilon": str(eps), "absolute_residual": str(residual)})
                    if residual > Decimal("1e-70") + Decimal("1e-65")*abs(eps*dq):
                        raise AssertionError("ideal unclipped scaling identity differs")
                    budget()
                report["ideal_scaling"] = {"status": "PASS", "rows": scaling}
            for name, eps, tails in cases:
                p = np.asarray([[float(Decimal(1)-eps), *[float(eps*value) for value in row]] for row in tails], dtype=np.float64)
                expected = oracle.oracle_allocation(p.tolist(), union)
                actual = allocation_diagnostics(x, p, reference_ids=ids, allocation_ids=ids, k=5, already_simplex=True, return_probabilities=True)
                row = {"control": name, "epsilon_decimal": str(eps), "epsilon_float64": float(eps), "expected": expected, "actual": actual}
                report["rows"].append(row)
                compare_allocation(actual, expected, close)
                if (name in ("constant", "uniform") or eps == 0) and expected["status"] != "degenerate":
                    raise AssertionError("predeclared constant/zero control is not degenerate")
                row["check_status"] = "PASS"
                budget()
            if not args.pilot:
                eta = Decimal("1e-4")
                near = np.asarray([[float((1-eta)/10 + (eta if column == (0 if row % 2 == 0 else 9) else 0)) for column in range(10)] for row in range(32)])
                expected = oracle.oracle_allocation(near.tolist(), union)
                actual = allocation_diagnostics(x, near, reference_ids=ids, allocation_ids=ids, k=5, already_simplex=True, return_probabilities=True)
                report["rows"].append({"control": "varying_near_uniform", "expected": expected, "actual": actual})
                compare_allocation(actual, expected, close)
                if expected["status"] != "near_uniform":
                    raise AssertionError("predeclared near-uniform control has the wrong status")
                report["rows"][-1]["check_status"] = "PASS"
                report["precision_probes"] = precision_probes(np, oracle, build_knn_graph, raag_battery, budget)
            budget()
        report["status"] = "PASS_PILOT" if args.pilot else "PASS_P1_FIXTURES"
    except Exception as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        report["elapsed_seconds"] = time.monotonic()-started
        report["peak_rss_mib"] = peak_rss_mib()
        report["rss_measurement"] = "Linux /proc/self/status VmHWM" if Path('/proc/self/status').exists() else "resource.getrusage"
        report["rusage_peak_rss_mib_including_possible_launcher_history"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        with (out/"report.json").open("x") as handle:
            json.dump(plain(report), handle, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps({"status": report["status"], "report": str(out/"report.json"), "completed_rows": sum(row.get("check_status") == "PASS" for row in report["rows"])}))
    return 0 if report["status"].startswith("PASS_") else 1


def compare_allocation(actual, expected, compare):
    expected_numeric = expected["decimal_summary"]
    for actual_key, expected_key in (("allocation_status", "status"), ("flooring_entries", "flooring_entries"),
                                     ("allocation_degenerate", "degenerate"), ("allocation_near_uniform", "near_uniform")):
        if actual[actual_key] != expected[expected_key]:
            raise AssertionError(f"{actual_key} differs from independent expectation")
    if expected["status"] != "eligible" and actual["allocation_coherence"] is not None:
        raise AssertionError("ineligible native coherence must be exactly null")
    if expected["degenerate"] and actual["allocation_coherence_raw"] != 0.0:
        raise AssertionError("degenerate raw-zero convention must hold exactly")
    for actual_key, expected_key, tolerance in (
        ("allocation_pair_js", "pair_js", JSD_TOLERANCE), ("allocation_edge_js", "edge_js", JSD_TOLERANCE),
        ("allocation_mean_pair_tv", "mean_pair_tv", TV_TOLERANCE), ("allocation_max_coordinate_range", "max_coordinate_range", TV_TOLERANCE),
        ("allocation_mean_perplexity", "mean_perplexity", PERPLEXITY_TOLERANCE), ("allocation_perplexity_fraction", "perplexity_fraction", PERPLEXITY_TOLERANCE),
        ("allocation_coherence_raw", "raw_coherence", RATIO_TOLERANCE), ("allocation_coherence", "coherence", RATIO_TOLERANCE),
    ):
        compare(actual[actual_key], expected_numeric[expected_key], tolerance, actual_key)
    if not expected["degenerate"]:
        compare(actual["allocation_relative_coherence_unclipped"], expected_numeric["relative_ratio"], RATIO_TOLERANCE, "relative_ratio")
    for actual_row, expected_row in zip(actual["measured_probabilities"], expected["decimal_measured_probabilities"], strict=True):
        for actual_value, expected_value in zip(actual_row, expected_row, strict=True):
            compare(actual_value, expected_value, (2e-16, 2e-15), "measured_probability")


def precision_probes(np, oracle, build_graph, battery, budget):
    ids = [("raag-v3-precision", row) for row in range(4)]
    baseline = [[0., 0.], [1., 0.], [-1., 0.], [0., 2.]]
    rows = []
    first = None
    for dtype in (np.float32, np.float64):
        for delta in (0., 2.**-25, -2.**-25, 2.**-40, -2.**-40):
            values = [row.copy() for row in baseline]
            values[2][0] += delta
            coordinates = np.asarray(values, dtype=dtype)
            graph = build_graph(coordinates, 1, cell_ids=ids)
            directed, union = oracle.oracle_graph(coordinates.tolist(), 1, ids)
            if not np.array_equal(graph.directed.toarray(), dense_adjacency(np, directed)) or not np.array_equal(graph.graph.toarray(), dense_adjacency(np, union)):
                raise AssertionError("precision probe differs from oracle for the delivered cast matrix")
            scores = battery(np.asarray(baseline), coordinates, reference_ids=ids, candidate_ids=ids, k=1, already_simplex=False)
            if first is None:
                first = graph.directed.toarray()
            rows.append({"dtype": str(coordinates.dtype), "delta": delta, "actual_coordinates": coordinates.tolist(),
                         "directed": directed, "union": union, "metadata": graph.metadata, "scores": scores,
                         "directed_entries_changed_from_first": int(np.count_nonzero(graph.directed.toarray() != first))})
            budget()
        coordinates = np.asarray([[0.,0.],[2.**-30,0.],[2.**-29,0.],[1.,0.]], dtype=dtype)
        graph = build_graph(coordinates, 1, cell_ids=ids)
        directed, union = oracle.oracle_graph(coordinates.tolist(), 1, ids)
        if not np.array_equal(graph.directed.toarray(), dense_adjacency(np, directed)) or not np.array_equal(graph.graph.toarray(), dense_adjacency(np, union)):
            raise AssertionError("near-duplicate probe differs from oracle")
        rows.append({"dtype": str(coordinates.dtype), "control": "near_duplicates", "actual_coordinates": coordinates.tolist(),
                     "directed": directed, "union": union, "metadata": graph.metadata})
        budget()
    return {"status": "PASS_CONTRACT_FOR_DELIVERED_COORDINATES", "rows": rows,
            "interpretation": "precision-induced changes are reported, not a robustness guarantee; four-cell graph scores may be guarded"}


def dense_adjacency(np, neighbours):
    matrix = np.zeros((len(neighbours), len(neighbours)), dtype=np.int8)
    for row, selected in enumerate(neighbours):
        matrix[row, list(selected)] = 1
    return matrix


if __name__ == "__main__":
    raise SystemExit(main())
