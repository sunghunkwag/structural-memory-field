#!/usr/bin/env python3
"""Run all benchmarks and save combined results."""
import json
import os
import sys
import traceback
from datetime import datetime, timezone


def run_all():
    # Ensure benchmarks dir is importable
    bench_dir = os.path.dirname(os.path.abspath(__file__))
    if bench_dir not in sys.path:
        sys.path.insert(0, bench_dir)

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmarks": {},
        "status": {},
    }

    benchmarks = [
        ("capacity_curve", "capacity_curve", "run_capacity_curve"),
        ("forgetting_curve", "forgetting_curve", "run_forgetting_curve"),
        ("baselines", "baselines", "run_baselines"),
        ("sensitivity", "sensitivity", "run_sensitivity"),
    ]

    all_pass = True
    for name, module_path, func_name in benchmarks:
        print(f"\n{'#' * 70}", flush=True)
        print(f"# BENCHMARK: {name}", flush=True)
        print(f"{'#' * 70}\n", flush=True)
        try:
            mod = __import__(module_path, fromlist=[func_name])
            fn = getattr(mod, func_name)
            result = fn()
            results["benchmarks"][name] = _make_serializable(result)
            results["status"][name] = "PASS"
        except SystemExit as e:
            results["status"][name] = f"FAIL (exit {e.code})"
            all_pass = False
            print(f"\n[{name}] exited with code {e.code}", flush=True)
        except Exception:
            results["status"][name] = "ERROR"
            all_pass = False
            traceback.print_exc()
            print(f"\n[{name}] raised exception", flush=True)

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "latest.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)

    # Summary
    print(f"\n{'=' * 70}", flush=True)
    print("BENCHMARK SUMMARY", flush=True)
    print(f"{'=' * 70}", flush=True)
    for name, status in results["status"].items():
        icon = "PASS" if status == "PASS" else "FAIL"
        print(f"  [{icon}] {name}: {status}", flush=True)
    print(flush=True)

    if not all_pass:
        print("Some benchmarks failed. See details above.", flush=True)
        sys.exit(1)
    else:
        print("All benchmarks passed.", flush=True)


def _make_serializable(obj):
    """Convert numpy types, tuples, and tuple-keys to JSON-safe types."""
    import numpy as np
    if isinstance(obj, dict):
        return {
            str(k) if isinstance(k, tuple) else _make_serializable(k):
            _make_serializable(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


if __name__ == "__main__":
    run_all()
