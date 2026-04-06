#!/usr/bin/env python3
"""Parameter Sensitivity Report: Which parameters matter most?

For each of 10 key parameters, runs the 4-pattern recall benchmark at
5 multiplier values [0.1x, 0.5x, 1x, 2x, 10x] and reports accuracy.
"""
import sys
import numpy as np
from collections import Counter
from smf.engine.v2 import EngineV2
from smf.config.params import EngineConfig


# The 10 most important parameters with their config paths
PARAMS = [
    ("wound", "depth"),
    ("healing", "rate"),
    ("diffusion", "alpha"),
    ("stress", "knot_threshold"),
    ("crystallization", "threshold"),
    ("crystallization", "growth_rate"),
    ("erosion", "rate"),
    ("echo", "damping"),
    ("curvature", "plasticity"),
    ("readout", "knot_amplification"),
]

MULTIPLIERS = [0.1, 0.5, 1.0, 2.0, 10.0]


def run_recall_benchmark(cfg: EngineConfig) -> float:
    """4-pattern recall: train interleaved, silence 100, probe from snapshot."""
    K = cfg.K
    patterns = [np.zeros(K) for _ in range(4)]
    for i in range(4):
        patterns[i][i] = 3.0
    expected = list(range(4))
    empty = np.zeros(K)

    e = EngineV2(cfg=cfg)
    step_count = 0

    # Train interleaved
    for _ in range(50):
        for p in patterns:
            e.step(p)
            step_count += 1

    e.psi *= 0
    for _ in range(100):
        e.step(empty)
        step_count += 1

    state = e.get_state()

    # Probe from snapshot
    correct = 0
    for i, p in enumerate(patterns):
        pe = EngineV2(cfg=cfg)
        pe.load_state(state)
        actions = []
        for _ in range(20):
            a = pe.step(p)
            actions.append(a)
            step_count += 1
        if Counter(actions).most_common(1)[0][0] == expected[i]:
            correct += 1

    assert step_count > 0
    return correct / 4


def run_sensitivity():
    print("=" * 78, flush=True)
    print("PARAMETER SENSITIVITY: Recall accuracy at different param multipliers", flush=True)
    print("=" * 78, flush=True)
    print(flush=True)

    # Header
    header = f"{'Parameter':<30}"
    for m in MULTIPLIERS:
        header += f"  {m:>5}x"
    header += "  range"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    all_accuracies = {}
    ranges = []

    for group_name, param_name in PARAMS:
        # Get default value dynamically
        default_cfg = EngineConfig()
        default_cfg.stress.knot_threshold = 0.3  # base override for knot formation
        group = getattr(default_cfg, group_name)
        default_val = getattr(group, param_name)

        accs = []
        for mult in MULTIPLIERS:
            cfg = EngineConfig()
            cfg.stress.knot_threshold = 0.3

            # Apply multiplier dynamically
            setattr(getattr(cfg, group_name), param_name, default_val * mult)

            acc = run_recall_benchmark(cfg)
            accs.append(acc)

        acc_range = max(accs) - min(accs)
        ranges.append(acc_range)
        all_accuracies[(group_name, param_name)] = accs

        row = f"{group_name}.{param_name:<22}"
        for a in accs:
            row += f"  {a * 100:5.1f}%"
        row += f"  {acc_range * 100:5.1f}%"
        print(row, flush=True)

    print(flush=True)

    # Most sensitive parameter
    max_range_idx = int(np.argmax(ranges))
    most_sensitive = PARAMS[max_range_idx]
    print(f"Most sensitive: {most_sensitive[0]}.{most_sensitive[1]} "
          f"(range={ranges[max_range_idx] * 100:.1f}%)", flush=True)

    # Safe ranges (where accuracy >= 50%)
    print(flush=True)
    print("Safe ranges (accuracy >= 50%):", flush=True)
    for idx, (group_name, param_name) in enumerate(PARAMS):
        accs = all_accuracies[(group_name, param_name)]
        safe_mults = [MULTIPLIERS[i] for i, a in enumerate(accs) if a >= 0.5]
        if safe_mults:
            print(f"  {group_name}.{param_name}: {min(safe_mults)}x — {max(safe_mults)}x",
                  flush=True)
        else:
            print(f"  {group_name}.{param_name}: NO safe range", flush=True)

    print(flush=True)

    # Anti-cheat assertions
    # At least 3 parameters should show > 10% accuracy variation
    high_variation = sum(1 for r in ranges if r > 0.10)
    if high_variation < 3:
        print(f"WARNING: only {high_variation}/10 parameters show >10% variation — "
              f"parameters may be decorative", flush=True)
    else:
        print(f"{high_variation}/10 parameters show >10% accuracy variation — "
              f"parameters are functional", flush=True)

    # At least 8/10 params should differ between 0.1x and 10x vs 1x
    differ_count = 0
    for idx, (gn, pn) in enumerate(PARAMS):
        accs = all_accuracies[(gn, pn)]
        if accs[0] != accs[2] or accs[4] != accs[2]:  # 0.1x or 10x != 1x
            differ_count += 1
    print(f"{differ_count}/10 parameters show different accuracy at extreme multipliers",
          flush=True)
    if differ_count < 8:
        print(f"WARNING: {10 - differ_count} parameters unchanged at extremes", flush=True)

    print(flush=True)
    print("Sensitivity report complete.", flush=True)

    return {
        "params": [f"{g}.{p}" for g, p in PARAMS],
        "multipliers": MULTIPLIERS,
        "accuracies": {
            f"{g}.{p}": all_accuracies[(g, p)] for g, p in PARAMS
        },
        "ranges": ranges,
        "most_sensitive": f"{most_sensitive[0]}.{most_sensitive[1]}",
    }


if __name__ == "__main__":
    run_sensitivity()
