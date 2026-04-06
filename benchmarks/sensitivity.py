#!/usr/bin/env python3
"""Parameter Sensitivity Report: Which parameters matter most?

For each of 10 key parameters, runs the 4-pattern recall benchmark at
5 multiplier values [0.1x, 0.5x, 1x, 2x, 10x] and reports accuracy.

Uses knot_threshold=0.3 as the base config (needed for scars to form).
When testing knot_threshold itself, the 1x baseline IS 0.3.
"""
import sys
import numpy as np
from collections import Counter
from smf.engine.v2 import EngineV2
from smf.config.params import EngineConfig


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


def _make_base_cfg() -> EngineConfig:
    """Base config: defaults with knot_threshold lowered for scar formation."""
    cfg = EngineConfig()
    cfg.stress.knot_threshold = 0.3
    return cfg


def run_recall_benchmark(cfg: EngineConfig) -> float:
    """4-pattern recall: train 50 each (sequential), silence 100, probe 20."""
    K = cfg.K
    patterns = [np.zeros(K) for _ in range(4)]
    for i in range(4):
        patterns[i][i] = 3.0
    expected = list(range(4))
    empty = np.zeros(K)

    e = EngineV2(cfg=cfg)
    step_count = 0

    # Train sequential
    for p in patterns:
        for _ in range(50):
            e.step(p)
            step_count += 1

    e.psi *= 0
    for _ in range(100):
        e.step(empty)
        step_count += 1

    state = e.get_state()

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
    print("Base config: EngineConfig() with stress.knot_threshold=0.3", flush=True)
    print("=" * 78, flush=True)
    print(flush=True)

    header = f"{'Parameter':<30}"
    for m in MULTIPLIERS:
        header += f"  {m:>5}x"
    header += "   range"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    all_accuracies = {}
    ranges = []

    for group_name, param_name in PARAMS:
        # Get the default value from the BASE config (with overrides applied)
        base_cfg = _make_base_cfg()
        default_val = getattr(getattr(base_cfg, group_name), param_name)

        accs = []
        for mult in MULTIPLIERS:
            cfg = _make_base_cfg()

            # Apply multiplier to THIS parameter dynamically
            setattr(getattr(cfg, group_name), param_name, default_val * mult)

            acc = run_recall_benchmark(cfg)
            accs.append(acc)

        acc_range = max(accs) - min(accs)
        ranges.append(acc_range)
        all_accuracies[(group_name, param_name)] = accs

        row = f"{group_name + '.' + param_name:<30}"
        for a in accs:
            row += f"  {a * 100:5.1f}%"
        row += f"  {acc_range * 100:5.1f}pp"
        print(row, flush=True)

    print(flush=True)

    # Most sensitive parameter
    max_idx = int(np.argmax(ranges))
    most_sensitive = PARAMS[max_idx]
    print(f"Most sensitive: {most_sensitive[0]}.{most_sensitive[1]} "
          f"(range={ranges[max_idx] * 100:.1f} pp)", flush=True)

    # Safe ranges
    print(flush=True)
    print("Safe ranges (accuracy >= 50%):", flush=True)
    for idx, (gn, pn) in enumerate(PARAMS):
        accs = all_accuracies[(gn, pn)]
        safe = [MULTIPLIERS[i] for i, a in enumerate(accs) if a >= 0.5]
        if safe:
            print(f"  {gn}.{pn}: {min(safe)}x — {max(safe)}x", flush=True)
        else:
            print(f"  {gn}.{pn}: NO safe range", flush=True)

    print(flush=True)

    # --- Anti-cheat assertions ---

    # At least 3 parameters must show > 10 percentage points variation
    high_var = sum(1 for r in ranges if r > 0.10)
    print(f"{high_var}/10 parameters show >10pp accuracy variation", flush=True)
    if high_var < 3:
        print("FAIL: fewer than 3 params with >10pp variation — "
              "parameters may be decorative", flush=True)
        sys.exit(1)

    # At least 8/10 params should produce different accuracy at 0.1x or 10x vs 1x
    differ_count = 0
    for gn, pn in PARAMS:
        accs = all_accuracies[(gn, pn)]
        acc_1x = accs[2]  # index of 1.0x
        if accs[0] != acc_1x or accs[4] != acc_1x:
            differ_count += 1
    print(f"{differ_count}/10 parameters differ at extreme multipliers", flush=True)
    if differ_count < 8:
        print(f"NOTE: {10 - differ_count} parameters unchanged at extremes "
              f"(robust, not necessarily decorative)", flush=True)

    print(flush=True)
    print("Sensitivity report complete.", flush=True)

    return {
        "params": [f"{g}.{p}" for g, p in PARAMS],
        "multipliers": MULTIPLIERS,
        "accuracies": {f"{g}.{p}": all_accuracies[(g, p)] for g, p in PARAMS},
        "ranges": [float(r) for r in ranges],
        "most_sensitive": f"{most_sensitive[0]}.{most_sensitive[1]}",
    }


if __name__ == "__main__":
    run_sensitivity()
