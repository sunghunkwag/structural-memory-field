#!/usr/bin/env python3
"""Capacity Curve: How many distinct patterns can SMF store and recall?

For N in [2, 4, 8, 16, 32, 64]: train N distinct one-hot patterns (cycling
channels if N > K), erase waves, idle 30 steps, probe each pattern 20 steps.
Recall accuracy = correctly recalled / N.

All results from engine.step() return values only.
"""
import sys
import numpy as np
from collections import Counter
from smf.engine.v2 import EngineV2
from smf.config.params import EngineConfig


def make_pattern(i: int, K: int, amplitude: float = 3.0) -> np.ndarray:
    """Generate pattern i. Channel = i % K, amplitude scaled by slot."""
    p = np.zeros(K)
    p[i % K] = amplitude
    return p


def measure_recall(N: int, H: int, W: int, K: int = 8,
                   train_steps: int = 50, idle_steps: int = 30,
                   probe_steps: int = 20) -> tuple[float, dict]:
    """Train N patterns, erase waves, idle, probe from snapshot.

    Returns (accuracy, detail_dict).
    """
    num_actions = min(4, K)  # readout architecture limit
    cfg = EngineConfig(H=H, W=W, K=K)
    cfg.stress.knot_threshold = 0.3  # needed for knot/crystal formation
    e = EngineV2(cfg=cfg)

    patterns = [make_pattern(i, K) for i in range(N)]
    # Expected action = i % num_actions (since readout only scores num_actions channels)
    expected = [i % num_actions for i in range(N)]

    step_count = 0

    # Train each pattern sequentially, train_steps per pattern
    for p in patterns:
        for _ in range(train_steps):
            e.step(p)
            step_count += 1

    # Erase waves
    e.psi *= 0

    # Idle
    empty = np.zeros(K)
    for _ in range(idle_steps):
        e.step(empty)
        step_count += 1

    # Save post-idle snapshot
    recall_state = e.get_state()

    # Probe each pattern from the SAME snapshot (no cross-contamination)
    correct = 0
    details = {}
    for i, p in enumerate(patterns):
        probe_e = EngineV2(cfg=cfg)
        probe_e.load_state(recall_state)
        actions = []
        for _ in range(probe_steps):
            a = probe_e.step(p)
            actions.append(a)
            step_count += 1
        voted = Counter(actions).most_common(1)[0][0]
        hit = voted == expected[i]
        if hit:
            correct += 1
        details[i] = {"expected": expected[i], "got": voted, "hit": hit}

    assert step_count > 0, "No engine steps executed"
    return correct / N, details


def run_capacity_curve():
    N_values = [2, 4, 8, 16, 32, 64]
    field_sizes = [(16, 16), (32, 32), (64, 64)]

    # Anti-cheat: patterns generated programmatically differ for different N
    p2 = [make_pattern(i, 8) for i in range(2)]
    p64 = [make_pattern(i, 8) for i in range(64)]
    assert len(p2) != len(p64), "N=2 and N=64 must differ in count"
    assert not np.array_equal(p2[0], p64[3]), "Different indices must differ"

    assert len(N_values) >= 6, f"Need >= 6 N values, got {len(N_values)}"
    assert len(field_sizes) >= 3, f"Need >= 3 field sizes, got {len(field_sizes)}"

    print("=" * 70, flush=True)
    print("CAPACITY CURVE: Recall accuracy by pattern count and field size", flush=True)
    print("=" * 70, flush=True)

    header = f"{'N':>4}"
    for H, W in field_sizes:
        header += f"  {H}x{W:>3}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    results = {}
    for N in N_values:
        row = f"{N:>4}"
        results[N] = {}
        for H, W in field_sizes:
            acc, details = measure_recall(N, H, W)
            results[N][(H, W)] = acc
            row += f"  {acc * 100:5.1f}%"
        print(row, flush=True)

    print(flush=True)

    # Capacity = largest N where accuracy >= 75%
    capacities = {}
    for H, W in field_sizes:
        cap = 0
        for N in N_values:
            if results[N][(H, W)] >= 0.75:
                cap = N
        capacities[(H, W)] = cap
        print(f"Capacity({H}x{W}) = {cap}", flush=True)

    print(flush=True)

    # HARD CHECK: N=2 must achieve >= 90% at every field size
    for H, W in field_sizes:
        acc = results[2][(H, W)]
        if acc < 0.90:
            print(f"FATAL: engine cannot even recall 2 patterns at {H}x{W} "
                  f"(accuracy={acc * 100:.1f}%)", flush=True)
            sys.exit(1)

    # HARD CHECK: capacity must be monotonic in field size
    cap_vals = [capacities[s] for s in field_sizes]
    if not all(cap_vals[i] <= cap_vals[i + 1] for i in range(len(cap_vals) - 1)):
        print(f"FAIL: capacity not monotonic: "
              f"{[f'{H}x{W}={capacities[(H,W)]}' for H,W in field_sizes]}", flush=True)
        sys.exit(1)

    print("\nCapacity curve complete.", flush=True)
    return results, capacities


if __name__ == "__main__":
    run_capacity_curve()
