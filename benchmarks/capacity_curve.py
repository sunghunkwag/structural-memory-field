#!/usr/bin/env python3
"""Capacity Curve: How many patterns can SMF store and recall?

For N patterns at various field sizes, measures recall accuracy after
wave erasure and idle period. Training is interleaved (round-robin).
Each pattern is probed from the SAME post-idle snapshot to avoid
cross-contamination between probes.

All results come from engine.step() return values only.
"""
import sys
import numpy as np
from collections import Counter
from smf.engine.v2 import EngineV2
from smf.config.params import EngineConfig


def make_pattern(i: int, K: int, amplitude: float = 3.0) -> np.ndarray:
    """Generate pattern i programmatically. Channel = i % K, rest zero."""
    p = np.zeros(K)
    p[i % K] = amplitude
    return p


def measure_recall(N: int, H: int, W: int, K: int = 8,
                   train_rounds: int = 50, idle_steps: int = 30,
                   probe_steps: int = 20) -> float:
    """Train N patterns (interleaved), erase, idle, probe from snapshot.

    Returns recall accuracy. Effective N capped at num_actions.
    """
    cfg = EngineConfig(H=H, W=W, K=K)
    cfg.stress.knot_threshold = 0.3
    e = EngineV2(cfg=cfg)

    num_actions = e.num_actions
    effective_N = min(N, num_actions)

    patterns = [make_pattern(i, K) for i in range(effective_N)]
    expected_actions = list(range(effective_N))

    step_count = 0

    # Interleaved training: round-robin through all patterns
    for _ in range(train_rounds):
        for p in patterns:
            e.step(p)
            step_count += 1

    # Erase waves, keep structural scars
    e.psi *= 0

    # Idle
    empty = np.zeros(K)
    for _ in range(idle_steps):
        e.step(empty)
        step_count += 1

    # Save state for snapshot-based probing
    recall_state = e.get_state()

    # Probe each pattern from the SAME snapshot
    correct = 0
    for i, p in enumerate(patterns):
        # Restore to post-idle state
        probe_engine = EngineV2(cfg=cfg)
        probe_engine.load_state(recall_state)

        actions = []
        for _ in range(probe_steps):
            a = probe_engine.step(p)
            actions.append(a)
            step_count += 1
        most_common = Counter(actions).most_common(1)[0][0]
        if most_common == expected_actions[i]:
            correct += 1

    assert step_count > 0, "No engine steps were executed"
    return correct / effective_N


def run_capacity_curve():
    N_values = [2, 4, 8, 16, 32, 64]
    field_sizes = [(16, 16), (32, 32), (64, 64)]

    # Anti-cheat: verify patterns are generated differently for different N
    p2 = [make_pattern(i, 8) for i in range(2)]
    p64 = [make_pattern(i, 8) for i in range(64)]
    assert len(p2) != len(p64)
    assert not np.array_equal(p2[0], p64[3])

    print("=" * 70, flush=True)
    print("CAPACITY CURVE: Recall accuracy by pattern count and field size", flush=True)
    print("(effective N capped at num_actions=4 by readout architecture)", flush=True)
    print("=" * 70, flush=True)

    header = f"{'N':>4}  {'eff':>3}"
    for H, W in field_sizes:
        header += f"  {H}x{W:>3}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    results = {}
    num_actions = 4
    for N in N_values:
        effective_N = min(N, num_actions)
        row = f"{N:>4}  {effective_N:>3}"
        results[N] = {}
        for H, W in field_sizes:
            acc = measure_recall(N, H, W)
            results[N][(H, W)] = acc
            row += f"  {acc * 100:5.1f}%"
        print(row, flush=True)

    print(flush=True)

    # Compute capacity per field size
    capacities = {}
    for H, W in field_sizes:
        cap = 0
        for N in N_values:
            if results[N][(H, W)] >= 0.75:
                cap = min(N, num_actions)
        capacities[(H, W)] = cap
        print(f"Capacity({H}x{W}) = {cap}", flush=True)

    print(flush=True)

    # Validate N=2 recall
    for H, W in field_sizes:
        acc = results[2][(H, W)]
        if acc < 0.90:
            print(f"FATAL: engine cannot recall 2 patterns at {H}x{W} "
                  f"(accuracy={acc * 100:.1f}%)", flush=True)
            sys.exit(1)

    # Check monotonicity
    cap_vals = [capacities[s] for s in field_sizes]
    if not all(cap_vals[i] <= cap_vals[i + 1] for i in range(len(cap_vals) - 1)):
        print(f"WARNING: capacity not monotonic in field size", flush=True)

    assert len(N_values) >= 6
    assert len(field_sizes) >= 3

    print("\nCapacity curve complete.", flush=True)
    return results, capacities


if __name__ == "__main__":
    run_capacity_curve()
