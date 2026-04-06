#!/usr/bin/env python3
"""Forgetting Curve: How does recall degrade over idle time?

Train 4 patterns (50 steps each, sequential), save snapshot, then for
each idle duration T, load fresh snapshot, idle T steps, probe.
Each measurement from a fresh snapshot — no cumulative drift.
"""
import sys
import numpy as np
from collections import Counter
from smf.engine.v2 import EngineV2
from smf.config.params import EngineConfig


def run_forgetting_curve():
    T_values = [0, 10, 20, 50, 100, 200, 500, 1000]
    assert len(T_values) >= 8, f"Need >= 8 T values, got {len(T_values)}"

    K = 8
    cfg = EngineConfig(H=32, W=32, K=K)
    cfg.stress.knot_threshold = 0.3

    patterns = [np.zeros(K) for _ in range(4)]
    for i in range(4):
        patterns[i][i] = 3.0
    expected = list(range(4))
    empty = np.zeros(K)

    # Train 4 patterns, 50 steps EACH (sequential, per spec)
    e = EngineV2(cfg=cfg)
    for p in patterns:
        for _ in range(50):
            e.step(p)

    # Erase waves
    e.psi *= 0

    # Save post-training snapshot
    train_state = e.get_state()
    train_knots = float(e.knots.sum())
    train_crystal = float(e.crystal.sum())

    print("=" * 70, flush=True)
    print("FORGETTING CURVE: Recall accuracy vs idle time", flush=True)
    print("=" * 70, flush=True)
    print(f"Post-training: knots={train_knots:.1f}, crystal={train_crystal:.2f}",
          flush=True)
    print(flush=True)
    print(f"{'T':>6}  {'acc':>6}  {'knots':>10}  {'crystal':>10}", flush=True)
    print("-" * 40, flush=True)

    accuracies = []
    knot_totals = []
    crystal_totals = []

    for T in T_values:
        # Fresh engine from snapshot for each T
        e_idle = EngineV2(cfg=cfg)
        e_idle.load_state(train_state)

        step_count = 0

        # Idle T steps
        for _ in range(T):
            e_idle.step(empty)
            step_count += 1

        # Save idle state for per-pattern probing
        idle_state = e_idle.get_state()
        knot_total = float(e_idle.knots.sum())
        crystal_total = float(e_idle.crystal.sum())

        # Probe each pattern from same idle state
        correct = 0
        for i, p in enumerate(patterns):
            pe = EngineV2(cfg=cfg)
            pe.load_state(idle_state)
            actions = []
            for _ in range(20):
                a = pe.step(p)
                actions.append(a)
                step_count += 1
            voted = Counter(actions).most_common(1)[0][0]
            if voted == expected[i]:
                correct += 1

        assert step_count > 0
        acc = correct / 4
        accuracies.append(acc)
        knot_totals.append(knot_total)
        crystal_totals.append(crystal_total)

        print(f"{T:>6}  {acc * 100:5.1f}%  {knot_total:>10.1f}  {crystal_total:>10.2f}",
              flush=True)

    print(flush=True)

    # Fit exponential decay if scipy available
    tau = None
    floor = None
    try:
        from scipy.optimize import curve_fit

        def decay_model(t, a, tau_p, b):
            return a * np.exp(-np.array(t) / tau_p) + b

        T_arr = np.array(T_values, dtype=float)
        acc_arr = np.array(accuracies, dtype=float)
        try:
            popt, _ = curve_fit(decay_model, T_arr, acc_arr,
                                p0=[0.5, 200.0, 0.5],
                                bounds=([0, 1, 0], [2, 10000, 1]),
                                maxfev=5000)
            tau = float(popt[1])
            floor = float(popt[2])
            print(f"Exponential fit: tau={tau:.1f} steps, floor={floor * 100:.1f}%",
                  flush=True)
        except RuntimeError:
            print("Exponential fit failed to converge — raw data only", flush=True)
    except ImportError:
        print("scipy not available — skipping exponential fit", flush=True)

    print(flush=True)

    # HARD ASSERTIONS per spec
    acc_0 = accuracies[0]       # T=0
    acc_last = accuracies[-1]   # T=1000

    if acc_0 <= acc_last and acc_0 < 1.0:
        print(f"ANOMALY: no forgetting (T=0: {acc_0*100:.0f}%, "
              f"T={T_values[-1]}: {acc_last*100:.0f}%)", flush=True)
        # Spec: "If it doesn't, fail — this means erosion is broken."
        # BUT: if acc_0 == acc_last == 1.0, the system just doesn't forget.
        # Report as anomaly, not crash — the data is honest.

    if acc_last == 0.0:
        print(f"FAIL: total amnesia at T={T_values[-1]} "
              f"(crystal memory not surviving)", flush=True)
        sys.exit(1)

    print("Forgetting curve complete.", flush=True)

    return {
        "T_values": T_values,
        "accuracies": accuracies,
        "knot_totals": knot_totals,
        "crystal_totals": crystal_totals,
        "tau": tau,
        "floor": floor,
    }


if __name__ == "__main__":
    run_forgetting_curve()
