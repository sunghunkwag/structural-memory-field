#!/usr/bin/env python3
"""Benchmark: vectorized readout vs original loop.

Measures timing of compute_action_scores to verify the vectorized
version is faster than the loop version.
"""
import time
import numpy as np
from smf.config.params import ReadoutParams


def loop_readout(V, amp, knots, stress, crystal, num_actions, rp):
    """Original loop version for comparison."""
    scores = []
    for c in range(num_actions):
        val = float(
            ((1 - V[:, :, c])
             * amp[:, :, c]
             * (1 + knots[:, :, c] * rp.knot_amplification)
             * (stress + rp.stress_offset)
             * (1 + crystal[:, :, c] * rp.crystal_amplification)).sum()
        )
        scores.append(val)
    return scores


def vectorized_readout(V, amp, knots, stress, crystal, num_actions, rp):
    """Current vectorized version."""
    from smf.engine.dynamics.readout import compute_action_scores
    return compute_action_scores(V, amp, knots, stress, crystal, num_actions, rp)


def run_readout_benchmark():
    rp = ReadoutParams()
    sizes = [(16, 16, 8), (32, 32, 8), (64, 64, 8), (128, 128, 8)]
    reps = 1000

    print("=" * 60, flush=True)
    print("READOUT BENCHMARK: loop vs vectorized", flush=True)
    print("=" * 60, flush=True)
    print(f"{'Size':>12}  {'Loop (ms)':>10}  {'Vec (ms)':>10}  {'Speedup':>8}", flush=True)
    print("-" * 50, flush=True)

    for H, W, K in sizes:
        V = np.random.rand(H, W, K).astype(np.float32)
        amp = np.random.rand(H, W, K).astype(np.float32)
        knots = np.random.rand(H, W, K).astype(np.float32)
        crystal = np.random.rand(H, W, K).astype(np.float32)
        stress = np.random.rand(H, W).astype(np.float32)

        t0 = time.perf_counter()
        for _ in range(reps):
            loop_readout(V, amp, knots, stress, crystal, 4, rp)
        t_loop = (time.perf_counter() - t0) / reps * 1000

        t0 = time.perf_counter()
        for _ in range(reps):
            vectorized_readout(V, amp, knots, stress, crystal, 4, rp)
        t_vec = (time.perf_counter() - t0) / reps * 1000

        speedup = t_loop / t_vec if t_vec > 0 else float("inf")
        print(f"{H}x{W}x{K:>3}  {t_loop:>10.3f}  {t_vec:>10.3f}  {speedup:>7.1f}x", flush=True)

        # Verify identical results
        s_loop = loop_readout(V, amp, knots, stress, crystal, 4, rp)
        s_vec = vectorized_readout(V, amp, knots, stress, crystal, 4, rp)
        for i in range(4):
            # float32 summation order may differ; use relative tolerance
            ref = max(abs(s_loop[i]), 1e-10)
            assert abs(s_loop[i] - s_vec[i]) / ref < 1e-4, f"Mismatch at ch{i}"

    print("\nReadout benchmark complete.", flush=True)


if __name__ == "__main__":
    run_readout_benchmark()
