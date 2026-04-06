# Structural Memory Field (SMF)

Physics-based memory architecture where persistence, forgetting, and behavioral
differentiation emerge from **structural scars** in a continuous field — not
learned parameters, neural networks, or symbolic lookup tables.

Memory lives in topological defects (knots), crystallized regions, and curvature
gradients that form when stimuli wound the void field and leave permanent marks.

## Installation

```bash
pip install -e .

# With test dependencies
pip install -e ".[test]"

# With visualization
pip install -e ".[viz]"
```

## Quick Start

```python
import numpy as np
from smf import EngineV2, EngineConfig

cfg = EngineConfig()
cfg.stress.knot_threshold = 0.3
engine = EngineV2(cfg=cfg)

# Feed stimulus (K=8 channels)
stimulus = np.array([3, 0, 0, 0, 0, 0, 0, 0])
for _ in range(50):
    action = engine.step(stimulus)

print(f"Action: {action}, Knots: {engine.knots.sum():.1f}")
```

## Architecture

```
EngineConfig (all ~60 named parameters)
    |
    v
EngineV2 / EngineV3   (orchestrator — zero magic numbers in step())
    |
    +-- dynamics/wound.py          apply_wound(V, stimulus, bk, params)
    +-- dynamics/injection.py      apply_injection(psi, V, R, crystal, ev, bk, params)
    +-- dynamics/healing.py        apply_healing(V, knots, crystal, bk, params)
    +-- dynamics/echo.py           apply_echo(V, V_before, echo, bk, params)
    +-- dynamics/diffusion.py      apply_diffusion(psi, V, R, knots, crystal, idle, bk, params)
    +-- dynamics/stress.py         apply_stress_and_knots(V, amp, knots, H, W, bk, params)
    +-- dynamics/crystallization.py apply_crystallization(knots, crystal, bk, params)
    +-- dynamics/erosion.py        apply_erosion(V, knots, idle, bk, params)
    +-- dynamics/curvature.py      apply_curvature(R, stress, amp, bk, params)
    +-- dynamics/readout.py        compute_action_scores(V, amp, knots, stress, crystal, ...)
    +-- dynamics/phase.py          [V3 only] fracture, fusion, nucleation
    |
    v
NumpyBackend           (tensor ops abstraction — all np.* calls here)
    |
    v
FieldState             (psi, V, R, knots, crystal, echo, stress)
```

Each dynamics module is a **standalone function** — no engine imports, callable
independently in tests. V3 inherits V2 with zero duplicated physics.

## Parameter Reference

Every numeric constant is a named parameter in `EngineConfig`:

| Group | Parameters | Description |
|-------|-----------|-------------|
| `wound` | depth, layers, phantom_threshold, resonance_range | Wound injection strength and cascade |
| `idle` | threshold | Input level below which engine is "idle" |
| `injection` | layers, curvature_recall_offset/max/scale, crystal_block | Wave injection into psi field |
| `healing` | rate, recovery_strength, knot/crystal_resistance | Void gravity toward V=1 |
| `echo` | damping, coupling | Post-healing oscillation feedback |
| `diffusion` | alpha, crystal_block_strength, knot_resistance, scatter_*, decay_* | Geodesic holographic diffusion |
| `stress` | gradient_amp_weight, diffusion_strength, stress_max, knot_* | Stress and knot formation |
| `crystallization` | threshold, growth_rate, decay_rate, forming_threshold | Crystal nucleation from knots |
| `erosion` | rate, eligibility_threshold, idle_rate | Competitive forgetting |
| `curvature` | plasticity, relaxation_rate, R_min, R_max | Ricci curvature update |
| `readout` | num_actions, knot/crystal_amplification, stress_offset | Action score computation |
| `backend` | gradient_epsilon, laplacian_center_weight | Numerical constants |
| `phase` | neighbor_count, fracture_*, fusion_*, nucleation_* | V3 crystal phase dynamics |

## Examples

| Script | What it demonstrates |
|--------|---------------------|
| `basic_simulation.py` | Train, silence, probe — shows memory formation |
| `text_encoding_demo.py` | TextEncoder + structural recall after wave erasure |
| `pattern_memory_demo.py` | 4-pattern discrimination through structural scars |
| `comparison_baseline.py` | SMF vs random/most-frequent baselines |

Run any example:
```bash
python examples/basic_simulation.py
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Key test suites:
#   test_mutation.py           — 59 params, each provably affects output
#   test_backward_compat.py    — bit-exact match with original v2/v3
#   test_v2_v3_shared.py       — proves v3 shares v2 physics (no duplication)
#   test_dynamics/              — standalone module tests
#   test_encoder_quality.py     — encoder produces real, meaningful vectors
#   test_dynamics_independence.py — structural: no engine imports in dynamics
```

## Benchmarks

Empirical evidence that the physics produces measurable, reproducible behavior.

```bash
# Run all benchmarks
python benchmarks/run_all.py

# Run individually
python benchmarks/capacity_curve.py
python benchmarks/forgetting_curve.py
python benchmarks/baselines.py
python benchmarks/sensitivity.py
```

| Benchmark | What it measures | Key finding |
|-----------|-----------------|-------------|
| **Capacity curve** | Patterns stored vs field size | Capacity = 4 (capped by readout.num_actions) |
| **Forgetting curve** | Recall vs idle time (0-1000 steps) | Crystal memory prevents forgetting (100% at T=1000) |
| **Baselines** | SMF vs LUT, ESN, exponential decay | All methods 100% on 4-pattern task |
| **Sensitivity** | Which parameters break recall | wound.depth, diffusion.alpha, echo.damping are critical |

Results are saved to `benchmarks/results/latest.json` after each run.

## Project Structure

```
src/smf/
  config/params.py      All ~60 parameters as dataclasses
  core/backend.py        NumpyBackend tensor abstraction
  core/fields.py         FieldState container
  core/operators.py      Physics operators (laplacian, gradient, diffusion)
  engine/v2.py           V2 engine orchestrator
  engine/v3.py           V3 engine (inherits V2 + phase dynamics)
  engine/dynamics/       10 standalone physics modules
  io/encoders.py         TextEncoder, encode_onehot
  io/serialization.py    State save/load to .npz
  viz/field_plots.py     Field visualization (matplotlib)
tests/
  reference/             Original v2/v3 (never modified)
  test_mutation.py       THE anti-cheat test
  test_backward_compat.py  Bit-exact parity
  ...
examples/                Runnable demos
```
