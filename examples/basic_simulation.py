"""Basic simulation: train engine, observe memory formation, test recall."""
import numpy as np
from smf.engine.v2 import EngineV2
from smf.config.params import EngineConfig

cfg = EngineConfig()
cfg.stress.knot_threshold = 0.3
e = EngineV2(cfg=cfg)

s_A = np.array([3, 0, 0, 0, 0, 0, 0, 0])
s_B = np.array([0, 0, 3, 0, 0, 0, 0, 0])
empty = np.zeros(8)

print("=== Phase 1: Training with pattern A (50 steps) ===")
for i in range(50):
    a = e.step(s_A)
print(f"  V mean: {e.V.mean():.4f}")
print(f"  Knots total: {e.knots.sum():.2f}")
print(f"  Crystal total: {e.crystal.sum():.2f}")
print(f"  Action: {a}")

print("\n=== Phase 2: Silence (50 steps) ===")
for i in range(50):
    a = e.step(empty)
diag = e.get_diagnostics()
print(f"  V mean: {diag['V_mean']:.4f}")
print(f"  Knots remaining: {diag['knots_total']:.2f}")
print(f"  Crystal remaining: {diag['crystal_total']:.2f}")
print(f"  Wave energy: {diag['field_energy']:.6f}")

print("\n=== Phase 3: Probe with A (does it remember?) ===")
actions_A = [e.step(s_A) for _ in range(20)]
print(f"  Actions: {actions_A}")
print(f"  Most common: {max(set(actions_A), key=actions_A.count)}")

print("\n=== Phase 4: Probe with B (different response?) ===")
actions_B = [e.step(s_B) for _ in range(20)]
print(f"  Actions: {actions_B}")
print(f"  Most common: {max(set(actions_B), key=actions_B.count)}")

print("\n=== Result ===")
if max(set(actions_A), key=actions_A.count) != max(set(actions_B), key=actions_B.count):
    print("SUCCESS: Engine differentiates A from B after silence")
else:
    print("PARTIAL: Actions overlap, but structural scars persist:")
    print(f"  Knots: {e.knots.sum():.2f}, Crystal: {e.crystal.sum():.2f}")
