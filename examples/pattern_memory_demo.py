"""Pattern memory demo: train 3 patterns, test discrimination after erasure."""
import numpy as np
from collections import Counter
from smf.engine.v2 import EngineV2
from smf.config.params import EngineConfig

cfg = EngineConfig()
cfg.stress.knot_threshold = 0.3

patterns = {
    "P0": np.array([3, 0, 0, 0, 0, 0, 0, 0]),
    "P1": np.array([0, 3, 0, 0, 0, 0, 0, 0]),
    "P2": np.array([0, 0, 3, 0, 0, 0, 0, 0]),
}
empty = np.zeros(8)

print("=== Training 3 patterns sequentially ===")
e = EngineV2(cfg=cfg)
for name, p in patterns.items():
    for _ in range(40):
        e.step(p)
    print(f"  Trained {name}: knots={e.knots.sum():.1f}, crystal={e.crystal.sum():.2f}")

print("\n=== Erasing wave field (keeping structural scars) ===")
e.psi *= 0
for _ in range(30):
    e.step(empty)
print(f"  After erasure: knots={e.knots.sum():.1f}, crystal={e.crystal.sum():.2f}")

print("\n=== Testing pattern recall ===")
results = {}
for name, p in patterns.items():
    actions = [e.step(p) for _ in range(20)]
    most_common = Counter(actions).most_common(1)[0][0]
    results[name] = most_common
    print(f"  {name} → action {most_common}  (distribution: {dict(Counter(actions))})")

unique_actions = len(set(results.values()))
print(f"\n=== Result: {unique_actions}/3 unique actions ===")
if unique_actions >= 2:
    print("SUCCESS: Engine discriminates patterns through structural memory")
else:
    print("PARTIAL: Structural memory exists but readout doesn't discriminate")
