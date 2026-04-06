"""Text encoding demo: train on words, erase waves, test structural recall.

Uses the TextEncoder to create rich multi-channel stimulus vectors.
Trains on 'cat', then erases wave field but keeps structural scars.
Probes with both 'cat' and 'dog' to test if scars guide recall.
"""
import sys
import numpy as np
from smf.engine.v2 import EngineV2
from smf.config.params import EngineConfig
from smf.io.encoders import TextEncoder

cfg = EngineConfig()
cfg.stress.knot_threshold = 0.3
enc = TextEncoder(K=cfg.K)

TRAIN_STEPS = 50
IDLE_STEPS = 30
PROBE_STEPS = 15

# --- Train exclusively on "cat" ---
e = EngineV2(cfg=cfg)
amp_cat, ph_cat = enc.encode("cat")
amp_dog, ph_dog = enc.encode("dog")

print("=== Training on 'cat' ===")
for _ in range(TRAIN_STEPS):
    e.step(amp_cat * 3, ph_cat)
print(f"  knots={e.knots.sum():.1f}, crystal={e.crystal.sum():.2f}")

# --- Erase waves, idle ---
print(f"\n=== Erasing waves, idling {IDLE_STEPS} steps ===")
snap_after_train = e.f.snapshot()
e.psi *= 0
for _ in range(IDLE_STEPS):
    e.step(np.zeros(cfg.K))
snap_after_idle = e.f.snapshot()
print(f"  knots={e.knots.sum():.1f}, crystal={e.crystal.sum():.2f}")
print(f"  Structural scars persist: crystal={e.crystal.sum():.2f}")

# --- Probe with "cat" (trained) ---
e_cat = EngineV2(cfg=cfg)
e_cat.f.load_dict(snap_after_idle.to_dict())
print(f"\n=== Probing with 'cat' ({PROBE_STEPS} steps) ===")
for _ in range(PROBE_STEPS):
    e_cat.step(amp_cat * 3, ph_cat)
cat_energy = float(np.abs(e_cat.psi).sum())
print(f"  Wave energy after 'cat' probe: {cat_energy:.2f}")

# --- Probe with "dog" (untrained) from same snapshot ---
e_dog = EngineV2(cfg=cfg)
e_dog.f.load_dict(snap_after_idle.to_dict())
print(f"\n=== Probing with 'dog' ({PROBE_STEPS} steps) ===")
for _ in range(PROBE_STEPS):
    e_dog.step(amp_dog * 3, ph_dog)
dog_energy = float(np.abs(e_dog.psi).sum())
print(f"  Wave energy after 'dog' probe: {dog_energy:.2f}")

# --- Compare structural distances ---
dist_cat = 0.0
dist_dog = 0.0
for fname in ("V", "knots", "crystal"):
    ref = getattr(snap_after_train, fname)
    dist_cat += float(np.sum((getattr(e_cat.f, fname) - ref) ** 2))
    dist_dog += float(np.sum((getattr(e_dog.f, fname) - ref) ** 2))
dist_cat = np.sqrt(dist_cat)
dist_dog = np.sqrt(dist_dog)

print(f"\n  Distance to training snapshot:")
print(f"    'cat' probe: {dist_cat:.2f}")
print(f"    'dog' probe: {dist_dog:.2f}")

closer_to_trained = dist_cat < dist_dog
recall_pct = 100 * dist_dog / (dist_cat + dist_dog) if (dist_cat + dist_dog) > 0 else 50
print(f"\n=== Recall accuracy: {recall_pct:.1f}% ===")
if closer_to_trained:
    print("SUCCESS: 'cat' probe is closer to training snapshot (structural recall works)")
else:
    print("FAIL: 'dog' probe is closer (structural recall failed)")
    sys.exit(1)
