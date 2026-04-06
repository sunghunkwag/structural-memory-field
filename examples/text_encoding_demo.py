"""Text encoding demo: encode characters → observe engine differentiating them."""
import numpy as np
from smf.engine.v2 import EngineV2
from smf.config.params import EngineConfig
from smf.io.encoders import encode_text_char

cfg = EngineConfig()
cfg.stress.knot_threshold = 0.3
e = EngineV2(cfg=cfg)

text = "ABCABC"
print(f"Feeding text: '{text}'")
print()

for i, ch in enumerate(text):
    amp = encode_text_char(ch, K=8) * 3  # amplify
    action = e.step(amp)
    diag = e.get_diagnostics()
    print(f"  Step {i}: char='{ch}' → action={action}  "
          f"(V={diag['V_mean']:.3f}, knots={diag['knots_total']:.1f}, "
          f"energy={diag['field_energy']:.4f})")

print(f"\nFinal field state after '{text}':")
print(f"  V mean: {e.V.mean():.4f}")
print(f"  Knots: {e.knots.sum():.2f}")
print(f"  Crystal: {e.crystal.sum():.2f}")
print(f"  Stress max: {e.stress.max():.2f}")
