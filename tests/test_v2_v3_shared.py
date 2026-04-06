"""No-duplication proof: V3 with phase features disabled == V2.

Run V2 and V3 (with v3 phase features disabled) for 100 steps.
Assert np.array_equal on all shared fields including psi and crystal.
"""
import numpy as np
from smf.engine.v2 import EngineV2
from smf.engine.v3 import EngineV3
from smf.config.params import EngineConfig


def test_v3_matches_v2_with_phase_disabled():
    """V3 with phase dynamics disabled must match V2 on ALL fields."""
    cfg_v2 = EngineConfig()
    cfg_v2.stress.knot_threshold = 0.3

    cfg_v3 = EngineConfig()
    cfg_v3.stress.knot_threshold = 0.3
    cfg_v3.phase.fracture_threshold = 1000.0   # never fractures
    cfg_v3.phase.fusion_rate = 0.0             # no fusion
    cfg_v3.phase.nucleation_rate = 0.0         # no nucleation

    e2 = EngineV2(cfg=cfg_v2)
    e3 = EngineV3(cfg=cfg_v3)

    s_A = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    s_B = np.array([0, 0, 1, 0, 0, 0, 0, 0])
    empty = np.zeros(8)
    stimuli = [s_A] * 40 + [s_B] * 30 + [empty] * 30

    for i, s in enumerate(stimuli):
        a2 = e2.step(s)
        a3 = e3.step(s)

        for f in ("V", "psi", "R", "knots", "crystal", "echo", "stress"):
            arr_v2 = getattr(e2, f)
            arr_v3 = getattr(e3, f)
            assert np.array_equal(arr_v2, arr_v3), (
                f"V2/V3 mismatch at step {i}, field '{f}': "
                f"max_diff={np.max(np.abs(arr_v2 - arr_v3))}"
            )

        assert a2 == a3, f"Action mismatch at step {i}: v2={a2}, v3={a3}"
