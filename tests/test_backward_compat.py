"""Bit-exact parity with original v2/v3 code.

Runs original and refactored side-by-side for 200 steps with varied stimuli.
Asserts np.array_equal (NOT allclose) on ALL fields.
"""
import sys
import os
import numpy as np
import pytest

# Import originals from tests/reference/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "reference"))
from void_resonance_v2 import VoidResonanceV2  # noqa: E402
from void_resonance_v3 import VoidResonanceV3  # noqa: E402

from smf.engine.v2 import EngineV2
from smf.engine.v3 import EngineV3


def _make_stimuli():
    s_A = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    s_B = np.array([0, 0, 1, 0, 0, 0, 0, 0])
    s_AB = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    empty = np.zeros(8)
    big = np.array([3, 0, 0, 0, 0, 0, 0, 0])
    return [s_A] * 50 + [s_B] * 50 + [empty] * 50 + [big] * 30 + [s_AB] * 20


V2_FIELDS = ["V", "psi", "R", "knots", "crystal", "echo", "stress"]
V3_FIELDS = V2_FIELDS + ["crystal_phase", "boundary"]


def test_v2_backward_compat():
    """EngineV2 must produce bit-identical fields to original VoidResonanceV2."""
    stimuli = _make_stimuli()
    old = VoidResonanceV2()
    new = EngineV2()

    for i, s in enumerate(stimuli):
        old.step(s)
        new.step(s)
        for f in V2_FIELDS:
            a = getattr(old, f)
            b = getattr(new, f)
            assert np.array_equal(a, b), (
                f"V2 mismatch at step {i}, field '{f}': "
                f"max_diff={np.max(np.abs(a - b))}"
            )


def test_v3_backward_compat():
    """EngineV3 must produce bit-identical fields to original VoidResonanceV3."""
    stimuli = _make_stimuli()
    old = VoidResonanceV3()
    new = EngineV3()

    for i, s in enumerate(stimuli):
        old.step(s)
        new.step(s)
        for f in V3_FIELDS:
            a = getattr(old, f)
            b = getattr(new, f)
            assert np.array_equal(a, b), (
                f"V3 mismatch at step {i}, field '{f}': "
                f"max_diff={np.max(np.abs(a - b))}"
            )


def test_v2_backward_compat_with_phases():
    """V2 backward compat with explicit phase input."""
    old = VoidResonanceV2()
    new = EngineV2()
    s_amp = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    s_ph = np.array([0.7, -0.7, 0, 0, 0, 0, 0, 0])

    for i in range(100):
        old.step(s_amp, s_ph)
        new.step(s_amp, s_ph)

    for f in V2_FIELDS:
        assert np.array_equal(getattr(old, f), getattr(new, f)), f"V2 phase mismatch: {f}"


def test_v3_backward_compat_with_phases():
    """V3 backward compat with phase conflict stimuli."""
    old = VoidResonanceV3(knot_threshold=0.3, crystal_threshold=0.1, fracture_threshold=0.05)
    new = EngineV3(knot_threshold=0.3, crystal_threshold=0.1, fracture_threshold=0.05)

    for i in range(60):
        ph = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
        s = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        old.step(s, ph)
        new.step(s, ph)

    for i in range(60):
        ph = np.array([-2.0, 0, 0, 0, 0, 0, 0, 0])
        s = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        old.step(s, ph)
        new.step(s, ph)

    for f in V3_FIELDS:
        assert np.array_equal(getattr(old, f), getattr(new, f)), f"V3 phase mismatch: {f}"


def test_v2_backward_compat_varied_params():
    """V2 backward compat with non-default parameters."""
    kw = dict(knot_threshold=0.3, erosion_rate=0.5, echo_coupling=0.1,
              resonance_range=0.8, healing_rate=0.2)
    old = VoidResonanceV2(**kw)
    new = EngineV2(**kw)

    stimuli = _make_stimuli()
    for i, s in enumerate(stimuli):
        old.step(s)
        new.step(s)

    for f in V2_FIELDS:
        assert np.array_equal(getattr(old, f), getattr(new, f)), f"V2 param mismatch: {f}"
