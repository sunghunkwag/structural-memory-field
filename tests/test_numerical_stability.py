"""Numerical stability tests.

Verifies the engine handles extreme inputs and adversarial configs
without crashing, producing NaN, or Inf.
"""
import numpy as np
import pytest
import warnings
from smf.engine.v2 import EngineV2
from smf.config.params import EngineConfig
from smf.io.encoders import TextEncoder


class TestExtremeInputs:
    """Feed extreme stimulus values — engine must not crash."""

    def test_very_large_amplitude(self):
        """Large stimulus should not produce NaN."""
        e = EngineV2()
        for _ in range(20):
            a = e.step(np.array([1e6, 0, 0, 0, 0, 0, 0, 0]))
        assert not np.any(np.isnan(e.V))
        assert isinstance(a, int)

    def test_very_small_amplitude(self):
        """Tiny stimulus should work normally."""
        e = EngineV2()
        for _ in range(20):
            a = e.step(np.array([1e-15, 0, 0, 0, 0, 0, 0, 0]))
        assert not np.any(np.isnan(e.V))

    def test_negative_amplitude(self):
        """Negative stimulus should not crash (just wounds V below 0, clamped)."""
        e = EngineV2()
        a = e.step(np.array([-5, 0, 0, 0, 0, 0, 0, 0]))
        assert isinstance(a, int)
        assert not np.any(np.isnan(e.V))

    def test_nan_input_rejected(self):
        """NaN in stimulus must raise ValueError."""
        e = EngineV2()
        with pytest.raises(ValueError, match="NaN"):
            e.step(np.array([np.nan, 0, 0, 0, 0, 0, 0, 0]))

    def test_inf_input_rejected(self):
        """Inf in stimulus must raise ValueError."""
        e = EngineV2()
        with pytest.raises(ValueError, match="inf"):
            e.step(np.array([np.inf, 0, 0, 0, 0, 0, 0, 0]))


class TestLongRunStability:
    """Run for many steps — no NaN/Inf should accumulate."""

    def test_10000_steps_no_nan(self):
        """10000 steps with varied stimulus must not produce NaN."""
        cfg = EngineConfig()
        cfg.stress.knot_threshold = 0.3
        e = EngineV2(cfg=cfg)
        s_A = np.array([2, 0, 0, 0, 0, 0, 0, 0])
        s_B = np.array([0, 0, 2, 0, 0, 0, 0, 0])
        empty = np.zeros(8)
        for i in range(10000):
            if i % 3 == 0:
                e.step(s_A)
            elif i % 3 == 1:
                e.step(s_B)
            else:
                e.step(empty)
        assert not np.any(np.isnan(e.V)), "NaN in V after 10000 steps"
        assert not np.any(np.isinf(e.V)), "Inf in V after 10000 steps"
        assert not np.any(np.isnan(e.psi)), "NaN in psi after 10000 steps"


class TestAdversarialConfigs:
    """Configs designed to stress numerical limits."""

    def test_high_wound_high_diffusion_survives(self):
        """Previously caused overflow at step ~214. Now should survive
        (diffusion NaN guard catches overflow)."""
        cfg = EngineConfig()
        cfg.wound.depth = 5.0
        cfg.diffusion.alpha = 0.99
        cfg.diffusion.active_base_decay = 0.99
        cfg.healing.rate = 0.001
        e = EngineV2(cfg=cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(500):
                e.step(np.array([5, 0, 0, 0, 0, 0, 0, 0]))
        # Must not have NaN — diffusion fallback should have caught it
        assert not np.any(np.isnan(e.V)), "NaN in V with adversarial config"

    def test_zero_erosion_no_crash(self):
        """Zero erosion rate should not cause issues."""
        cfg = EngineConfig()
        cfg.erosion.rate = 0.0
        cfg.erosion.idle_rate = 0.0
        e = EngineV2(cfg=cfg)
        for _ in range(100):
            e.step(np.array([1, 0, 0, 0, 0, 0, 0, 0]))
        assert not np.any(np.isnan(e.knots))

    def test_minimal_field_size(self):
        """H=2, W=2, K=1 — degenerate but must not crash."""
        cfg = EngineConfig(H=2, W=2, K=1)
        e = EngineV2(cfg=cfg)
        for _ in range(50):
            a = e.step(np.array([1.0]))
        assert isinstance(a, int)

    def test_debug_validate_interval(self):
        """Field validation runs at configured interval."""
        cfg = EngineConfig()
        cfg.debug_validate_interval = 5
        e = EngineV2(cfg=cfg)
        for _ in range(20):
            e.step(np.zeros(8))
        assert e.t == 20  # completed without validation errors


class TestReadoutEdgeCases:
    """Readout must handle degenerate score cases."""

    def test_zero_stimulus_scores(self):
        """Zero input → all scores zero → action 0 (not crash)."""
        e = EngineV2()
        a = e.step(np.zeros(8))
        assert a == 0


class TestTextEncoderEdgeCases:
    """TextEncoder handles unicode edge cases."""

    def test_whitespace_only(self):
        enc = TextEncoder(K=8)
        a, p = enc.encode("   ")
        assert np.allclose(a, 0)

    def test_very_long_string(self):
        enc = TextEncoder(K=8)
        a, p = enc.encode("x" * 20000)  # truncated to 10000 internally
        assert not np.any(np.isnan(a))
        assert np.sum(np.abs(a) > 0.01) >= 1

    def test_non_bmp_characters(self):
        enc = TextEncoder(K=8)
        a, p = enc.encode("hello 🌍 world 🎉")
        assert not np.any(np.isnan(a))
        assert np.sum(np.abs(a) > 0.01) >= 1

    def test_empty_string(self):
        enc = TextEncoder(K=8)
        a, p = enc.encode("")
        assert np.allclose(a, 0)
        assert np.allclose(p, 0)
