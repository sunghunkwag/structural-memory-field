"""Tests for EngineConfig validation.

Verifies that invalid configurations raise ValueError with descriptive messages.
Tests edge-case valid configs. Tests from_dict/from_legacy_kwargs robustness.
"""
import pytest
from smf.config.params import (
    EngineConfig, WoundParams, IdleParams, InjectionParams, HealingParams,
    EchoParams, DiffusionParams, StressParams, CrystallizationParams,
    ErosionParams, CurvatureParams, ReadoutParams, BackendParams, PhaseParams,
)


# ---- Invalid configs that must raise ValueError ----

class TestInvalidConfigs:
    """At least 15 distinct invalid configurations must be caught."""

    def test_H_zero(self):
        with pytest.raises(ValueError, match="H"):
            EngineConfig(H=0)

    def test_W_negative(self):
        with pytest.raises(ValueError, match="W"):
            EngineConfig(W=-1)

    def test_K_zero(self):
        with pytest.raises(ValueError, match="K"):
            EngineConfig(K=0)

    def test_wound_depth_negative(self):
        with pytest.raises(ValueError, match="wound.depth"):
            WoundParams(depth=-0.1)

    def test_wound_layers_zero(self):
        with pytest.raises(ValueError, match="wound.layers"):
            WoundParams(layers=0)

    def test_idle_threshold_zero(self):
        with pytest.raises(ValueError, match="idle.threshold"):
            IdleParams(threshold=0)

    def test_injection_crystal_block_above_one(self):
        with pytest.raises(ValueError, match="injection.crystal_block"):
            InjectionParams(crystal_block=1.5)

    def test_healing_rate_zero(self):
        with pytest.raises(ValueError, match="healing.rate"):
            HealingParams(rate=0)

    def test_echo_damping_above_one(self):
        with pytest.raises(ValueError, match="echo.damping"):
            EchoParams(damping=1.1)

    def test_echo_damping_negative(self):
        with pytest.raises(ValueError, match="echo.damping"):
            EchoParams(damping=-0.1)

    def test_diffusion_alpha_above_one(self):
        with pytest.raises(ValueError, match="diffusion.alpha"):
            DiffusionParams(alpha=1.5)

    def test_diffusion_active_base_decay_above_one(self):
        with pytest.raises(ValueError, match="diffusion.active_base_decay"):
            DiffusionParams(active_base_decay=2.0)

    def test_stress_knot_max_zero(self):
        with pytest.raises(ValueError, match="stress.knot_max"):
            StressParams(knot_max=0)

    def test_crystallization_decay_above_one(self):
        with pytest.raises(ValueError, match="crystallization.decay_rate"):
            CrystallizationParams(decay_rate=1.5)

    def test_curvature_R_max_less_than_R_min(self):
        with pytest.raises(ValueError, match="R_max"):
            CurvatureParams(R_min=5.0, R_max=2.0)

    def test_readout_num_actions_zero(self):
        with pytest.raises(ValueError, match="readout.num_actions"):
            ReadoutParams(num_actions=0)

    def test_backend_laplacian_positive(self):
        with pytest.raises(ValueError, match="laplacian_center_weight"):
            BackendParams(laplacian_center_weight=4.0)

    def test_phase_fracture_threshold_zero(self):
        with pytest.raises(ValueError, match="phase.fracture_threshold"):
            PhaseParams(fracture_threshold=0)

    def test_phase_neighbor_count_negative(self):
        with pytest.raises(ValueError, match="phase.neighbor_count"):
            PhaseParams(neighbor_count=-1)

    def test_erosion_rate_negative(self):
        with pytest.raises(ValueError, match="erosion.rate"):
            ErosionParams(rate=-0.5)


# ---- Valid edge cases ----

class TestValidEdgeCases:
    def test_K_equals_one(self):
        cfg = EngineConfig(K=1)
        assert cfg.K == 1

    def test_H_equals_one(self):
        cfg = EngineConfig(H=1, W=1)
        assert cfg.H == 1

    def test_echo_damping_zero(self):
        p = EchoParams(damping=0.0)
        assert p.damping == 0.0

    def test_echo_damping_one(self):
        p = EchoParams(damping=1.0)
        assert p.damping == 1.0

    def test_erosion_rate_zero(self):
        p = ErosionParams(rate=0.0)
        assert p.rate == 0.0

    def test_default_config_valid(self):
        cfg = EngineConfig()
        assert cfg.H == 32 and cfg.W == 32 and cfg.K == 8


# ---- from_dict / from_legacy_kwargs robustness ----

class TestConfigSerialization:
    def test_from_dict_missing_keys(self):
        d = {"H": 16, "W": 16}  # missing K and all sub-configs
        cfg = EngineConfig.from_dict(d)
        assert cfg.H == 16
        assert cfg.K == 8  # default

    def test_from_dict_extra_keys_ignored(self):
        d = EngineConfig().to_dict()
        d["nonexistent_key"] = 42
        cfg = EngineConfig.from_dict(d)
        assert cfg.H == 32  # still works

    def test_from_dict_roundtrip(self):
        cfg = EngineConfig(H=16, W=16, K=4)
        cfg.stress.knot_threshold = 0.5
        d = cfg.to_dict()
        cfg2 = EngineConfig.from_dict(d)
        assert cfg2.H == 16
        assert cfg2.stress.knot_threshold == 0.5

    def test_from_legacy_kwargs_known(self):
        cfg = EngineConfig.from_legacy_kwargs(wound_depth=0.5, knot_threshold=0.3)
        assert cfg.wound.depth == 0.5
        assert cfg.stress.knot_threshold == 0.3

    def test_from_legacy_kwargs_unknown_ignored(self):
        # Unknown kwargs are silently ignored (not in _FLAT_MAP)
        cfg = EngineConfig.from_legacy_kwargs(
            wound_depth=0.5, totally_unknown_param=999
        )
        assert cfg.wound.depth == 0.5
