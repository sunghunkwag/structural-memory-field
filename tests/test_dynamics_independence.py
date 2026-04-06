"""Structural enforcement: dynamics modules must be standalone.

Prevents future regressions by verifying:
1. No dynamics module imports engine classes
2. No dynamics function has 'self' or 'engine' parameters
"""
import inspect
from smf.engine.dynamics import (
    wound, injection, healing, echo, diffusion,
    stress, crystallization, erosion, curvature, phase, readout,
)

ALL = [wound, injection, healing, echo, diffusion,
       stress, crystallization, erosion, curvature, phase, readout]


def test_no_engine_imports():
    """Dynamics modules must not import engine classes."""
    for mod in ALL:
        src = inspect.getsource(mod)
        assert "EngineV2" not in src, f"{mod.__name__} references EngineV2"
        assert "EngineV3" not in src, f"{mod.__name__} references EngineV3"
        assert "BaseEngine" not in src, f"{mod.__name__} references BaseEngine"


def test_no_self_parameter():
    """Dynamics functions must be standalone (no 'self' or 'engine' args)."""
    for mod in ALL:
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if name.startswith("_"):
                continue
            params = list(inspect.signature(fn).parameters.keys())
            assert "self" not in params, f"{mod.__name__}.{name} has 'self'"
            assert "engine" not in params, f"{mod.__name__}.{name} has 'engine'"
