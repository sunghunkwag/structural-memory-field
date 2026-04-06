"""THE anti-cheat test.

For EVERY parameter P in the config:
  1. Run engine N steps with default P → hash all field arrays + actions
  2. Run engine N steps with mutated P → hash all fields + actions
  3. assert hash1 != hash2

If this passes for all parameters, every parameter is real.
If it fails for any, that parameter is fake/unused.
"""
import hashlib
import numpy as np
import pytest
from dataclasses import fields as dc_fields
from smf.engine.v3 import EngineV3
from smf.config.params import EngineConfig


def _make_stimuli():
    """Strong mixed stimuli with phases to exercise ALL pathways.

    Uses multi-channel stimuli so readout params can swing action selection.
    Includes long idle periods so erosion can activate after V heals.
    """
    stims = []
    phases = []
    # Strong wound ch0 + ch2 (close competition for readout)
    for _ in range(25):
        stims.append(np.array([4, 0, 3.5, 0, 0, 0, 0, 0]))
        phases.append(np.array([1.0, 0, 0.8, 0, 0, 0, 0, 0]))
    # Conflicting phase (triggers fracture/nucleation)
    for _ in range(25):
        stims.append(np.array([4, 0, 3.5, 0, 0, 0, 0, 0]))
        phases.append(np.array([-2.0, 0, -1.5, 0, 0, 0, 0, 0]))
    # Long idle (V heals back above threshold → erosion activates)
    for _ in range(50):
        stims.append(np.zeros(8))
        phases.append(np.zeros(8))
    # Moderate multi-channel
    for _ in range(15):
        stims.append(np.array([2, 2, 2, 0, 0, 0, 0, 0]))
        phases.append(np.array([0.5, -0.5, 0.3, 0, 0, 0, 0, 0]))
    # Tiny stimulus (exercises idle threshold)
    for _ in range(5):
        stims.append(np.array([1e-7, 0, 0, 0, 0, 0, 0, 0]))
        phases.append(np.zeros(8))
    return stims, phases


STIMULI, PHASES = _make_stimuli()

# Low knot threshold, low crystal threshold, low erosion threshold,
# low fracture threshold — ensures all parameter pathways activate
BASE_OVERRIDES = {
    "stress": {"knot_threshold": 0.3},
    "crystallization": {"threshold": 0.05},
    "erosion": {"eligibility_threshold": 0.3},  # Low so erosion activates in wounded regions
    "phase": {"fracture_threshold": 0.05},
}


def _apply_overrides(cfg: EngineConfig, overrides: dict):
    for group_name, params in overrides.items():
        group = getattr(cfg, group_name)
        for k, v in params.items():
            setattr(group, k, v)


def _run_and_hash(cfg: EngineConfig) -> str:
    """Run engine with stimuli, return hash of all fields + actions."""
    e = EngineV3(cfg=cfg)
    h = hashlib.sha256()
    actions = []
    for s, ph in zip(STIMULI, PHASES):
        a = e.step(s, ph)
        actions.append(a)
    for name in ("psi", "V", "R", "knots", "crystal", "echo", "stress",
                 "crystal_phase", "boundary"):
        arr = getattr(e.f, name)
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(bytes(actions))
    return h.hexdigest()


# Per-parameter mutation strategies
_SPECIAL_MUTATIONS = {
    # Thresholds that won't be crossed by simple *3: set to values that force activation
    ("idle", "threshold"): 10.0,           # Makes engine think all input is idle
    ("stress", "stress_max"): 0.5,         # Very low clamp — stress clipped early
    ("stress", "knot_max"): 0.5,           # Very low clamp — knots clipped early
    ("injection", "curvature_recall_max"): 0.1,  # Very low — clips recall
    ("curvature", "R_min"): 1.5,           # Above initial R=1.0 — forces R up
    ("curvature", "R_max"): 1.01,          # Very tight ceiling — clips R growth
    ("erosion", "eligibility_threshold"): 0.1,    # Much lower — triggers erosion everywhere
    ("erosion", "rate"): 5.0,              # Extreme rate to show clear effect
    ("diffusion", "conductivity_epsilon"): 1.0,   # Large epsilon disrupts conductivity
    # Readout: these only affect action selection, not fields — use extreme values
    ("readout", "num_actions"): 1,         # Only 1 action available → always 0
    ("readout", "stress_offset"): 1000.0,  # Overwhelms score computation
    ("readout", "knot_amplification"): 100.0,     # Overwhelms via knots
    ("readout", "crystal_amplification"): 100.0,  # Overwhelms via crystal
}


def _mutate(group_name, param_name, value):
    """Mutate a parameter value."""
    key = (group_name, param_name)
    if key in _SPECIAL_MUTATIONS:
        return _SPECIAL_MUTATIONS[key]
    if isinstance(value, int):
        return max(1, int(value * 3))
    if value == 0.0:
        return 0.1
    if abs(value) > 1.0:
        return value / 3.0
    return value * 3.0


def _collect_params():
    """Collect all (group_name, param_name, default_value) triples."""
    cfg = EngineConfig()
    result = []
    for group_field in dc_fields(cfg):
        group = getattr(cfg, group_field.name)
        if not hasattr(group, '__dataclass_fields__'):
            continue
        for param_field in dc_fields(group):
            val = getattr(group, param_field.name)
            if isinstance(val, (int, float)):
                result.append((group_field.name, param_field.name, val))
    return result


ALL_PARAMS = _collect_params()


@pytest.mark.parametrize(
    "group_name,param_name,default_val",
    ALL_PARAMS,
    ids=[f"{g}.{p}" for g, p, _ in ALL_PARAMS],
)
def test_param_mutation(group_name, param_name, default_val):
    """Mutating parameter must change engine behavior."""
    # Default config
    cfg_default = EngineConfig()
    _apply_overrides(cfg_default, BASE_OVERRIDES)
    hash_default = _run_and_hash(cfg_default)

    # Mutated config
    cfg_mutated = EngineConfig()
    _apply_overrides(cfg_mutated, BASE_OVERRIDES)
    group = getattr(cfg_mutated, group_name)
    mutated_val = _mutate(group_name, param_name, default_val)
    setattr(group, param_name, mutated_val)
    hash_mutated = _run_and_hash(cfg_mutated)

    assert hash_default != hash_mutated, (
        f"Changing {group_name}.{param_name} from {default_val} to {mutated_val} "
        f"had NO effect — parameter is unused!"
    )
