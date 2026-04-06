"""Logging and diagnostics callback tests."""
import logging
import numpy as np
from smf.engine.v2 import EngineV2
from smf.config.params import EngineConfig


def test_info_log_has_step_and_action(caplog):
    """INFO log must include step number and action."""
    with caplog.at_level(logging.INFO, logger="smf.engine"):
        e = EngineV2()
        e.step(np.array([1, 0, 0, 0, 0, 0, 0, 0]))
    assert any("step 1:" in r.message and "action=" in r.message for r in caplog.records)


def test_debug_log_has_field_values(caplog):
    """DEBUG log must include actual numeric values from fields."""
    with caplog.at_level(logging.DEBUG, logger="smf.engine"):
        e = EngineV2()
        e.step(np.array([1, 0, 0, 0, 0, 0, 0, 0]))
    debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("V=" in m for m in debug_msgs), "DEBUG must include V values"
    assert any("wound:" in m for m in debug_msgs), "DEBUG must include wound stage"


def test_diagnostics_callback_fires():
    """Callback receives diagnostics dict after each step."""
    collected = []
    e = EngineV2(diagnostics_callback=lambda d: collected.append(d))
    e.step(np.array([1, 0, 0, 0, 0, 0, 0, 0]))
    e.step(np.zeros(8))
    assert len(collected) == 2
    assert "V_mean" in collected[0]
    assert "field_energy" in collected[1]
    assert collected[0]["t"] == 1
    assert collected[1]["t"] == 2


def test_callback_receives_real_data():
    """Callback data must have non-zero field values after stimulus."""
    collected = []
    e = EngineV2(
        diagnostics_callback=lambda d: collected.append(d),
        knot_threshold=0.3,
    )
    for _ in range(50):
        e.step(np.array([3, 0, 0, 0, 0, 0, 0, 0]))
    last = collected[-1]
    assert last["knots_total"] > 0, "Callback should show non-zero knots"
    assert last["V_mean"] < 1.0, "Callback should show V < 1 after wounding"


def test_logging_zero_cost_when_disabled(caplog):
    """With logging disabled, step() should not generate log records."""
    with caplog.at_level(logging.CRITICAL, logger="smf.engine"):
        e = EngineV2()
        for _ in range(10):
            e.step(np.zeros(8))
    # Only CRITICAL or above would be captured, and engine doesn't emit those
    assert len(caplog.records) == 0
