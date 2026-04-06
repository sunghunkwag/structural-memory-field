"""Serialization roundtrip tests."""
import numpy as np
from smf.engine.v2 import EngineV2
from smf.engine.v3 import EngineV3
from smf.io.serialization import save_state, load_state


def test_save_load_roundtrip_v2(tmp_path):
    e1 = EngineV2()
    for _ in range(30):
        e1.step(np.array([1, 0, 0, 0, 0, 0, 0, 0]))
    save_state(e1, str(tmp_path / "state.npz"))
    e2 = EngineV2()
    state = load_state(str(tmp_path / "state.npz"))
    e2.load_state(state)
    np.testing.assert_array_equal(e1.V, e2.V)
    np.testing.assert_array_equal(e1.psi, e2.psi)
    np.testing.assert_array_equal(e1.knots, e2.knots)
    np.testing.assert_array_equal(e1.stress, e2.stress)
    assert e1.t == e2.t


def test_save_load_roundtrip_v3(tmp_path):
    e1 = EngineV3(knot_threshold=0.3)
    for _ in range(30):
        e1.step(np.array([2, 0, 0, 0, 0, 0, 0, 0]),
                np.array([0.5, 0, 0, 0, 0, 0, 0, 0]))
    save_state(e1, str(tmp_path / "state_v3.npz"))
    e2 = EngineV3(knot_threshold=0.3)
    state = load_state(str(tmp_path / "state_v3.npz"))
    e2.load_state(state)
    np.testing.assert_array_equal(e1.V, e2.V)
    np.testing.assert_array_equal(e1.crystal_phase, e2.crystal_phase)
    np.testing.assert_array_equal(e1.boundary, e2.boundary)
    assert e1.t == e2.t


def test_save_includes_config(tmp_path):
    e1 = EngineV2(wound_depth=0.5, knot_threshold=0.3)
    for _ in range(10):
        e1.step(np.array([1, 0, 0, 0, 0, 0, 0, 0]))
    save_state(e1, str(tmp_path / "state.npz"))
    state = load_state(str(tmp_path / "state.npz"))
    assert "config" in state
    assert state["config"]["wound"]["depth"] == 0.5
