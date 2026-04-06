"""Migrated v2 tests — all 14 original tests as pytest."""
import numpy as np
from collections import Counter
from smf.engine.v2 import EngineV2


s_A = np.array([1, 0, 0, 0, 0, 0, 0, 0])
s_B = np.array([0, 0, 1, 0, 0, 0, 0, 0])
empty = np.zeros(8)


def test_t1_resonance_cascade():
    e1 = EngineV2(resonance_range=0.5)
    e2 = EngineV2(resonance_range=0)
    for _ in range(50):
        e1.step(s_A); e2.step(s_A)
    assert e1.V[5:, :, 0].mean() < e2.V[5:, :, 0].mean()


def test_t2_crystallization():
    e = EngineV2(knot_threshold=0.3, crystal_threshold=0.1)
    for _ in range(80):
        e.step(s_A)
    assert e.crystal[:, :, 0].sum() > 1.0


def test_t3_scar_erosion():
    e4 = EngineV2(knot_threshold=0.3, erosion_rate=0.5)
    e5 = EngineV2(knot_threshold=0.3, erosion_rate=0.0)
    for _ in range(50):
        e4.step(s_A); e5.step(s_A)
    k4 = e4.knots.sum(); k5 = e5.knots.sum()
    for _ in range(100):
        e4.step(empty); e5.step(empty)
    assert (k4 - e4.knots.sum()) > (k5 - e5.knots.sum())


def test_t4_wound_echo():
    e_echo = EngineV2(echo_coupling=0.1)
    e_noecho = EngineV2(echo_coupling=0)
    for _ in range(30):
        e_echo.step(s_A); e_noecho.step(s_A)
    v_ec = []; v_ne = []
    for _ in range(40):
        e_echo.step(empty); e_noecho.step(empty)
        v_ec.append(e_echo.V[:, :, 0].mean())
        v_ne.append(e_noecho.V[:, :, 0].mean())
    assert np.var(v_ec) > np.var(v_ne)


def test_t5_holographic_recall():
    e = EngineV2()
    s_AB = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    for _ in range(50):
        e.step(s_AB, np.array([0.7, -0.7, 0, 0, 0, 0, 0, 0]))
    e.psi *= 0
    for _ in range(20):
        e.step(s_A, np.array([0.7, 0, 0, 0, 0, 0, 0, 0]))
    a = np.abs(e.psi)
    assert a[:, :, 1].mean() > a[:, :, 2].mean() * 3


def test_t6_behavioral_differentiation():
    e = EngineV2(knot_threshold=1.2)
    aA = [e.step(s_A) for _ in range(80)]
    aB = [e.step(s_B) for _ in range(80)]
    assert Counter(aA).most_common(1)[0][0] != Counter(aB).most_common(1)[0][0]


def test_t7_timestep_counter():
    e = EngineV2()
    for _ in range(25):
        e.step(s_A)
    assert e.t == 25


def test_t8_state_reset():
    e = EngineV2()
    for _ in range(30):
        e.step(s_A)
    e.reset()
    assert e.t == 0
    assert e.V.mean() == 1.0
    assert e.knots.sum() == 0.0
    assert np.abs(e.psi).sum() == 0.0


def test_t9_save_load_state():
    e = EngineV2(knot_threshold=0.3)
    for _ in range(40):
        e.step(s_A)
    saved = e.get_state()
    t_saved = e.t
    knots_saved = e.knots.sum()
    for _ in range(20):
        e.step(s_B)
    e.load_state(saved)
    assert e.t == t_saved
    assert abs(e.knots.sum() - knots_saved) < 1e-6


def test_t10_diagnostics_completeness():
    e = EngineV2()
    for _ in range(10):
        e.step(s_A)
    diag = e.get_diagnostics()
    required = {'t', 'psi_mean', 'psi_max', 'V_mean', 'V_min',
                'knots_total', 'knots_max', 'crystal_total', 'crystal_max',
                'stress_mean', 'stress_max', 'echo_energy', 'R_mean', 'R_max',
                'field_energy'}
    assert required.issubset(set(diag.keys()))


def test_t11_zero_input_stability():
    e = EngineV2()
    for _ in range(200):
        e.step(empty)
    assert e.V.mean() > 0.99
    assert np.abs(e.psi).max() < 1e-6


def test_t12_multi_channel_independence():
    e = EngineV2(knot_threshold=0.3)
    s_ch0 = np.array([5, 0, 0, 0, 0, 0, 0, 0])
    for _ in range(60):
        e.step(s_ch0)
    assert e.crystal[:, :, 0].sum() > e.crystal[:, :, 3].sum() * 2


def test_t13_idle_energy_decay():
    e = EngineV2()
    for _ in range(40):
        e.step(s_A)
    energies = []
    for _ in range(50):
        e.step(empty)
        energies.append(e.get_diagnostics()['field_energy'])
    decay_count = sum(1 for i in range(1, len(energies))
                      if energies[i] <= energies[i-1] * 1.01)
    assert decay_count > len(energies) * 0.7


def test_t14_stress_proportionality():
    e_lo = EngineV2()
    e_hi = EngineV2()
    s_lo = np.array([0.5, 0, 0, 0, 0, 0, 0, 0])
    s_hi = np.array([3.0, 0, 0, 0, 0, 0, 0, 0])
    for _ in range(30):
        e_lo.step(s_lo); e_hi.step(s_hi)
    assert e_hi.stress.mean() > e_lo.stress.mean()
