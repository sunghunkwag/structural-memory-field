"""Migrated v3 tests — all 20 original tests as pytest."""
import numpy as np
from collections import Counter
from smf.engine.v3 import EngineV3


s_A = np.array([1, 0, 0, 0, 0, 0, 0, 0])
s_B = np.array([0, 0, 1, 0, 0, 0, 0, 0])
empty = np.zeros(8)


# --- T1-T6: v2 core tests on v3 engine ---

def test_t1_resonance_cascade():
    e1 = EngineV3(resonance_range=0.5)
    e2 = EngineV3(resonance_range=0)
    for _ in range(50):
        e1.step(s_A); e2.step(s_A)
    assert e1.V[5:, :, 0].mean() < e2.V[5:, :, 0].mean()


def test_t2_crystallization():
    e = EngineV3(knot_threshold=0.3, crystal_threshold=0.1)
    for _ in range(80):
        e.step(s_A)
    assert e.crystal[:, :, 0].sum() > 1.0


def test_t3_scar_erosion():
    e4 = EngineV3(knot_threshold=0.3, erosion_rate=0.5)
    e5 = EngineV3(knot_threshold=0.3, erosion_rate=0.0)
    for _ in range(50):
        e4.step(s_A); e5.step(s_A)
    k4 = e4.knots.sum(); k5 = e5.knots.sum()
    for _ in range(100):
        e4.step(empty); e5.step(empty)
    assert (k4 - e4.knots.sum()) > (k5 - e5.knots.sum())


def test_t4_wound_echo():
    e_echo = EngineV3(echo_coupling=0.1)
    e_noecho = EngineV3(echo_coupling=0)
    for _ in range(30):
        e_echo.step(s_A); e_noecho.step(s_A)
    v_ec = []; v_ne = []
    for _ in range(40):
        e_echo.step(empty); e_noecho.step(empty)
        v_ec.append(e_echo.V[:, :, 0].mean())
        v_ne.append(e_noecho.V[:, :, 0].mean())
    assert np.var(v_ec) > np.var(v_ne)


def test_t5_holographic_recall():
    e = EngineV3()
    s_AB = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    for _ in range(50):
        e.step(s_AB, np.array([0.7, -0.7, 0, 0, 0, 0, 0, 0]))
    e.psi *= 0
    for _ in range(20):
        e.step(s_A, np.array([0.7, 0, 0, 0, 0, 0, 0, 0]))
    a = np.abs(e.psi)
    assert a[:, :, 1].mean() > a[:, :, 2].mean() * 3


def test_t6_behavioral_differentiation():
    e = EngineV3(knot_threshold=1.2)
    aA = [e.step(s_A) for _ in range(80)]
    aB = [e.step(s_B) for _ in range(80)]
    assert Counter(aA).most_common(1)[0][0] != Counter(aB).most_common(1)[0][0]


# --- T7-T12: v3 crystal phase dynamics tests ---

def test_t7_crystal_phase_imprinting():
    e_pa = EngineV3(knot_threshold=0.3, crystal_threshold=0.1)
    e_pb = EngineV3(knot_threshold=0.3, crystal_threshold=0.1)
    for _ in range(80):
        e_pa.step(s_A, np.array([0.5, 0, 0, 0, 0, 0, 0, 0]))
        e_pb.step(s_A, np.array([-1.5, 0, 0, 0, 0, 0, 0, 0]))
    mask_a = e_pa.crystal[:, :, 0] > 0.3
    mask_b = e_pb.crystal[:, :, 0] > 0.3
    assert mask_a.any() and mask_b.any()
    mean_phase_a = np.mean(e_pa.crystal_phase[:, :, 0][mask_a])
    mean_phase_b = np.mean(e_pb.crystal_phase[:, :, 0][mask_b])
    assert abs(mean_phase_a - mean_phase_b) > 0.1


def test_t8_boundary_tension():
    e = EngineV3(knot_threshold=0.3, crystal_threshold=0.1, fracture_threshold=10.0)
    for _ in range(60):
        e.step(s_A, np.array([1.0, 0, 0, 0, 0, 0, 0, 0]))
    for _ in range(60):
        e.step(s_A, np.array([-2.0, 0, 0, 0, 0, 0, 0, 0]))
    assert e.boundary[:, :, 0].max() > 0.01


def test_t9_crystal_fracture():
    e_frac = EngineV3(knot_threshold=0.3, crystal_threshold=0.1, fracture_threshold=0.05)
    e_nofrac = EngineV3(knot_threshold=0.3, crystal_threshold=0.1, fracture_threshold=100.0)
    for _ in range(60):
        e_frac.step(s_A, np.array([1.0, 0, 0, 0, 0, 0, 0, 0]))
        e_nofrac.step(s_A, np.array([1.0, 0, 0, 0, 0, 0, 0, 0]))
    for _ in range(60):
        e_frac.step(s_A, np.array([-2.0, 0, 0, 0, 0, 0, 0, 0]))
        e_nofrac.step(s_A, np.array([-2.0, 0, 0, 0, 0, 0, 0, 0]))
    assert e_frac.crystal[:, :, 0].sum() < e_nofrac.crystal[:, :, 0].sum()


def test_t10_crystal_fusion():
    e_fuse = EngineV3(knot_threshold=0.3, crystal_threshold=0.1,
                      fusion_rate=0.15, fracture_threshold=100.0)
    e_nofuse = EngineV3(knot_threshold=0.3, crystal_threshold=0.1,
                        fusion_rate=0.0, fracture_threshold=100.0)
    for _ in range(100):
        e_fuse.step(s_A, np.array([0.5, 0, 0, 0, 0, 0, 0, 0]))
        e_nofuse.step(s_A, np.array([0.5, 0, 0, 0, 0, 0, 0, 0]))
    assert e_fuse.crystal[:, :, 0].sum() > e_nofuse.crystal[:, :, 0].sum()


def test_t11_boundary_nucleation():
    e_nuc = EngineV3(knot_threshold=0.3, crystal_threshold=0.1,
                     nucleation_rate=0.3, fracture_threshold=100.0)
    e_nonuc = EngineV3(knot_threshold=0.3, crystal_threshold=0.1,
                       nucleation_rate=0.0, fracture_threshold=100.0)
    for _ in range(60):
        e_nuc.step(s_A, np.array([1.0, 0, 0, 0, 0, 0, 0, 0]))
        e_nonuc.step(s_A, np.array([1.0, 0, 0, 0, 0, 0, 0, 0]))
    for _ in range(60):
        e_nuc.step(s_A, np.array([-2.0, 0, 0, 0, 0, 0, 0, 0]))
        e_nonuc.step(s_A, np.array([-2.0, 0, 0, 0, 0, 0, 0, 0]))
    assert e_nuc.knots[:, :, 0].sum() > e_nonuc.knots[:, :, 0].sum()


def test_t12_fracture_wave_burst():
    e_burst = EngineV3(knot_threshold=0.3, crystal_threshold=0.1, fracture_threshold=0.05)
    e_noburst = EngineV3(knot_threshold=0.3, crystal_threshold=0.1, fracture_threshold=100.0)
    for _ in range(60):
        e_burst.step(s_A, np.array([1.0, 0, 0, 0, 0, 0, 0, 0]))
        e_noburst.step(s_A, np.array([1.0, 0, 0, 0, 0, 0, 0, 0]))
    burst_wins = 0
    total = 60
    for _ in range(total):
        e_burst.step(s_A, np.array([-2.0, 0, 0, 0, 0, 0, 0, 0]))
        e_noburst.step(s_A, np.array([-2.0, 0, 0, 0, 0, 0, 0, 0]))
        if np.abs(e_burst.psi[:, :, 0]).mean() > np.abs(e_noburst.psi[:, :, 0]).mean():
            burst_wins += 1
    assert burst_wins > total // 2


# --- T13-T20: v3 extended validation tests ---

def test_t13_timestep_counter():
    e = EngineV3()
    for _ in range(25):
        e.step(s_A)
    assert e.t == 25


def test_t14_state_reset():
    e = EngineV3()
    for _ in range(30):
        e.step(s_A)
    e.reset()
    assert e.t == 0 and e.V.mean() == 1.0 and e.knots.sum() == 0.0
    assert np.abs(e.psi).sum() == 0.0
    assert e.boundary.sum() == 0.0 and e.crystal_phase.sum() == 0.0


def test_t15_save_load_state():
    e = EngineV3(knot_threshold=0.3)
    for _ in range(40):
        e.step(s_A)
    saved = e.get_state()
    t_saved = e.t
    knots_saved = e.knots.sum()
    crystal_saved = e.crystal.sum()
    for _ in range(20):
        e.step(s_B)
    e.load_state(saved)
    assert e.t == t_saved
    assert abs(e.knots.sum() - knots_saved) < 1e-6
    assert abs(e.crystal.sum() - crystal_saved) < 1e-6


def test_t16_diagnostics_completeness():
    e = EngineV3()
    for _ in range(10):
        e.step(s_A)
    diag = e.get_diagnostics()
    required = {'t', 'psi_mean', 'psi_max', 'V_mean', 'V_min',
                'knots_total', 'knots_max', 'crystal_total', 'crystal_max',
                'crystal_phase_std', 'boundary_max', 'boundary_mean',
                'stress_mean', 'stress_max', 'echo_energy', 'R_mean', 'R_max',
                'field_energy'}
    assert required.issubset(set(diag.keys()))


def test_t17_zero_input_stability():
    e = EngineV3()
    for _ in range(200):
        e.step(empty)
    assert e.V.mean() > 0.99
    assert np.abs(e.psi).max() < 1e-6


def test_t18_crystal_phase_persistence():
    e = EngineV3(knot_threshold=0.3, crystal_threshold=0.1, fracture_threshold=100.0)
    for _ in range(80):
        e.step(s_A, np.array([1.0, 0, 0, 0, 0, 0, 0, 0]))
    phase_before = e.crystal_phase.copy()
    crystal_before = e.crystal.copy()
    for _ in range(20):
        e.step(empty)
    mask = crystal_before[:, :, 0] > 0.3
    assert mask.any()
    phase_drift = np.abs(e.crystal_phase[:, :, 0][mask] - phase_before[:, :, 0][mask]).mean()
    assert phase_drift < 0.5


def test_t19_channel_independence():
    e = EngineV3(knot_threshold=0.3)
    s_ch0 = np.array([5, 0, 0, 0, 0, 0, 0, 0])
    for _ in range(60):
        e.step(s_ch0)
    assert e.crystal[:, :, 0].sum() > e.crystal[:, :, 3].sum() * 2


def test_t20_fracture_fusion_antagonism():
    e_both = EngineV3(knot_threshold=0.3, crystal_threshold=0.1,
                      fracture_threshold=0.05, fusion_rate=0.15)
    e_fuse = EngineV3(knot_threshold=0.3, crystal_threshold=0.1,
                      fracture_threshold=100.0, fusion_rate=0.15)
    for _ in range(60):
        e_both.step(s_A, np.array([1.0, 0, 0, 0, 0, 0, 0, 0]))
        e_fuse.step(s_A, np.array([1.0, 0, 0, 0, 0, 0, 0, 0]))
    for _ in range(60):
        e_both.step(s_A, np.array([-2.0, 0, 0, 0, 0, 0, 0, 0]))
        e_fuse.step(s_A, np.array([-2.0, 0, 0, 0, 0, 0, 0, 0]))
    assert e_both.crystal[:, :, 0].sum() < e_fuse.crystal[:, :, 0].sum()
