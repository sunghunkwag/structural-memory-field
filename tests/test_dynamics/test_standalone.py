"""Standalone tests — each dynamics module callable without the engine."""
import numpy as np
import pytest
from smf.core.backend import NumpyBackend
from smf.config.params import (
    WoundParams, BackendParams, InjectionParams, HealingParams,
    EchoParams, DiffusionParams, StressParams, CrystallizationParams,
    ErosionParams, CurvatureParams, PhaseParams,
)
from smf.engine.dynamics.wound import apply_wound, apply_phantom
from smf.engine.dynamics.injection import apply_injection
from smf.engine.dynamics.healing import apply_healing
from smf.engine.dynamics.echo import apply_echo
from smf.engine.dynamics.diffusion import apply_diffusion
from smf.engine.dynamics.stress import apply_stress_and_knots
from smf.engine.dynamics.crystallization import apply_crystallization, apply_crystallization_v3
from smf.engine.dynamics.erosion import apply_erosion
from smf.engine.dynamics.curvature import apply_curvature
from smf.engine.dynamics.phase import compute_boundary_tension, apply_phase_dynamics
from smf.engine.dynamics.readout import compute_action_scores


@pytest.fixture
def bk():
    return NumpyBackend()


@pytest.fixture
def fields(bk):
    """Create test fields (8x8x4)."""
    H, W, K = 8, 8, 4
    return {
        "psi": bk.zeros_complex((H, W, K)),
        "V": bk.ones_float((H, W, K)),
        "R": bk.ones_float((H, W, K)),
        "knots": bk.zeros_float((H, W, K)),
        "crystal": bk.zeros_float((H, W, K)),
        "echo": bk.zeros_float((H, W, K)),
        "stress": bk.zeros_float((H, W)),
        "crystal_phase": bk.zeros_float((H, W, K)),
        "boundary": bk.zeros_float((H, W, K)),
    }


def test_wound_standalone(bk, fields):
    stim = np.array([1.0, 0, 0, 0])
    V = apply_wound(fields["V"].copy(), stim, bk, WoundParams())
    assert V[0, 0, 0] < 1.0  # channel 0 wounded
    assert np.allclose(V[:, :, 1], 1.0)  # channel 1 untouched


def test_phantom_standalone(bk, fields):
    V = fields["V"].copy()
    V[2, 2, 0] = 0.3  # create a wound spot
    V_after = apply_phantom(V, bk, WoundParams(), BackendParams())
    # Phantom should affect neighbors
    assert not np.array_equal(V, V_after)


def test_injection_standalone(bk, fields):
    ev = bk.array_complex(np.array([1.0, 0, 0, 0]))
    psi = apply_injection(
        fields["psi"].copy(), fields["V"], fields["R"],
        fields["crystal"], ev, bk, InjectionParams(),
    )
    assert np.abs(psi[0, 0, 0]) > 0  # wave injected


def test_healing_standalone(bk, fields):
    V = fields["V"].copy()
    V[:] = 0.5  # wound everything
    V_healed = apply_healing(V, fields["knots"], fields["crystal"], bk,
                             HealingParams(), BackendParams())
    assert V_healed.mean() > 0.5  # healed toward 1.0


def test_echo_standalone(bk, fields):
    V = fields["V"].copy()
    V_before = fields["V"].copy()
    V[:] = 0.8  # simulate wound
    V_new, echo_new = apply_echo(V, V_before, fields["echo"], bk, EchoParams())
    assert not np.allclose(echo_new, 0)  # echo activated


def test_diffusion_standalone(bk, fields):
    psi = fields["psi"].copy()
    psi[4, 4, 0] = 1.0 + 0j  # point source
    psi_new, amp = apply_diffusion(
        psi, fields["V"], fields["R"], fields["knots"],
        fields["crystal"], 0.0, bk, DiffusionParams(),
    )
    # Wave should spread from point source
    assert np.abs(psi_new[3, 4, 0]) > 0


def test_stress_standalone(bk, fields):
    V = fields["V"].copy()
    V[2:4, 2:4, :] = 0.3  # localized wound
    amp = bk.abs(fields["psi"])
    stress, knots = apply_stress_and_knots(
        V, amp, fields["knots"].copy(), 8, 8, bk,
        StressParams(knot_threshold=0.3), BackendParams(),
    )
    assert stress.max() > 0  # stress from gradient


def test_crystallization_standalone(bk, fields):
    knots = fields["knots"].copy()
    knots[:] = 0.5  # above threshold
    crystal = apply_crystallization(knots, fields["crystal"].copy(), bk,
                                    CrystallizationParams())
    assert crystal.sum() > 0  # crystals formed


def test_erosion_standalone(bk, fields):
    knots = fields["knots"].copy()
    knots[:] = 1.0
    V = fields["V"].copy()  # V=1.0, above eligibility
    knots_after = apply_erosion(V, knots, 0.0, bk, ErosionParams())
    assert knots_after.sum() < knots.sum()  # eroded


def test_curvature_standalone(bk, fields):
    stress = fields["stress"].copy()
    stress[:] = 5.0
    amp = bk.ones_float((8, 8, 4)) * 0.5
    R = apply_curvature(fields["R"].copy(), stress, amp, bk, CurvatureParams())
    assert R.mean() > 1.0  # curvature grew


def test_phase_standalone(bk, fields):
    crystal = fields["crystal"].copy()
    crystal[:] = 0.5
    crystal_phase = fields["crystal_phase"].copy()
    crystal_phase[0:4, :, :] = 0.0
    crystal_phase[4:8, :, :] = np.pi
    from smf.config.params import PhaseParams
    boundary, neighbor = compute_boundary_tension(crystal, crystal_phase, bk, PhaseParams())
    assert boundary.max() > 0  # tension at phase boundary


def test_readout_standalone(bk, fields):
    V = fields["V"].copy()
    V[:, :, 0] = 0.5  # wound channel 0
    amp = bk.ones_float((8, 8, 4)) * 0.1
    from smf.config.params import ReadoutParams
    scores = compute_action_scores(
        V, amp, fields["knots"], fields["stress"], fields["crystal"],
        4, ReadoutParams(),
    )
    assert len(scores) == 4
    assert scores[0] > scores[1]  # wounded channel has higher score
