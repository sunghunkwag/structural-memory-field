"""V3 crystal phase dynamics: boundary tension, fracture, fusion, nucleation."""
from __future__ import annotations
import numpy as np
from smf.core.backend import NumpyBackend
from smf.config.params import PhaseParams


def compute_boundary_tension(
    crystal: np.ndarray,
    crystal_phase: np.ndarray,
    bk: NumpyBackend,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute boundary tension and neighbour crystal mean."""
    cp = crystal_phase
    cr = crystal
    dp_up = bk.abs(bk.sin(cp - bk.roll(cp, -1, 0)))
    dp_dn = bk.abs(bk.sin(cp - bk.roll(cp, 1, 0)))
    dp_lt = bk.abs(bk.sin(cp - bk.roll(cp, -1, 1)))
    dp_rt = bk.abs(bk.sin(cp - bk.roll(cp, 1, 1)))
    phase_gradient = (dp_up + dp_dn + dp_lt + dp_rt) / 4.0

    cr_up = bk.roll(cr, -1, 0)
    cr_dn = bk.roll(cr, 1, 0)
    cr_lt = bk.roll(cr, -1, 1)
    cr_rt = bk.roll(cr, 1, 1)
    neighbor_crystal = (cr_up + cr_dn + cr_lt + cr_rt) / 4.0

    boundary = phase_gradient * cr * neighbor_crystal
    return boundary, neighbor_crystal


def apply_phase_dynamics(
    crystal: np.ndarray,
    crystal_phase: np.ndarray,
    knots: np.ndarray,
    psi: np.ndarray,
    V: np.ndarray,
    boundary: np.ndarray,
    neighbor_crystal: np.ndarray,
    bk: NumpyBackend,
    pp: PhaseParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fracture, fusion, nucleation. Returns (crystal, knots, psi)."""
    # Fracture
    fracture = bk.maximum(boundary - pp.fracture_threshold, 0)
    shatter = fracture * crystal * pp.fracture_shatter_multiplier
    knots = knots + shatter
    crystal = bk.maximum(crystal - fracture * pp.fracture_decay, 0)
    burst = (fracture * np.exp(1j * crystal_phase)).astype(np.complex64)
    psi = psi + burst * pp.fracture_energy_release

    # Fusion
    low_tension = bk.maximum(pp.fusion_phase_tolerance - boundary, 0)
    fusion = low_tension * crystal * neighbor_crystal
    crystal = bk.clamp(crystal + fusion * pp.fusion_rate, 0, 1)

    # Nucleation
    empty_space = bk.maximum(1.0 - crystal, 0)
    viable = bk.maximum(V - pp.nucleation_viability_threshold, 0)
    nucleation = boundary * empty_space * viable * pp.nucleation_rate
    knots = knots + nucleation

    return crystal, knots, psi
