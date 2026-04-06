"""Void crystallization."""
from __future__ import annotations
import numpy as np
from smf.core.backend import NumpyBackend
from smf.config.params import CrystallizationParams


def apply_crystallization(
    knots: np.ndarray,
    crystal: np.ndarray,
    bk: NumpyBackend,
    cp: CrystallizationParams,
) -> np.ndarray:
    """Grow crystals from excess knots. Returns updated crystal."""
    new_crystal = bk.maximum(knots - cp.threshold, 0) * cp.growth_rate
    return bk.clamp(crystal * cp.decay_rate + new_crystal, 0, 1)


def apply_crystallization_v3(
    knots: np.ndarray,
    crystal: np.ndarray,
    crystal_phase: np.ndarray,
    psi: np.ndarray,
    bk: NumpyBackend,
    cp: CrystallizationParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """V3 crystallization with phase imprinting.

    Returns (new_crystal_amount, updated crystal, updated crystal_phase).
    """
    new_crystal = bk.maximum(knots - cp.threshold, 0) * cp.growth_rate
    forming = bk.array_float((new_crystal > cp.forming_threshold))
    psi_angle = bk.angle(psi)
    crystal_phase = crystal_phase * (1 - forming) + psi_angle * forming
    crystal = bk.clamp(crystal * cp.decay_rate + new_crystal, 0, 1)
    return new_crystal, crystal, crystal_phase
