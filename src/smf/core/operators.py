"""Thin wrappers — physics operators delegating to backend.

Kept separate so dynamics modules import operators, not backend directly.
"""
from __future__ import annotations
import numpy as np
from smf.core.backend import NumpyBackend
from smf.config.params import DiffusionParams, BackendParams


def laplacian_2d(x: np.ndarray, bk: NumpyBackend, bp: BackendParams) -> np.ndarray:
    """2D discrete Laplacian."""
    return bk.laplacian_2d(x, bp.laplacian_center_weight)


def gradient_magnitude(x: np.ndarray, bk: NumpyBackend, bp: BackendParams) -> np.ndarray:
    """Gradient magnitude via forward differences."""
    return bk.gradient_magnitude(x, bp.gradient_epsilon)


def curvature_diffusion(
    wave: np.ndarray,
    curvature: np.ndarray,
    knots: np.ndarray,
    crystal: np.ndarray,
    bk: NumpyBackend,
    dp: DiffusionParams,
) -> np.ndarray:
    """Geodesic holographic diffusion."""
    return bk.curvature_diffusion(
        wave, curvature, knots, crystal,
        crystal_block_strength=dp.crystal_block_strength,
        knot_resistance=dp.knot_resistance,
        conductivity_epsilon=dp.conductivity_epsilon,
        scatter_curvature_offset=dp.scatter_curvature_offset,
        scatter_strength=dp.scatter_strength,
        crystal_wave_trap=dp.crystal_wave_trap,
    )
