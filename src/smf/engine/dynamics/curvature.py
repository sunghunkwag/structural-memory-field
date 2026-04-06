"""Ricci curvature field update."""
from __future__ import annotations
import numpy as np
from smf.core.backend import NumpyBackend
from smf.config.params import CurvatureParams


def apply_curvature(
    R: np.ndarray,
    stress: np.ndarray,
    amp: np.ndarray,
    bk: NumpyBackend,
    cp: CurvatureParams,
) -> np.ndarray:
    """Update curvature from stress and amplitude. Returns updated R."""
    st3 = bk.expand_dim(stress, -1) if stress.ndim == 2 else stress
    return bk.clamp(
        R + st3 * amp * cp.plasticity + cp.relaxation_rate * (1 - R),
        cp.R_min, cp.R_max,
    )
