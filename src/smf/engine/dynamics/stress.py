"""Stress computation and knot formation."""
from __future__ import annotations
import numpy as np
from smf.core.backend import NumpyBackend
from smf.config.params import StressParams, BackendParams


def apply_stress_and_knots(
    V: np.ndarray,
    amp: np.ndarray,
    knots: np.ndarray,
    H: int,
    W: int,
    bk: NumpyBackend,
    sp: StressParams,
    bp: BackendParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute stress from gradients and grow knots. Returns (stress, knots)."""
    gv = bk.gradient_magnitude(V, bp.gradient_epsilon)
    ga = bk.gradient_magnitude(amp, bp.gradient_epsilon)
    raw_stress = (gv + ga * sp.gradient_amp_weight).sum(-1)
    # Replace NaN/Inf in raw_stress to prevent propagation
    raw_stress = np.where(np.isfinite(raw_stress), raw_stress, 0.0)
    stress_lap = bk.laplacian_2d(
        bk.expand_dim(raw_stress, -1), bp.laplacian_center_weight
    ).reshape(H, W)
    stress = bk.clamp(
        raw_stress + sp.diffusion_strength * stress_lap,
        0, sp.stress_max,
    )
    st3 = bk.expand_dim(stress, -1)
    knots = bk.clamp(
        knots
        + bk.maximum(st3 - sp.knot_threshold, 0)
        * bk.maximum(sp.knot_wound_threshold - V, 0)
        * sp.knot_formation_rate,
        0, sp.knot_max,
    )
    return stress, knots
