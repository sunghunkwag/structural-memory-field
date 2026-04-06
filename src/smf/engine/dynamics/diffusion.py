"""Geodesic holographic diffusion + wave decay."""
from __future__ import annotations
import numpy as np
from smf.core.backend import NumpyBackend
from smf.core.operators import curvature_diffusion
from smf.config.params import DiffusionParams


def apply_diffusion(
    psi: np.ndarray,
    V: np.ndarray,
    R: np.ndarray,
    knots: np.ndarray,
    crystal: np.ndarray,
    idle: float,
    bk: NumpyBackend,
    dp: DiffusionParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Diffuse + decay psi. Returns (updated psi, amplitude)."""
    diffused = curvature_diffusion(psi, R, knots, crystal, bk, dp)
    psi = (1 - dp.alpha) * psi + dp.alpha * diffused
    decay = dp.active_base_decay + dp.V_modulation_scale * V
    idle_factor = dp.idle_decay if idle else 1.0
    # Compute full multiplier first, then apply — matches original operation order
    psi *= decay * idle_factor
    amp = bk.abs(psi)
    return psi, amp
