"""Geodesic holographic diffusion + wave decay."""
from __future__ import annotations
import logging
import numpy as np
from smf.core.backend import NumpyBackend
from smf.core.operators import curvature_diffusion
from smf.config.params import DiffusionParams

_log = logging.getLogger(__name__)


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
    """Diffuse + decay psi. Returns (updated psi, amplitude).

    If curvature_diffusion produces NaN/Inf, falls back to pre-diffusion psi
    with a warning.
    """
    psi_before = psi  # keep reference for fallback
    diffused = curvature_diffusion(psi, R, knots, crystal, bk, dp)

    # NaN/Inf guard: fall back to undiffused psi if diffusion blew up
    if np.any(np.isnan(diffused)) or np.any(np.isinf(diffused)):
        _log.warning("NaN/Inf in curvature_diffusion output — falling back to previous psi")
        diffused = psi_before

    psi = (1 - dp.alpha) * psi + dp.alpha * diffused
    decay = dp.active_base_decay + dp.V_modulation_scale * V
    idle_factor = dp.idle_decay if idle else 1.0
    # Compute full multiplier first, then apply — matches original operation order
    psi *= decay * idle_factor
    amp = bk.abs(psi)
    return psi, amp
