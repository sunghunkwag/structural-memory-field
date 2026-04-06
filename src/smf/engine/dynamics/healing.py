"""Void gravity — healing toward V=1."""
from __future__ import annotations
import numpy as np
from smf.core.backend import NumpyBackend
from smf.config.params import HealingParams, BackendParams


def apply_healing(
    V: np.ndarray,
    knots: np.ndarray,
    crystal: np.ndarray,
    bk: NumpyBackend,
    hp: HealingParams,
    bp: BackendParams,
) -> np.ndarray:
    """Heal void field. Returns updated V (not clamped)."""
    lapV = bk.laplacian_2d(V, bp.laplacian_center_weight)
    recovery = hp.recovery_strength * (1.0 - V)
    divisor = 1 + knots * hp.knot_resistance + crystal * hp.crystal_resistance
    return V + hp.rate * (lapV + recovery) / divisor
