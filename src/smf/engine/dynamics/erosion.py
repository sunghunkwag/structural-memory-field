"""Scar erosion — competitive forgetting."""
from __future__ import annotations
import numpy as np
from smf.core.backend import NumpyBackend
from smf.config.params import ErosionParams


def apply_erosion(
    V: np.ndarray,
    knots: np.ndarray,
    idle: float,
    bk: NumpyBackend,
    ep: ErosionParams,
) -> np.ndarray:
    """Erode knots in healthy regions and during idle. Returns updated knots."""
    elig = bk.maximum(V - ep.eligibility_threshold, 0)
    knots = bk.maximum(knots - ep.rate * elig * knots, 0)
    knots = bk.maximum(knots - (ep.idle_rate * idle) * knots, 0)
    return knots
