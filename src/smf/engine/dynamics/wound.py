"""Wound injection and resonance cascade."""
from __future__ import annotations
import numpy as np
from smf.core.backend import NumpyBackend
from smf.config.params import WoundParams, BackendParams


def apply_wound(
    V: np.ndarray,
    stimulus: np.ndarray,
    bk: NumpyBackend,
    wp: WoundParams,
) -> np.ndarray:
    """Inject wound into V field. Returns updated V (not clamped)."""
    w = bk.array_float(stimulus) * wp.depth
    layers = min(wp.layers, V.shape[0])  # clamp to field height
    for r in range(layers):
        V[r, :, :] -= w * (1 - r / layers)
    return V


def apply_phantom(
    V: np.ndarray,
    bk: NumpyBackend,
    wp: WoundParams,
    bp: BackendParams,
) -> np.ndarray:
    """Resonance cascade — phantom wounds from wounded neighbours."""
    wf = bk.maximum(wp.phantom_threshold - V, 0)
    lap = bk.laplacian_2d(wf, bp.laplacian_center_weight)
    phantom = bk.maximum(lap, 0) * V * wp.resonance_range
    return V - phantom
