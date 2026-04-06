"""Wave (psi) injection."""
from __future__ import annotations
import numpy as np
from smf.core.backend import NumpyBackend
from smf.config.params import InjectionParams


def apply_injection(
    psi: np.ndarray,
    V: np.ndarray,
    R: np.ndarray,
    crystal: np.ndarray,
    ev: np.ndarray,
    bk: NumpyBackend,
    ip: InjectionParams,
) -> np.ndarray:
    """Inject stimulus wave into psi field. Returns updated psi."""
    layers = min(ip.layers, psi.shape[0])  # clamp to field height
    for r in range(layers):
        rec = bk.maximum(
            V[r, :, :],
            bk.clamp(R[r, :, :] - ip.curvature_recall_offset,
                      0, ip.curvature_recall_max) * ip.curvature_recall_scale,
        )
        psi[r, :, :] += ev * (1 - r / layers) * rec * (1 - crystal[r, :, :] * ip.crystal_block)
    return psi
