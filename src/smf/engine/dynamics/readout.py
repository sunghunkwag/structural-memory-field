"""Action readout from field state."""
from __future__ import annotations
import numpy as np
from smf.config.params import ReadoutParams


def compute_action_scores(
    V: np.ndarray,
    amp: np.ndarray,
    knots: np.ndarray,
    stress: np.ndarray,
    crystal: np.ndarray,
    num_actions: int,
    rp: ReadoutParams,
) -> list[float]:
    """Compute per-action scores via vectorized broadcast.

    Formula per channel c:
      score_c = sum((1-V[:,:,c]) * amp[:,:,c] * (1+knots[:,:,c]*knot_amp)
                     * (stress+offset) * (1+crystal[:,:,c]*crystal_amp))

    Vectorized over all channels simultaneously. NaN/Inf replaced with 0.
    """
    n = num_actions
    # Slice [:,:,:n] for all active channels at once
    V_n = V[:, :, :n]
    amp_n = amp[:, :, :n]
    knots_n = knots[:, :, :n]
    crystal_n = crystal[:, :, :n]

    # stress is (H, W) — expand to (H, W, 1) for broadcast
    stress_3d = stress[:, :, np.newaxis]

    # Vectorized: (H, W, n) element-wise product, then sum over (H, W) → (n,)
    raw = (
        (1 - V_n)
        * amp_n
        * (1 + knots_n * rp.knot_amplification)
        * (stress_3d + rp.stress_offset)
        * (1 + crystal_n * rp.crystal_amplification)
    ).sum(axis=(0, 1))  # sum over spatial dims → shape (n,)

    # Replace NaN/Inf with 0
    raw = np.where(np.isfinite(raw), raw, 0.0)

    return [float(raw[c]) for c in range(n)]
