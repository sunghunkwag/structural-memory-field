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
    """Compute per-action scores. Returns list of floats.

    If all scores are zero or contain NaN, returns uniform scores
    so argmax picks action 0 deterministically.
    """
    scores = []
    for c in range(num_actions):
        val = float(
            ((1 - V[:, :, c])
             * amp[:, :, c]
             * (1 + knots[:, :, c] * rp.knot_amplification)
             * (stress + rp.stress_offset)
             * (1 + crystal[:, :, c] * rp.crystal_amplification)).sum()
        )
        # Replace NaN/Inf with 0
        if np.isnan(val) or np.isinf(val):
            val = 0.0
        scores.append(val)
    return scores
