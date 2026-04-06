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
    """Compute per-action scores. Returns list of floats."""
    scores = []
    for c in range(num_actions):
        scores.append(float(
            ((1 - V[:, :, c])
             * amp[:, :, c]
             * (1 + knots[:, :, c] * rp.knot_amplification)
             * (stress + rp.stress_offset)
             * (1 + crystal[:, :, c] * rp.crystal_amplification)).sum()
        ))
    return scores
