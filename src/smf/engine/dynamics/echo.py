"""Wound echo — post-healing oscillation."""
from __future__ import annotations
import numpy as np
from smf.core.backend import NumpyBackend
from smf.config.params import EchoParams


def apply_echo(
    V: np.ndarray,
    V_before: np.ndarray,
    echo: np.ndarray,
    bk: NumpyBackend,
    ep: EchoParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply echo feedback. Returns (updated V not clamped, updated echo)."""
    delta = V - V_before
    echo = echo * ep.damping + delta * (1 - ep.damping)
    V = V + echo * ep.coupling
    return V, echo
