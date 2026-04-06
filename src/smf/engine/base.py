"""Input preparation and validation shared by v2 and v3 engines."""
from __future__ import annotations
import numpy as np
from smf.core.backend import NumpyBackend
from smf.config.params import EngineConfig


def validate_input(s_amp: np.ndarray, s_ph: np.ndarray | None, K: int):
    """Validate stimulus arrays. Raises ValueError on NaN/inf/shape issues."""
    if np.any(np.isnan(s_amp)):
        raise ValueError("NaN detected in s_amp input")
    if np.any(np.isinf(s_amp)):
        raise ValueError("inf detected in s_amp input")
    if s_ph is not None:
        if np.any(np.isnan(s_ph)):
            raise ValueError("NaN detected in s_ph input")
        if np.any(np.isinf(s_ph)):
            raise ValueError("inf detected in s_ph input")


def prepare_input(s_amp, s_ph, K: int, bk: NumpyBackend):
    """Prepare stimulus arrays. Returns (s, s_ph_arr, n, ev, input_level).

    Reproduces the exact dtype dance of the original code for backward compat:
    - s_amp cast to float64
    - s padded to K with float64 zeros
    - w = bk.array_float(s) * depth produces float64 (float32 * Python float)
    - ev = bk.array_complex(s_amp[:n] * exp(1j * s_ph[:n])) → complex64
    """
    s_amp = np.asarray(s_amp, dtype=np.float64)
    if s_ph is None:
        s_ph = np.zeros_like(s_amp)
    else:
        s_ph = np.asarray(s_ph, dtype=np.float64)

    validate_input(s_amp, s_ph, K)

    s = np.zeros(K)
    n = min(len(s_amp), K)
    s[:n] = s_amp[:n]

    ev = bk.array_complex(s_amp[:n] * np.exp(1j * s_ph[:n]))
    input_level = float(np.abs(s_amp[:n]).sum())

    return s, s_ph, n, ev, input_level
