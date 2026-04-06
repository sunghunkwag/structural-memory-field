"""Input encoders — convert raw data to engine stimulus format."""
from __future__ import annotations
import numpy as np


def encode_text_char(char: str, K: int = 8) -> np.ndarray:
    """Encode a character as stimulus amplitude vector."""
    code = ord(char) % K
    amp = np.zeros(K)
    amp[code] = 1.0
    return amp


def encode_onehot(index: int, K: int = 8) -> np.ndarray:
    """One-hot encode an integer index."""
    amp = np.zeros(K)
    amp[index % K] = 1.0
    return amp
