"""Input encoders — convert raw data to engine stimulus format."""
from __future__ import annotations
import numpy as np


def encode_onehot(index: int, K: int = 8) -> np.ndarray:
    """One-hot encode an integer index."""
    amp = np.zeros(K)
    amp[index % K] = 1.0
    return amp


class TextEncoder:
    """Encode text strings into (amplitude, phase) stimulus vectors.

    Uses character n-gram frequency distribution for amplitude and
    positional character hashing for phase angles.
    """

    def __init__(self, K: int = 8):
        self.K = K

    def encode(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Encode text into (amplitude, phase) arrays of shape (K,).

        Different texts produce different (amp, phase) pairs.
        Similar texts produce more similar vectors than dissimilar texts.
        Multiple channels are active (not one-hot).
        Phase carries positional information.
        Deterministic.
        """
        K = self.K
        amp = np.zeros(K, dtype=np.float64)
        phase = np.zeros(K, dtype=np.float64)

        if not text:
            return amp, phase

        # Amplitude: distribute character energy across channels via n-grams.
        # Each character contributes to its primary channel (ord % K) and
        # its bigram channel ((ord_prev + ord_cur) % K), spreading energy.
        for i, ch in enumerate(text):
            c = ord(ch)
            # Primary channel: character identity
            primary = c % K
            amp[primary] += 1.0
            # Bigram channel: context-sensitive spread
            if i > 0:
                bigram = (ord(text[i - 1]) + c) % K
                amp[bigram] += 0.5
            # Trigram channel: longer context
            if i > 1:
                trigram = (ord(text[i - 2]) * 31 + ord(text[i - 1]) * 7 + c) % K
                amp[trigram] += 0.25

        # Normalize so total energy is consistent (proportional to sqrt(len))
        total = np.sqrt(np.sum(amp * amp))
        if total > 0:
            amp = amp / total * np.sqrt(len(text))

        # Phase: positional hashing. Each character contributes a
        # position-weighted phase offset to its channel.
        phase_acc = np.zeros(K, dtype=np.float64)
        phase_count = np.zeros(K, dtype=np.float64)
        for i, ch in enumerate(text):
            c = ord(ch)
            chan = c % K
            # Phase from character value and position
            angle = (c * 2.3561 + i * 1.5708) % (2 * np.pi) - np.pi
            phase_acc[chan] += angle
            phase_count[chan] += 1.0

            # Bigram phase contribution
            if i > 0:
                bchan = (ord(text[i - 1]) + c) % K
                bangle = ((ord(text[i - 1]) ^ c) * 0.7854 + i * 0.3927) % (2 * np.pi) - np.pi
                phase_acc[bchan] += bangle
                phase_count[bchan] += 1.0

        # Average phase per channel
        for k in range(K):
            if phase_count[k] > 0:
                phase[k] = phase_acc[k] / phase_count[k]

        return amp, phase
