"""Backend abstraction for tensor operations.

All physics code calls backend methods — never raw numpy.
Fixes: curvature_diffusion raw np.* calls moved to backend methods.
"""
from __future__ import annotations
import numpy as np


class NumpyBackend:
    """NumPy tensor backend. All tensor ops go through here."""

    name = "numpy"

    def zeros_float(self, shape):
        return np.zeros(shape, dtype=np.float32)

    def ones_float(self, shape):
        return np.ones(shape, dtype=np.float32)

    def zeros_complex(self, shape):
        return np.zeros(shape, dtype=np.complex64)

    def array_float(self, x):
        return np.asarray(x, dtype=np.float32)

    def array_complex(self, x):
        return np.asarray(x, dtype=np.complex64)

    def roll(self, x, shift, axis):
        return np.roll(x, shift, axis)

    def clamp(self, x, lo, hi):
        return np.clip(x, lo, hi)

    def abs(self, x):
        return np.abs(x)

    def sqrt(self, x):
        return np.sqrt(x)

    def maximum(self, a, b):
        return np.maximum(a, b)

    def expand_dim(self, x, axis):
        return np.expand_dims(x, axis)

    def mean(self, x, axis=None):
        return np.mean(x, axis=axis)

    def angle(self, x):
        return np.angle(x)

    def sin(self, x):
        return np.sin(x)

    def exp_complex(self, phase):
        """exp(1j * phase) — returns complex64."""
        return np.exp(1j * phase).astype(np.complex64)

    def to_numpy(self, x):
        return np.asarray(x)

    def laplacian_2d(self, x, center_weight):
        """5-point discrete Laplacian on axes 0, 1."""
        return (
            self.roll(x, -1, 0) + self.roll(x, 1, 0)
            + self.roll(x, -1, 1) + self.roll(x, 1, 1)
            + center_weight * x
        )

    def gradient_magnitude(self, x, epsilon):
        """Gradient magnitude via forward differences on axes 0, 1."""
        dy = self.roll(x, -1, 0) - x
        dx = self.roll(x, -1, 1) - x
        return self.sqrt(dy * dy + dx * dx + epsilon)

    def curvature_diffusion(
        self, wave, curvature, knots, crystal,
        crystal_block_strength,
        knot_resistance,
        conductivity_epsilon,
        scatter_curvature_offset,
        scatter_strength,
        crystal_wave_trap,
    ):
        """Geodesic holographic diffusion — curvature-aware wave transport.

        All numpy calls go through self.* methods.
        """
        crystal_block = 1.0 / (1.0 + crystal * crystal_block_strength)
        conductivity = curvature * crystal_block / (1.0 + knots * knot_resistance)

        up = self.roll(wave, -1, 0) * self.roll(conductivity, -1, 0)
        down = self.roll(wave, 1, 0) * self.roll(conductivity, 1, 0)
        left = self.roll(wave, -1, 1) * self.roll(conductivity, -1, 1)
        right = self.roll(wave, 1, 1) * self.roll(conductivity, 1, 1)
        csum = (
            self.roll(conductivity, -1, 0) + self.roll(conductivity, 1, 0)
            + self.roll(conductivity, -1, 1) + self.roll(conductivity, 1, 1)
            + conductivity_epsilon
        )
        flow = (up + down + left + right) / csum

        energy = self.expand_dim(self.mean(self.abs(wave), axis=-1), -1)
        carved = self.clamp(curvature - scatter_curvature_offset, 0, None)
        scatter = (
            self.array_complex(energy)
            * self.array_complex(carved)
            * scatter_strength
        )
        return flow + scatter + wave * crystal * crystal_wave_trap
