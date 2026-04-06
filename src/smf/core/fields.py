"""FieldState — container for all physics field arrays."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from smf.core.backend import NumpyBackend


@dataclass
class FieldState:
    """Holds all mutable field arrays for the engine."""
    psi: np.ndarray       # (H, W, K) complex64
    V: np.ndarray          # (H, W, K) float32
    R: np.ndarray          # (H, W, K) float32
    knots: np.ndarray      # (H, W, K) float32
    crystal: np.ndarray    # (H, W, K) float32
    echo: np.ndarray       # (H, W, K) float32
    stress: np.ndarray     # (H, W) float32
    t: int
    # v3 fields
    crystal_phase: Optional[np.ndarray] = None  # (H, W, K) float32
    boundary: Optional[np.ndarray] = None       # (H, W, K) float32

    @staticmethod
    def create(H: int, W: int, K: int, bk: NumpyBackend, v3: bool = False) -> FieldState:
        """Create resting-state fields."""
        fs = FieldState(
            psi=bk.zeros_complex((H, W, K)),
            V=bk.ones_float((H, W, K)),
            R=bk.ones_float((H, W, K)),
            knots=bk.zeros_float((H, W, K)),
            crystal=bk.zeros_float((H, W, K)),
            echo=bk.zeros_float((H, W, K)),
            stress=bk.zeros_float((H, W)),
            t=0,
        )
        if v3:
            fs.crystal_phase = bk.zeros_float((H, W, K))
            fs.boundary = bk.zeros_float((H, W, K))
        return fs

    def snapshot(self) -> FieldState:
        """Deep copy."""
        return FieldState(
            psi=self.psi.copy(), V=self.V.copy(), R=self.R.copy(),
            knots=self.knots.copy(), crystal=self.crystal.copy(),
            echo=self.echo.copy(), stress=self.stress.copy(), t=self.t,
            crystal_phase=self.crystal_phase.copy() if self.crystal_phase is not None else None,
            boundary=self.boundary.copy() if self.boundary is not None else None,
        )

    def to_dict(self) -> dict:
        """Serialize to dict of arrays."""
        d = {
            "t": self.t,
            "psi": self.psi.copy(), "V": self.V.copy(), "R": self.R.copy(),
            "knots": self.knots.copy(), "crystal": self.crystal.copy(),
            "echo": self.echo.copy(), "stress": self.stress.copy(),
        }
        if self.crystal_phase is not None:
            d["crystal_phase"] = self.crystal_phase.copy()
        if self.boundary is not None:
            d["boundary"] = self.boundary.copy()
        return d

    def load_dict(self, d: dict) -> None:
        """Load from dict of arrays."""
        self.t = d["t"]
        self.psi = d["psi"].copy()
        self.V = d["V"].copy()
        self.R = d["R"].copy()
        self.knots = d["knots"].copy()
        self.crystal = d["crystal"].copy()
        self.echo = d["echo"].copy()
        self.stress = d["stress"].copy()
        if "crystal_phase" in d:
            self.crystal_phase = d["crystal_phase"].copy()
        if "boundary" in d:
            self.boundary = d["boundary"].copy()

    def validate(self) -> None:
        """Check for NaN/inf. Raises ValueError."""
        for name in ("psi", "V", "R", "knots", "crystal", "echo", "stress"):
            arr = getattr(self, name)
            if np.any(np.isnan(arr)):
                raise ValueError(f"NaN in field '{name}' at t={self.t}")
            if np.any(np.isinf(arr)):
                raise ValueError(f"Inf in field '{name}' at t={self.t}")
