"""V3 Engine — extends V2 with crystal phase dynamics.

Inherits from EngineV2 — ZERO duplicated physics code.
Only overrides _init_fields() and _crystallize_and_erode().
"""
from __future__ import annotations
import numpy as np

from smf.config.params import EngineConfig
from smf.engine.v2 import EngineV2
from smf.engine.dynamics.crystallization import apply_crystallization_v3
from smf.engine.dynamics.erosion import apply_erosion
from smf.engine.dynamics.phase import compute_boundary_tension, apply_phase_dynamics


class EngineV3(EngineV2):
    """V3 physics engine — V2 + crystal phase dynamics."""

    def _init_fields(self):
        from smf.core.fields import FieldState
        self.f = FieldState.create(self.H, self.W, self.K, self.bk, v3=True)

    # --- Extra accessors for v3 fields ---
    @property
    def crystal_phase(self): return self.f.crystal_phase
    @crystal_phase.setter
    def crystal_phase(self, v): self.f.crystal_phase = v
    @property
    def boundary(self): return self.f.boundary
    @boundary.setter
    def boundary(self, v): self.f.boundary = v

    def get_diagnostics(self) -> dict:
        d = super().get_diagnostics()
        d['crystal_phase_std'] = float(self.f.crystal_phase.std())
        d['boundary_max'] = float(self.f.boundary.max())
        d['boundary_mean'] = float(self.f.boundary.mean())
        return d

    def _crystallize_and_erode(self, idle: float, cfg: EngineConfig):
        """V3: crystallization with phase imprinting + phase dynamics + erosion."""
        bk = self.bk

        # Crystallization with phase imprinting
        new_crystal, self.f.crystal, self.f.crystal_phase = apply_crystallization_v3(
            self.f.knots, self.f.crystal, self.f.crystal_phase,
            self.f.psi, bk, cfg.crystallization,
        )

        # Boundary tension
        self.f.boundary, neighbor_crystal = compute_boundary_tension(
            self.f.crystal, self.f.crystal_phase, bk, cfg.phase,
        )

        # Phase dynamics: fracture, fusion, nucleation
        self.f.crystal, self.f.knots, self.f.psi = apply_phase_dynamics(
            self.f.crystal, self.f.crystal_phase, self.f.knots,
            self.f.psi, self.f.V, self.f.boundary, neighbor_crystal,
            bk, cfg.phase,
        )

        # Erosion
        self.f.knots = apply_erosion(
            self.f.V, self.f.knots, idle, bk, cfg.erosion,
        )
