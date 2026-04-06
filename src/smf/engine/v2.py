"""V2 Engine — physics-native emergent intelligence.

Orchestrates dynamics modules with zero magic numbers in step().
All numeric constants come from EngineConfig.
"""
from __future__ import annotations
import warnings
import numpy as np

from smf.core.backend import NumpyBackend
from smf.core.fields import FieldState
from smf.config.params import EngineConfig
from smf.engine.base import prepare_input
from smf.engine.dynamics.wound import apply_wound, apply_phantom
from smf.engine.dynamics.injection import apply_injection
from smf.engine.dynamics.healing import apply_healing
from smf.engine.dynamics.echo import apply_echo
from smf.engine.dynamics.diffusion import apply_diffusion
from smf.engine.dynamics.stress import apply_stress_and_knots
from smf.engine.dynamics.crystallization import apply_crystallization
from smf.engine.dynamics.erosion import apply_erosion
from smf.engine.dynamics.curvature import apply_curvature
from smf.engine.dynamics.readout import compute_action_scores


class EngineV2:
    """V2 physics engine. All parameters via EngineConfig."""

    def __init__(self, cfg: EngineConfig | None = None, **kwargs):
        if cfg is not None:
            self.cfg = cfg
        elif kwargs:
            self.cfg = EngineConfig.from_legacy_kwargs(**kwargs)
        else:
            self.cfg = EngineConfig()
        self.H = self.cfg.H
        self.W = self.cfg.W
        self.K = self.cfg.K
        self.bk = NumpyBackend()
        self.num_actions = min(self.cfg.readout.num_actions, self.K)
        self._init_fields()

    def _init_fields(self):
        self.f = FieldState.create(self.H, self.W, self.K, self.bk)

    # --- Convenience accessors (match original API) ---
    @property
    def psi(self): return self.f.psi
    @psi.setter
    def psi(self, v): self.f.psi = v
    @property
    def V(self): return self.f.V
    @V.setter
    def V(self, v): self.f.V = v
    @property
    def R(self): return self.f.R
    @R.setter
    def R(self, v): self.f.R = v
    @property
    def knots(self): return self.f.knots
    @knots.setter
    def knots(self, v): self.f.knots = v
    @property
    def crystal(self): return self.f.crystal
    @crystal.setter
    def crystal(self, v): self.f.crystal = v
    @property
    def echo(self): return self.f.echo
    @echo.setter
    def echo(self, v): self.f.echo = v
    @property
    def stress(self): return self.f.stress
    @stress.setter
    def stress(self, v): self.f.stress = v
    @property
    def t(self): return self.f.t
    @t.setter
    def t(self, v): self.f.t = v

    def reset(self):
        self._init_fields()

    def get_diagnostics(self) -> dict:
        amp = np.abs(self.f.psi)
        return {
            't': self.f.t,
            'psi_mean': float(amp.mean()),
            'psi_max': float(amp.max()),
            'V_mean': float(self.f.V.mean()),
            'V_min': float(self.f.V.min()),
            'knots_total': float(self.f.knots.sum()),
            'knots_max': float(self.f.knots.max()),
            'crystal_total': float(self.f.crystal.sum()),
            'crystal_max': float(self.f.crystal.max()),
            'stress_mean': float(self.f.stress.mean()),
            'stress_max': float(self.f.stress.max()),
            'echo_energy': float(np.abs(self.f.echo).sum()),
            'R_mean': float(self.f.R.mean()),
            'R_max': float(self.f.R.max()),
            'field_energy': float((amp ** 2).sum()),
        }

    def get_state(self) -> dict:
        d = {
            'H': self.H, 'W': self.W, 'K': self.K,
            'config': self.cfg.to_dict(),
        }
        d.update(self.f.to_dict())
        return d

    def load_state(self, state: dict):
        if 'config' in state:
            saved_cfg = EngineConfig.from_dict(state['config'])
            if saved_cfg.to_dict() != self.cfg.to_dict():
                warnings.warn(
                    "Config mismatch between saved state and current engine. "
                    "Loaded state may behave differently.",
                    UserWarning,
                    stacklevel=2,
                )
        self.f.load_dict(state)

    def _compute_idle(self, input_level: float) -> float:
        return 1.0 if input_level < self.cfg.idle.threshold else 0.0

    def step(self, s_amp, s_ph=None) -> int:
        """Run one physics step. Returns action index."""
        bk = self.bk
        cfg = self.cfg

        s, s_ph_arr, n, ev, input_level = prepare_input(s_amp, s_ph, self.K, bk)

        V_before = self.f.V.copy()

        # 1. Wound + resonance cascade
        self.f.V = apply_wound(self.f.V, s, bk, cfg.wound)
        self.f.V = bk.clamp(self.f.V, 0, 1)

        idle = self._compute_idle(input_level)

        self.f.V = apply_phantom(self.f.V, bk, cfg.wound, cfg.backend)
        self.f.V = bk.clamp(self.f.V, 0, 1)

        # 2. Wave injection
        self.f.psi = apply_injection(
            self.f.psi, self.f.V, self.f.R, self.f.crystal, ev, bk, cfg.injection,
        )

        # 3. Healing
        self.f.V = apply_healing(
            self.f.V, self.f.knots, self.f.crystal, bk, cfg.healing, cfg.backend,
        )
        self.f.V = bk.clamp(self.f.V, 0, 1)

        # 4. Echo
        self.f.V, self.f.echo = apply_echo(
            self.f.V, V_before, self.f.echo, bk, cfg.echo,
        )
        self.f.V = bk.clamp(self.f.V, 0, 1)

        # 5. Diffusion
        self.f.psi, amp = apply_diffusion(
            self.f.psi, self.f.V, self.f.R, self.f.knots, self.f.crystal,
            idle, bk, cfg.diffusion,
        )

        # 6. Stress + knots
        self.f.stress, self.f.knots = apply_stress_and_knots(
            self.f.V, amp, self.f.knots, self.H, self.W, bk, cfg.stress, cfg.backend,
        )

        # 7. Crystallization + erosion
        self._crystallize_and_erode(idle, cfg)

        # 8. Curvature update
        self.f.R = apply_curvature(
            self.f.R, self.f.stress, amp, bk, cfg.curvature,
        )

        # Action readout
        scores = compute_action_scores(
            self.f.V, amp, self.f.knots, self.f.stress, self.f.crystal,
            self.num_actions, cfg.readout,
        )
        self.f.t += 1
        return int(np.argmax(scores))

    def _crystallize_and_erode(self, idle: float, cfg: EngineConfig):
        """Crystallization + erosion — overridden by V3 to add phase dynamics."""
        self.f.crystal = apply_crystallization(
            self.f.knots, self.f.crystal, self.bk, cfg.crystallization,
        )
        self.f.knots = apply_erosion(
            self.f.V, self.f.knots, idle, self.bk, cfg.erosion,
        )
