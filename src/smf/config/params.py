"""All physics parameters extracted from original code into config dataclasses.

Every numeric literal from v2/v3 step() is here with its original value as default.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class WoundParams:
    """Wound injection parameters."""
    depth: float = 0.35
    layers: int = 4
    phantom_threshold: float = 0.7
    resonance_range: float = 0.5


@dataclass
class IdleParams:
    """Idle detection parameters."""
    threshold: float = 1e-6


@dataclass
class InjectionParams:
    """Wave injection parameters."""
    layers: int = 4
    curvature_recall_offset: float = 1.0
    curvature_recall_max: float = 5.0
    curvature_recall_scale: float = 0.3
    crystal_block: float = 0.8


@dataclass
class HealingParams:
    """Void gravity / healing parameters."""
    rate: float = 0.15
    recovery_strength: float = 0.3
    knot_resistance: float = 5.0
    crystal_resistance: float = 10.0


@dataclass
class EchoParams:
    """Wound echo (post-healing oscillation) parameters."""
    damping: float = 0.9
    coupling: float = 0.05


@dataclass
class DiffusionParams:
    """Geodesic holographic diffusion parameters."""
    alpha: float = 0.25
    crystal_block_strength: float = 20.0
    knot_resistance: float = 3.0
    conductivity_epsilon: float = 1e-8
    scatter_curvature_offset: float = 1.1
    scatter_strength: float = 0.15
    crystal_wave_trap: float = 0.3
    active_base_decay: float = 0.95
    V_modulation_scale: float = 0.04
    idle_decay: float = 0.90


@dataclass
class StressParams:
    """Stress and knot formation parameters."""
    gradient_amp_weight: float = 0.5
    diffusion_strength: float = 0.05
    stress_max: float = 100.0
    knot_threshold: float = 1.2
    knot_wound_threshold: float = 0.8
    knot_formation_rate: float = 0.1
    knot_max: float = 10.0


@dataclass
class CrystallizationParams:
    """Crystallization parameters."""
    threshold: float = 0.1
    growth_rate: float = 0.1
    decay_rate: float = 0.998
    forming_threshold: float = 0.001  # v3 phase imprint threshold


@dataclass
class ErosionParams:
    """Scar erosion parameters."""
    rate: float = 0.1
    eligibility_threshold: float = 0.9
    idle_rate: float = 0.02


@dataclass
class CurvatureParams:
    """Ricci curvature update parameters."""
    plasticity: float = 0.08
    relaxation_rate: float = 0.005
    R_min: float = 0.1
    R_max: float = 10.0


@dataclass
class ReadoutParams:
    """Action readout parameters."""
    knot_amplification: float = 3.0
    stress_offset: float = 0.01
    crystal_amplification: float = 5.0


@dataclass
class BackendParams:
    """Backend / numerical parameters."""
    gradient_epsilon: float = 1e-12
    laplacian_center_weight: float = -4.0


@dataclass
class PhaseParams:
    """V3 crystal phase dynamics parameters."""
    fracture_threshold: float = 0.6
    fracture_energy_release: float = 0.3
    fracture_decay: float = 0.5
    fracture_shatter_multiplier: float = 2.0
    fusion_rate: float = 0.05
    fusion_phase_tolerance: float = 0.1
    nucleation_rate: float = 0.1
    nucleation_viability_threshold: float = 0.3


@dataclass
class EngineConfig:
    """Top-level configuration holding all parameter groups."""
    H: int = 32
    W: int = 32
    K: int = 8
    wound: WoundParams = field(default_factory=WoundParams)
    idle: IdleParams = field(default_factory=IdleParams)
    injection: InjectionParams = field(default_factory=InjectionParams)
    healing: HealingParams = field(default_factory=HealingParams)
    echo: EchoParams = field(default_factory=EchoParams)
    diffusion: DiffusionParams = field(default_factory=DiffusionParams)
    stress: StressParams = field(default_factory=StressParams)
    crystallization: CrystallizationParams = field(default_factory=CrystallizationParams)
    erosion: ErosionParams = field(default_factory=ErosionParams)
    curvature: CurvatureParams = field(default_factory=CurvatureParams)
    readout: ReadoutParams = field(default_factory=ReadoutParams)
    backend: BackendParams = field(default_factory=BackendParams)
    phase: PhaseParams = field(default_factory=PhaseParams)

    def to_dict(self) -> dict:
        """Serialize to nested dict for state saving."""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> EngineConfig:
        """Reconstruct from dict."""
        from dataclasses import fields as dc_fields
        import typing
        type_map = typing.get_type_hints(cls)
        kw = {}
        for f in dc_fields(cls):
            if f.name not in d:
                continue
            v = d[f.name]
            real_type = type_map.get(f.name, f.type)
            if isinstance(v, dict) and isinstance(real_type, type):
                kw[f.name] = real_type(**v)
            else:
                kw[f.name] = v
        return cls(**kw)

    @staticmethod
    def from_legacy_kwargs(**kwargs) -> EngineConfig:
        """Build config from flat v2/v3 keyword arguments."""
        cfg = EngineConfig(
            H=kwargs.pop("H", 32),
            W=kwargs.pop("W", 32),
            K=kwargs.pop("K", 8),
        )
        _FLAT_MAP = {
            "alpha": ("diffusion", "alpha"),
            "plasticity": ("curvature", "plasticity"),
            "healing_rate": ("healing", "rate"),
            "wound_depth": ("wound", "depth"),
            "knot_threshold": ("stress", "knot_threshold"),
            "crystal_threshold": ("crystallization", "threshold"),
            "erosion_rate": ("erosion", "rate"),
            "echo_damping": ("echo", "damping"),
            "echo_coupling": ("echo", "coupling"),
            "resonance_range": ("wound", "resonance_range"),
            "fracture_threshold": ("phase", "fracture_threshold"),
            "fracture_energy_release": ("phase", "fracture_energy_release"),
            "fracture_decay": ("phase", "fracture_decay"),
            "fusion_rate": ("phase", "fusion_rate"),
            "fusion_phase_tolerance": ("phase", "fusion_phase_tolerance"),
            "nucleation_rate": ("phase", "nucleation_rate"),
        }
        for k, v in kwargs.items():
            if k in _FLAT_MAP:
                group_name, param_name = _FLAT_MAP[k]
                setattr(getattr(cfg, group_name), param_name, v)
        return cfg
