"""All physics parameters extracted from original code into config dataclasses.

Every numeric literal from v2/v3 step() is here with its original value as default.
Includes runtime validation in __post_init__ for all dataclasses.
"""
from __future__ import annotations
from dataclasses import dataclass, field


def _check_positive(name: str, val: float | int) -> None:
    if val <= 0:
        raise ValueError(f"{name} must be positive, got {val}")


def _check_non_negative(name: str, val: float | int) -> None:
    if val < 0:
        raise ValueError(f"{name} must be non-negative, got {val}")


def _check_range(name: str, val: float, lo: float, hi: float) -> None:
    if not (lo <= val <= hi):
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {val}")


@dataclass
class WoundParams:
    """Wound injection parameters."""
    depth: float = 0.35       # wound magnitude multiplier (>0)
    layers: int = 4           # spatial rows affected by wound (>0)
    phantom_threshold: float = 0.7   # V below this generates phantom wound
    resonance_range: float = 0.5     # phantom wound coupling strength (>=0)

    def __post_init__(self) -> None:
        _check_positive("wound.depth", self.depth)
        _check_positive("wound.layers", self.layers)
        _check_positive("wound.phantom_threshold", self.phantom_threshold)
        _check_non_negative("wound.resonance_range", self.resonance_range)


@dataclass
class IdleParams:
    """Idle detection parameters."""
    threshold: float = 1e-6   # input below this → idle (>0)

    def __post_init__(self) -> None:
        _check_positive("idle.threshold", self.threshold)


@dataclass
class InjectionParams:
    """Wave injection parameters."""
    layers: int = 4                      # spatial rows for injection (>0)
    curvature_recall_offset: float = 1.0 # R offset for recall (>=0)
    curvature_recall_max: float = 5.0    # max curvature recall (>0)
    curvature_recall_scale: float = 0.3  # curvature recall multiplier (>=0)
    crystal_block: float = 0.8           # crystal blocking factor [0,1]

    def __post_init__(self) -> None:
        _check_positive("injection.layers", self.layers)
        _check_non_negative("injection.curvature_recall_offset", self.curvature_recall_offset)
        _check_positive("injection.curvature_recall_max", self.curvature_recall_max)
        _check_non_negative("injection.curvature_recall_scale", self.curvature_recall_scale)
        _check_range("injection.crystal_block", self.crystal_block, 0.0, 1.0)


@dataclass
class HealingParams:
    """Void gravity / healing parameters."""
    rate: float = 0.15             # base healing rate (>0)
    recovery_strength: float = 0.3 # recovery toward V=1 strength (>=0)
    knot_resistance: float = 5.0   # knot healing resistance (>=0)
    crystal_resistance: float = 10.0  # crystal healing resistance (>=0)

    def __post_init__(self) -> None:
        _check_positive("healing.rate", self.rate)
        _check_non_negative("healing.recovery_strength", self.recovery_strength)
        _check_non_negative("healing.knot_resistance", self.knot_resistance)
        _check_non_negative("healing.crystal_resistance", self.crystal_resistance)


@dataclass
class EchoParams:
    """Wound echo (post-healing oscillation) parameters."""
    damping: float = 0.9    # EMA decay, [0,1]
    coupling: float = 0.05  # echo feedback strength (>=0)

    def __post_init__(self) -> None:
        _check_range("echo.damping", self.damping, 0.0, 1.0)
        _check_non_negative("echo.coupling", self.coupling)


@dataclass
class DiffusionParams:
    """Geodesic holographic diffusion parameters."""
    alpha: float = 0.25                    # mixing ratio [0,1]
    crystal_block_strength: float = 20.0   # crystal blocking (>0)
    knot_resistance: float = 3.0           # knot resistance (>=0)
    conductivity_epsilon: float = 1e-8     # division safety (>0)
    scatter_curvature_offset: float = 1.1  # scatter threshold (>0)
    scatter_strength: float = 0.15         # scatter magnitude (>=0)
    crystal_wave_trap: float = 0.3         # wave trapping (>=0)
    active_base_decay: float = 0.95        # wave decay base [0,1]
    V_modulation_scale: float = 0.04       # V-dependent modulation (>=0)
    idle_decay: float = 0.90               # idle attenuation [0,1]

    def __post_init__(self) -> None:
        _check_range("diffusion.alpha", self.alpha, 0.0, 1.0)
        _check_positive("diffusion.crystal_block_strength", self.crystal_block_strength)
        _check_non_negative("diffusion.knot_resistance", self.knot_resistance)
        _check_positive("diffusion.conductivity_epsilon", self.conductivity_epsilon)
        _check_positive("diffusion.scatter_curvature_offset", self.scatter_curvature_offset)
        _check_non_negative("diffusion.scatter_strength", self.scatter_strength)
        _check_non_negative("diffusion.crystal_wave_trap", self.crystal_wave_trap)
        _check_range("diffusion.active_base_decay", self.active_base_decay, 0.0, 1.0)
        _check_non_negative("diffusion.V_modulation_scale", self.V_modulation_scale)
        _check_range("diffusion.idle_decay", self.idle_decay, 0.0, 1.0)


@dataclass
class StressParams:
    """Stress and knot formation parameters."""
    gradient_amp_weight: float = 0.5    # amplitude gradient weight (>=0)
    diffusion_strength: float = 0.05    # stress diffusion (>=0)
    stress_max: float = 100.0           # stress clamp ceiling (>0)
    knot_threshold: float = 1.2         # stress for knot formation (>0)
    knot_wound_threshold: float = 0.8   # V for knot formation (>0)
    knot_formation_rate: float = 0.1    # knot growth rate (>=0)
    knot_max: float = 10.0              # knot clamp ceiling (>0)

    def __post_init__(self) -> None:
        _check_non_negative("stress.gradient_amp_weight", self.gradient_amp_weight)
        _check_non_negative("stress.diffusion_strength", self.diffusion_strength)
        _check_positive("stress.stress_max", self.stress_max)
        _check_positive("stress.knot_threshold", self.knot_threshold)
        _check_positive("stress.knot_wound_threshold", self.knot_wound_threshold)
        _check_non_negative("stress.knot_formation_rate", self.knot_formation_rate)
        _check_positive("stress.knot_max", self.knot_max)


@dataclass
class CrystallizationParams:
    """Crystallization parameters."""
    threshold: float = 0.1         # knot density for nucleation (>=0)
    growth_rate: float = 0.1       # crystal growth rate (>=0)
    decay_rate: float = 0.998      # per-step retention [0,1]
    forming_threshold: float = 0.001  # v3 phase imprint threshold (>=0)

    def __post_init__(self) -> None:
        _check_non_negative("crystallization.threshold", self.threshold)
        _check_non_negative("crystallization.growth_rate", self.growth_rate)
        _check_range("crystallization.decay_rate", self.decay_rate, 0.0, 1.0)
        _check_non_negative("crystallization.forming_threshold", self.forming_threshold)


@dataclass
class ErosionParams:
    """Scar erosion parameters."""
    rate: float = 0.1                   # erosion rate (>=0)
    eligibility_threshold: float = 0.9  # V above this enables erosion (>=0)
    idle_rate: float = 0.02             # idle erosion rate (>=0)

    def __post_init__(self) -> None:
        _check_non_negative("erosion.rate", self.rate)
        _check_non_negative("erosion.eligibility_threshold", self.eligibility_threshold)
        _check_non_negative("erosion.idle_rate", self.idle_rate)


@dataclass
class CurvatureParams:
    """Ricci curvature update parameters."""
    plasticity: float = 0.08      # curvature growth rate (>=0)
    relaxation_rate: float = 0.005  # relaxation toward R=1 (>=0)
    R_min: float = 0.1            # curvature floor (>0)
    R_max: float = 10.0           # curvature ceiling (>R_min)

    def __post_init__(self) -> None:
        _check_non_negative("curvature.plasticity", self.plasticity)
        _check_non_negative("curvature.relaxation_rate", self.relaxation_rate)
        _check_positive("curvature.R_min", self.R_min)
        _check_positive("curvature.R_max", self.R_max)
        if self.R_max <= self.R_min:
            raise ValueError(
                f"curvature.R_max ({self.R_max}) must be > curvature.R_min ({self.R_min})"
            )


@dataclass
class ReadoutParams:
    """Action readout parameters."""
    num_actions: int = 4               # number of output actions (>0)
    knot_amplification: float = 3.0    # knot contribution (>=0)
    stress_offset: float = 0.01        # prevents zero stress (>=0)
    crystal_amplification: float = 5.0 # crystal contribution (>=0)

    def __post_init__(self) -> None:
        _check_positive("readout.num_actions", self.num_actions)
        _check_non_negative("readout.knot_amplification", self.knot_amplification)
        _check_non_negative("readout.stress_offset", self.stress_offset)
        _check_non_negative("readout.crystal_amplification", self.crystal_amplification)


@dataclass
class BackendParams:
    """Backend / numerical parameters."""
    gradient_epsilon: float = 1e-12     # sqrt safety (>0)
    laplacian_center_weight: float = -4.0  # 5-point stencil center (<0)

    def __post_init__(self) -> None:
        _check_positive("backend.gradient_epsilon", self.gradient_epsilon)
        if self.laplacian_center_weight >= 0:
            raise ValueError(
                f"backend.laplacian_center_weight must be negative, "
                f"got {self.laplacian_center_weight}"
            )


@dataclass
class PhaseParams:
    """V3 crystal phase dynamics parameters."""
    neighbor_count: float = 4.0                  # neighbor averaging divisor (>0)
    fracture_threshold: float = 0.6              # tension for fracture (>0)
    fracture_energy_release: float = 0.3         # wave burst on fracture (>=0)
    fracture_decay: float = 0.5                  # crystal decay on fracture (>=0)
    fracture_shatter_multiplier: float = 2.0     # knot creation on fracture (>=0)
    fusion_rate: float = 0.05                    # crystal fusion rate (>=0)
    fusion_phase_tolerance: float = 0.1          # max tension for fusion (>=0)
    nucleation_rate: float = 0.1                 # boundary nucleation rate (>=0)
    nucleation_viability_threshold: float = 0.3  # V floor for nucleation (>=0)

    def __post_init__(self) -> None:
        _check_positive("phase.neighbor_count", self.neighbor_count)
        _check_positive("phase.fracture_threshold", self.fracture_threshold)
        _check_non_negative("phase.fracture_energy_release", self.fracture_energy_release)
        _check_non_negative("phase.fracture_decay", self.fracture_decay)
        _check_non_negative("phase.fracture_shatter_multiplier", self.fracture_shatter_multiplier)
        _check_non_negative("phase.fusion_rate", self.fusion_rate)
        _check_non_negative("phase.fusion_phase_tolerance", self.fusion_phase_tolerance)
        _check_non_negative("phase.nucleation_rate", self.nucleation_rate)
        _check_non_negative("phase.nucleation_viability_threshold",
                            self.nucleation_viability_threshold)


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

    debug_validate_interval: int = 0  # validate fields every N steps (0=disabled)

    def __post_init__(self) -> None:
        _check_positive("H", self.H)
        _check_positive("W", self.W)
        _check_positive("K", self.K)
        _check_non_negative("debug_validate_interval", self.debug_validate_interval)

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
