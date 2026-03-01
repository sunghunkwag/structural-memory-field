"""
VOID RESONANCE v3: Crystal Phase Dynamics
=========================================================

Extends v2 with rich phase transition physics where crystal regions
carry directional identity, collide, fuse, fracture, and nucleate
new structures at their boundaries.

NEW in v3 (Extension 5: Crystal Phase Dynamics):

5a. CRYSTAL PHASE IMPRINTING
    Each crystal region remembers the wave phase at the moment of
    crystallization. This gives crystals directional identity --
    crystals born from different stimuli carry different phase angles.

5b. BOUNDARY TENSION
    Where two crystal regions with different phases meet, a tension
    field emerges proportional to the phase mismatch. This is the
    "fault line" between incompatible memories.

5c. FRACTURE (Crystal Shattering)
    When boundary tension exceeds a critical threshold, crystals
    shatter back into knots, releasing stored energy as a wave burst.
    Rigid memory violently decomposes under structural conflict.

5d. FUSION (Crystal Merging)
    When adjacent crystals share similar phases, they reinforce each
    other and merge into stronger unified structures. Compatible
    memories consolidate.

5e. NUCLEATION (Boundary Birth)
    At boundary zones, the collision energy seeds new knot structures
    in empty space -- novel patterns born from the conflict of existing
    memories, belonging to neither parent.

Preserved from v2:
    1. Wound Resonance Cascade
    2. Void Crystallization
    3. Scar Erosion (Competitive Forgetting)
    4. Wound Echo (Post-Healing Oscillation)

Physics Features:
    - Void Gravity: Global vacuum recovery toward resting state (V=1.0).
    - Geodesic Holographic Diffusion: Curvature-aware information flow.
    - Non-synaptic Memory: Topological defects (knots), phase-imprinted
      crystals, and boundary-born nucleation sites.

Requirements: numpy
"""

from __future__ import annotations

from typing import Optional, Literal
import numpy as np
from collections import Counter
import time

BackendName = Literal["numpy", "torch"]


class HybridBackend:
    """Simplified backend for real+complex tensor ops."""

    def __init__(self, name: BackendName = "numpy", device: str = "cpu"):
        self.name = name

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

    def to_numpy(self, x):
        return np.asarray(x)

    def laplacian_2d(self, x):
        return (self.roll(x, -1, 0) + self.roll(x, 1, 0)
                + self.roll(x, -1, 1) + self.roll(x, 1, 1) - 4.0 * x)

    def gradient_magnitude(self, x):
        dy = self.roll(x, -1, 0) - x
        dx = self.roll(x, -1, 1) - x
        return self.sqrt(dy * dy + dx * dx + 1e-12)

    def curvature_diffusion(self, wave, curvature, knots, crystal):
        crystal_block = 1.0 / (1.0 + crystal * 20.0)
        conductivity = curvature * crystal_block / (1.0 + knots * 3.0)
        up = self.roll(wave, -1, 0) * self.roll(conductivity, -1, 0)
        down = self.roll(wave, 1, 0) * self.roll(conductivity, 1, 0)
        left = self.roll(wave, -1, 1) * self.roll(conductivity, -1, 1)
        right = self.roll(wave, 1, 1) * self.roll(conductivity, 1, 1)
        csum = (self.roll(conductivity, -1, 0) + self.roll(conductivity, 1, 0)
                + self.roll(conductivity, -1, 1) + self.roll(conductivity, 1, 1)
                + 1e-8)
        flow = (up + down + left + right) / csum
        energy = np.expand_dims(np.mean(np.abs(wave), axis=-1), -1)
        carved = np.clip(curvature - 1.1, a_min=0, a_max=None)
        scatter = (energy.astype(np.complex64)
                   * carved.astype(np.complex64) * 0.15)
        return flow + scatter + wave * crystal * 0.3


class VoidResonanceV3:
    """
    Physics-native emergent intelligence engine with crystal phase dynamics.

    The field remembers through structural deformation, not learned parameters.
    Crystals now carry phase identity, enabling collision, fusion, fracture,
    and spontaneous nucleation at phase boundaries.
    """

    def __init__(self, H=32, W=32, K=8, **kwargs):
        self.H, self.W, self.K = H, W, K
        self.bk = HybridBackend()

        # --- v2 tuning parameters ---
        self.alpha = kwargs.get('alpha', 0.25)
        self.plasticity = kwargs.get('plasticity', 0.08)
        self.healing_rate = kwargs.get('healing_rate', 0.15)
        self.wound_depth = kwargs.get('wound_depth', 0.35)
        self.knot_threshold = kwargs.get('knot_threshold', 1.2)
        self.crystal_threshold = kwargs.get('crystal_threshold', 0.1)
        self.erosion_rate = kwargs.get('erosion_rate', 0.1)
        self.echo_damping = kwargs.get('echo_damping', 0.9)
        self.echo_coupling = kwargs.get('echo_coupling', 0.05)
        self.resonance_range = kwargs.get('resonance_range', 0.5)

        # --- v3 crystal phase dynamics parameters ---
        self.fracture_threshold = kwargs.get('fracture_threshold', 0.6)
        self.fracture_energy_release = kwargs.get('fracture_energy_release', 0.3)
        self.fracture_decay = kwargs.get('fracture_decay', 0.5)
        self.fusion_rate = kwargs.get('fusion_rate', 0.05)
        self.fusion_phase_tolerance = kwargs.get('fusion_phase_tolerance', 0.1)
        self.nucleation_rate = kwargs.get('nucleation_rate', 0.1)

        # --- State fields ---
        self._init_state()

    def _init_state(self):
        """Initialize / reset all state fields to resting configuration."""
        H, W, K = self.H, self.W, self.K
        self.psi = self.bk.zeros_complex((H, W, K))
        self.V = self.bk.ones_float((H, W, K))
        self.R = self.bk.ones_float((H, W, K))
        self.knots = self.bk.zeros_float((H, W, K))
        self.crystal = self.bk.zeros_float((H, W, K))
        self.crystal_phase = self.bk.zeros_float((H, W, K))
        self.echo = self.bk.zeros_float((H, W, K))
        self.stress = self.bk.zeros_float((H, W))
        self.boundary = self.bk.zeros_float((H, W, K))
        self.num_actions = min(4, K)
        self.t = 0

    def reset(self):
        """Reset all state fields to initial resting configuration."""
        self._init_state()

    def get_diagnostics(self):
        """Return a dictionary of field statistics for monitoring."""
        amp = np.abs(self.psi)
        return {
            't': self.t,
            'psi_mean': float(amp.mean()),
            'psi_max': float(amp.max()),
            'V_mean': float(self.V.mean()),
            'V_min': float(self.V.min()),
            'knots_total': float(self.knots.sum()),
            'knots_max': float(self.knots.max()),
            'crystal_total': float(self.crystal.sum()),
            'crystal_max': float(self.crystal.max()),
            'crystal_phase_std': float(self.crystal_phase.std()),
            'boundary_max': float(self.boundary.max()),
            'boundary_mean': float(self.boundary.mean()),
            'stress_mean': float(self.stress.mean()),
            'stress_max': float(self.stress.max()),
            'echo_energy': float(np.abs(self.echo).sum()),
            'R_mean': float(self.R.mean()),
            'R_max': float(self.R.max()),
            'field_energy': float((amp ** 2).sum()),
        }

    def get_state(self):
        """Serialize all state fields to a dictionary of numpy arrays."""
        return {
            'H': self.H, 'W': self.W, 'K': self.K, 't': self.t,
            'psi': self.psi.copy(), 'V': self.V.copy(),
            'R': self.R.copy(), 'knots': self.knots.copy(),
            'crystal': self.crystal.copy(),
            'crystal_phase': self.crystal_phase.copy(),
            'echo': self.echo.copy(), 'stress': self.stress.copy(),
            'boundary': self.boundary.copy(),
        }

    def load_state(self, state):
        """Restore state fields from a dictionary produced by get_state()."""
        self.t = state['t']
        self.psi = state['psi'].copy()
        self.V = state['V'].copy()
        self.R = state['R'].copy()
        self.knots = state['knots'].copy()
        self.crystal = state['crystal'].copy()
        self.crystal_phase = state['crystal_phase'].copy()
        self.echo = state['echo'].copy()
        self.stress = state['stress'].copy()
        self.boundary = state['boundary'].copy()

    def _crystal_boundary_tension(self):
        bk = self.bk
        cp = self.crystal_phase
        cr = self.crystal
        dp_up = np.abs(np.sin(cp - bk.roll(cp, -1, 0)))
        dp_dn = np.abs(np.sin(cp - bk.roll(cp, 1, 0)))
        dp_lt = np.abs(np.sin(cp - bk.roll(cp, -1, 1)))
        dp_rt = np.abs(np.sin(cp - bk.roll(cp, 1, 1)))
        phase_gradient = (dp_up + dp_dn + dp_lt + dp_rt) / 4.0
        cr_up = bk.roll(cr, -1, 0)
        cr_dn = bk.roll(cr, 1, 0)
        cr_lt = bk.roll(cr, -1, 1)
        cr_rt = bk.roll(cr, 1, 1)
        neighbor_crystal = (cr_up + cr_dn + cr_lt + cr_rt) / 4.0
        boundary = phase_gradient * cr * neighbor_crystal
        return boundary, neighbor_crystal

    def _crystal_phase_dynamics(self, boundary, neighbor_crystal, idle):
        bk = self.bk
        # --- FRACTURE ---
        fracture = bk.maximum(boundary - self.fracture_threshold, 0)
        shatter_amount = fracture * self.crystal * 2.0
        self.knots += shatter_amount
        self.crystal = bk.maximum(
            self.crystal - fracture * self.fracture_decay, 0)
        burst = (fracture * np.exp(1j * self.crystal_phase)).astype(
            np.complex64)
        self.psi += burst * self.fracture_energy_release
        # --- FUSION ---
        low_tension = bk.maximum(
            self.fusion_phase_tolerance - boundary, 0)
        fusion = low_tension * self.crystal * neighbor_crystal
        self.crystal = bk.clamp(
            self.crystal + fusion * self.fusion_rate, 0, 1)
        # --- NUCLEATION ---
        empty_space = bk.maximum(1.0 - self.crystal, 0)
        viable = bk.maximum(self.V - 0.3, 0)
        nucleation = boundary * empty_space * viable * self.nucleation_rate
        self.knots += nucleation
        return fracture.sum(), fusion.sum(), nucleation.sum()

    def step(self, s_amp, s_ph=None):
        bk = self.bk
        s_amp = np.asarray(s_amp, dtype=np.float64)
        if s_ph is None:
            s_ph = np.zeros_like(s_amp)
        else:
            s_ph = np.asarray(s_ph, dtype=np.float64)
        V_before = self.V.copy()

        # 1. WOUND + RESONANCE CASCADE
        s = np.zeros(self.K)
        n = min(len(s_amp), self.K)
        s[:n] = s_amp[:n]
        w = bk.array_float(s) * self.wound_depth
        for r in range(4):
            self.V[r, :, :] -= w * (1 - r / 4)
        self.V = bk.clamp(self.V, 0, 1)
        input_level = float(np.abs(s_amp[:n]).sum())
        idle = 1.0 if input_level < 1e-6 else 0.0
        wf = bk.maximum(0.7 - self.V, 0)
        phantom = (bk.maximum(bk.laplacian_2d(wf), 0)
                   * self.V * self.resonance_range)
        self.V = bk.clamp(self.V - phantom, 0, 1)

        # 2. WAVE INJECTION
        ev = bk.array_complex(s_amp[:n] * np.exp(1j * s_ph[:n]))
        for r in range(4):
            rec = bk.maximum(
                self.V[r, :, :],
                bk.clamp(self.R[r, :, :] - 1.0, 0, 5) * 0.3)
            self.psi[r, :, :] += (
                ev * (1 - r / 4) * rec
                * (1 - self.crystal[r, :, :] * 0.8))

        # 3. HEALING (Void Gravity)
        lapV = bk.laplacian_2d(self.V)
        recovery = 0.3 * (1.0 - self.V)
        self.V = bk.clamp(
            self.V + self.healing_rate * (lapV + recovery)
            / (1 + self.knots * 5 + self.crystal * 10), 0, 1)

        # 4. WOUND ECHO
        delta = self.V - V_before
        self.echo = (self.echo * self.echo_damping
                     + delta * (1 - self.echo_damping))
        self.V = bk.clamp(
            self.V + self.echo * self.echo_coupling, 0, 1)

        # 5. GEODESIC HOLOGRAPHIC DIFFUSION
        self.psi = (
            (1 - self.alpha) * self.psi
            + self.alpha * bk.curvature_diffusion(
                self.psi, self.R, self.knots, self.crystal))
        self.psi *= (0.95 + 0.04 * self.V) * (0.90 if idle else 1.0)
        amp = bk.abs(self.psi)

        # 6. STRESS + KNOTS
        gv = bk.gradient_magnitude(self.V)
        ga = bk.gradient_magnitude(amp)
        raw_stress = (gv + ga * 0.5).sum(-1)
        self.stress = bk.clamp(
            raw_stress + 0.05 * bk.laplacian_2d(
                bk.expand_dim(raw_stress, -1)).reshape(self.H, self.W),
            0, 100)
        st3 = bk.expand_dim(self.stress, -1)
        self.knots = bk.clamp(
            self.knots
            + bk.maximum(st3 - self.knot_threshold, 0)
            * bk.maximum(0.8 - self.V, 0) * 0.1,
            0, 10)

        # 7. CRYSTALLIZATION + PHASE DYNAMICS (v3)
        new_crystal = bk.maximum(
            self.knots - self.crystal_threshold, 0) * 0.1
        forming = (new_crystal > 0.001).astype(np.float32)
        psi_angle = np.angle(self.psi)
        self.crystal_phase = (
            self.crystal_phase * (1 - forming) + psi_angle * forming)
        self.crystal = bk.clamp(
            self.crystal * 0.998 + new_crystal, 0, 1)
        self.boundary, neighbor_crystal = (
            self._crystal_boundary_tension())
        self._crystal_phase_dynamics(
            self.boundary, neighbor_crystal, idle)
        elig = bk.maximum(self.V - 0.9, 0)
        self.knots = bk.maximum(
            self.knots - self.erosion_rate * elig * self.knots, 0)
        self.knots = bk.maximum(
            self.knots - (0.02 * idle) * self.knots, 0)

        # 8. CURVATURE UPDATE
        self.R = bk.clamp(
            self.R + st3 * amp * self.plasticity
            + 0.005 * (1 - self.R), 0.1, 10)

        # ACTION READOUT
        sc = []
        for c in range(self.num_actions):
            sc.append(
                ((1 - self.V[:, :, c])
                 * amp[:, :, c]
                 * (1 + self.knots[:, :, c] * 3)
                 * (self.stress + 0.01)
                 * (1 + self.crystal[:, :, c] * 5)).sum())
        self.t += 1
        return int(np.argmax(sc))


def run_tests():
    print("=" * 70)
    print(" VOID RESONANCE v3: Crystal Phase Dynamics")
    print("=" * 70)

    s_A = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    s_B = np.array([0, 0, 1, 0, 0, 0, 0, 0])
    empty = np.zeros(8)
    results = []
    t_start = time.time()

    # ---- T1: Resonance Cascade ----
    e1 = VoidResonanceV3(resonance_range=0.5)
    e2 = VoidResonanceV3(resonance_range=0)
    for _ in range(50):
        e1.step(s_A); e2.step(s_A)
    v1_far = e1.V[5:, :, 0].mean()
    v2_far = e2.V[5:, :, 0].mean()
    pass1 = v1_far < v2_far
    results.append(pass1)
    print(f"[PASS={pass1}] T1: Resonance Cascade "
          f"(resonant={v1_far:.4f} < no_res={v2_far:.4f})")

    # ---- T2: Crystallization ----
    e3 = VoidResonanceV3(knot_threshold=0.3, crystal_threshold=0.1)
    for _ in range(80):
        e3.step(s_A)
    c_sum = e3.crystal[:, :, 0].sum()
    pass2 = c_sum > 1.0
    results.append(pass2)
    print(f"[PASS={pass2}] T2: Crystallization "
          f"(crystal_sum={c_sum:.2f} > 1.0)")

    # ---- T3: Scar Erosion ----
    e4 = VoidResonanceV3(knot_threshold=0.3, erosion_rate=0.5)
    e5 = VoidResonanceV3(knot_threshold=0.3, erosion_rate=0.0)
    for _ in range(50):
        e4.step(s_A); e5.step(s_A)
    k4 = e4.knots.sum(); k5 = e5.knots.sum()
    for _ in range(100):
        e4.step(empty); e5.step(empty)
    erosion4 = k4 - e4.knots.sum()
    erosion5 = k5 - e5.knots.sum()
    pass3 = erosion4 > erosion5
    results.append(pass3)
    print(f"[PASS={pass3}] T3: Scar Erosion "
          f"(eroded={erosion4:.2f} > retained={erosion5:.2f})")

    # ---- T4: Wound Echo ----
    e_echo = VoidResonanceV3(echo_coupling=0.1)
    e_noecho = VoidResonanceV3(echo_coupling=0)
    for _ in range(30):
        e_echo.step(s_A); e_noecho.step(s_A)
    v_ec = []; v_ne = []
    for _ in range(40):
        e_echo.step(empty); e_noecho.step(empty)
        v_ec.append(e_echo.V[:, :, 0].mean())
        v_ne.append(e_noecho.V[:, :, 0].mean())
    var_ec = np.var(v_ec); var_ne = np.var(v_ne)
    pass4 = var_ec > var_ne
    results.append(pass4)
    print(f"[PASS={pass4}] T4: Wound Echo "
          f"(echo_var={var_ec:.2e} > no_echo_var={var_ne:.2e})")

    # ---- T5: Holographic Recall ----
    e_h = VoidResonanceV3()
    s_AB = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    for _ in range(50):
        e_h.step(s_AB, np.array([0.7, -0.7, 0, 0, 0, 0, 0, 0]))
    e_h.psi *= 0
    for _ in range(20):
        e_h.step(s_A, np.array([0.7, 0, 0, 0, 0, 0, 0, 0]))
    a = np.abs(e_h.psi)
    ch1_mean = a[:, :, 1].mean()
    ch2_mean = a[:, :, 2].mean()
    pass5 = ch1_mean > ch2_mean * 3
    results.append(pass5)
    print(f"[PASS={pass5}] T5: Holographic Recall "
          f"(ch1={ch1_mean:.4f} > 3*ch2={ch2_mean * 3:.4f})")

    # ---- T6: Behavioral Differentiation ----
    e_b = VoidResonanceV3(knot_threshold=1.2)
    aA = [e_b.step(s_A) for _ in range(80)]
    aB = [e_b.step(s_B) for _ in range(80)]
    act_A = Counter(aA).most_common(1)[0][0]
    act_B = Counter(aB).most_common(1)[0][0]
    pass6 = act_A != act_B
    results.append(pass6)
    print(f"[PASS={pass6}] T6: Behavioral Differentiation "
          f"(A={act_A} B={act_B})")

    print("-" * 70)
    print(" v3 Extension Tests: Crystal Phase Dynamics")
    print("-" * 70)

    # ---- T7: Crystal Phase Imprinting ----
    e_pa = VoidResonanceV3(knot_threshold=0.3, crystal_threshold=0.1)
    e_pb = VoidResonanceV3(knot_threshold=0.3, crystal_threshold=0.1)
    for _ in range(80):
        e_pa.step(s_A, np.array([0.5, 0, 0, 0, 0, 0, 0, 0]))
        e_pb.step(s_A, np.array([-1.5, 0, 0, 0, 0, 0, 0, 0]))
    mask_a = e_pa.crystal[:, :, 0] > 0.3
    mask_b = e_pb.crystal[:, :, 0] > 0.3
    if mask_a.any() and mask_b.any():
        mean_phase_a = np.mean(e_pa.crystal_phase[:, :, 0][mask_a])
        mean_phase_b = np.mean(e_pb.crystal_phase[:, :, 0][mask_b])
        phase_diff = abs(mean_phase_a - mean_phase_b)
        pass7 = phase_diff > 0.1
    else:
        phase_diff = 0.0
        pass7 = False
    results.append(pass7)
    print(f"[PASS={pass7}] T7: Crystal Phase Imprinting "
          f"(phase_diff={phase_diff:.4f})")

    # ---- T8: Boundary Tension Emergence ----
    e_bt = VoidResonanceV3(
        knot_threshold=0.3, crystal_threshold=0.1,
        fracture_threshold=10.0)
    for _ in range(60):
        e_bt.step(s_A, np.array([1.0, 0, 0, 0, 0, 0, 0, 0]))
    for _ in range(60):
        e_bt.step(s_A, np.array([-2.0, 0, 0, 0, 0, 0, 0, 0]))
    b_max = e_bt.boundary[:, :, 0].max()
    pass8 = b_max > 0.01
    results.append(pass8)
    print(f"[PASS={pass8}] T8: Boundary Tension "
          f"(max={b_max:.4f} > 0.01)")

    # ---- T9: Crystal Fracture ----
    e_frac = VoidResonanceV3(
        knot_threshold=0.3, crystal_threshold=0.1,
        fracture_threshold=0.05)
    e_nofrac = VoidResonanceV3(
        knot_threshold=0.3, crystal_threshold=0.1,
        fracture_threshold=100.0)
    for _ in range(60):
        e_frac.step(s_A, np.array([1.0, 0, 0, 0, 0, 0, 0, 0]))
        e_nofrac.step(s_A, np.array([1.0, 0, 0, 0, 0, 0, 0, 0]))
    for _ in range(60):
        e_frac.step(s_A, np.array([-2.0, 0, 0, 0, 0, 0, 0, 0]))
        e_nofrac.step(s_A, np.array([-2.0, 0, 0, 0, 0, 0, 0, 0]))
    c_frac = e_frac.crystal[:, :, 0].sum()
    c_nofrac = e_nofrac.crystal[:, :, 0].sum()
    pass9 = c_frac < c_nofrac
    results.append(pass9)
    print(f"[PASS={pass9}] T9: Crystal Fracture "
          f"(frac={c_frac:.2f} vs intact={c_nofrac:.2f})")

    # ---- T10: Crystal Fusion ----
    e_fuse = VoidResonanceV3(
        knot_threshold=0.3, crystal_threshold=0.1,
        fusion_rate=0.15, fracture_threshold=100.0)
    e_nofuse = VoidResonanceV3(
        knot_threshold=0.3, crystal_threshold=0.1,
        fusion_rate=0.0, fracture_threshold=100.0)
    for _ in range(100):
        e_fuse.step(s_A, np.array([0.5, 0, 0, 0, 0, 0, 0, 0]))
        e_nofuse.step(s_A, np.array([0.5, 0, 0, 0, 0, 0, 0, 0]))
    c_fuse = e_fuse.crystal[:, :, 0].sum()
    c_nofuse = e_nofuse.crystal[:, :, 0].sum()
    pass10 = c_fuse > c_nofuse
    results.append(pass10)
    print(f"[PASS={pass10}] T10: Crystal Fusion "
          f"(fused={c_fuse:.2f} vs separate={c_nofuse:.2f})")

    # ---- T11: Nucleation at Boundaries ----
    e_nuc = VoidResonanceV3(
        knot_threshold=0.3, crystal_threshold=0.1,
        nucleation_rate=0.3, fracture_threshold=100.0)
    e_nonuc = VoidResonanceV3(
        knot_threshold=0.3, crystal_threshold=0.1,
        nucleation_rate=0.0, fracture_threshold=100.0)
    for _ in range(60):
        e_nuc.step(s_A, np.array([1.0, 0, 0, 0, 0, 0, 0, 0]))
        e_nonuc.step(s_A, np.array([1.0, 0, 0, 0, 0, 0, 0, 0]))
    for _ in range(60):
        e_nuc.step(s_A, np.array([-2.0, 0, 0, 0, 0, 0, 0, 0]))
        e_nonuc.step(s_A, np.array([-2.0, 0, 0, 0, 0, 0, 0, 0]))
    k_nuc = e_nuc.knots[:, :, 0].sum()
    k_nonuc = e_nonuc.knots[:, :, 0].sum()
    pass11 = k_nuc > k_nonuc
    results.append(pass11)
    print(f"[PASS={pass11}] T11: Boundary Nucleation "
          f"(nucleated={k_nuc:.2f} vs baseline={k_nonuc:.2f})")

    # ---- T12: Fracture Wave Burst ----
    e_burst = VoidResonanceV3(
        knot_threshold=0.3, crystal_threshold=0.1,
        fracture_threshold=0.05)
    e_noburst = VoidResonanceV3(
        knot_threshold=0.3, crystal_threshold=0.1,
        fracture_threshold=100.0)
    for _ in range(60):
        e_burst.step(s_A, np.array([1.0, 0, 0, 0, 0, 0, 0, 0]))
        e_noburst.step(s_A, np.array([1.0, 0, 0, 0, 0, 0, 0, 0]))
    burst_wins = 0
    total_steps = 60
    for _ in range(total_steps):
        e_burst.step(s_A, np.array([-2.0, 0, 0, 0, 0, 0, 0, 0]))
        e_noburst.step(s_A, np.array([-2.0, 0, 0, 0, 0, 0, 0, 0]))
        if np.abs(e_burst.psi[:, :, 0]).mean() > np.abs(
                e_noburst.psi[:, :, 0]).mean():
            burst_wins += 1
    pass12 = burst_wins > total_steps // 2
    results.append(pass12)
    print(f"[PASS={pass12}] T12: Fracture Wave Burst "
          f"(burst wins {burst_wins}/{total_steps} steps)")

    print("-" * 70)
    print(" v3 Extended Validation Tests")
    print("-" * 70)

    # ---- T13: Timestep Counter ----
    e_t = VoidResonanceV3()
    for _ in range(25):
        e_t.step(s_A)
    pass13 = e_t.t == 25
    results.append(pass13)
    print(f"[PASS={pass13}] T13: Timestep Counter (t={e_t.t} == 25)")

    # ---- T14: State Reset ----
    e_r = VoidResonanceV3()
    for _ in range(30):
        e_r.step(s_A)
    e_r.reset()
    pass14 = (e_r.t == 0
              and e_r.V.mean() == 1.0
              and e_r.knots.sum() == 0.0
              and np.abs(e_r.psi).sum() == 0.0
              and e_r.boundary.sum() == 0.0
              and e_r.crystal_phase.sum() == 0.0)
    results.append(pass14)
    print(f"[PASS={pass14}] T14: State Reset "
          f"(t={e_r.t}, V={e_r.V.mean():.1f})")

    # ---- T15: Save / Load State ----
    e_sl = VoidResonanceV3(knot_threshold=0.3)
    for _ in range(40):
        e_sl.step(s_A)
    saved = e_sl.get_state()
    t_saved = e_sl.t
    knots_saved = e_sl.knots.sum()
    crystal_saved = e_sl.crystal.sum()
    for _ in range(20):
        e_sl.step(s_B)
    e_sl.load_state(saved)
    pass15 = (e_sl.t == t_saved
              and abs(e_sl.knots.sum() - knots_saved) < 1e-6
              and abs(e_sl.crystal.sum() - crystal_saved) < 1e-6)
    results.append(pass15)
    print(f"[PASS={pass15}] T15: Save/Load State "
          f"(t_restored={e_sl.t} == {t_saved})")

    # ---- T16: Diagnostics Completeness ----
    e_d = VoidResonanceV3()
    for _ in range(10):
        e_d.step(s_A)
    diag = e_d.get_diagnostics()
    required_keys = {'t', 'psi_mean', 'psi_max', 'V_mean', 'V_min',
                     'knots_total', 'knots_max', 'crystal_total',
                     'crystal_max', 'crystal_phase_std', 'boundary_max',
                     'boundary_mean', 'stress_mean', 'stress_max',
                     'echo_energy', 'R_mean', 'R_max', 'field_energy'}
    pass16 = required_keys.issubset(set(diag.keys()))
    results.append(pass16)
    print(f"[PASS={pass16}] T16: Diagnostics "
          f"({len(diag)} fields, energy={diag['field_energy']:.2f})")

    # ---- T17: Zero-Input Stability ----
    e_z = VoidResonanceV3()
    for _ in range(200):
        e_z.step(empty)
    v_stable = e_z.V.mean()
    psi_stable = np.abs(e_z.psi).max()
    pass17 = v_stable > 0.99 and psi_stable < 1e-6
    results.append(pass17)
    print(f"[PASS={pass17}] T17: Zero-Input Stability "
          f"(V={v_stable:.4f}, psi_max={psi_stable:.2e})")

    # ---- T18: Crystal Phase Persistence After Reset ----
    e_cp = VoidResonanceV3(knot_threshold=0.3, crystal_threshold=0.1,
                           fracture_threshold=100.0)
    for _ in range(80):
        e_cp.step(s_A, np.array([1.0, 0, 0, 0, 0, 0, 0, 0]))
    phase_before = e_cp.crystal_phase.copy()
    crystal_before = e_cp.crystal.copy()
    for _ in range(20):
        e_cp.step(empty)
    mask = crystal_before[:, :, 0] > 0.3
    if mask.any():
        phase_drift = np.abs(
            e_cp.crystal_phase[:, :, 0][mask]
            - phase_before[:, :, 0][mask]).mean()
        pass18 = phase_drift < 0.5
    else:
        pass18 = False
    results.append(pass18)
    print(f"[PASS={pass18}] T18: Crystal Phase Persistence "
          f"(drift={phase_drift:.4f} < 0.5)")

    # ---- T19: Multi-Channel Independence ----
    e_m = VoidResonanceV3(knot_threshold=0.3)
    s_ch0 = np.array([5, 0, 0, 0, 0, 0, 0, 0])
    for _ in range(60):
        e_m.step(s_ch0)
    ch0_crystal = e_m.crystal[:, :, 0].sum()
    ch3_crystal = e_m.crystal[:, :, 3].sum()
    pass19 = ch0_crystal > ch3_crystal * 2
    results.append(pass19)
    print(f"[PASS={pass19}] T19: Channel Independence "
          f"(ch0={ch0_crystal:.2f} >> ch3={ch3_crystal:.2f})")

    # ---- T20: Fracture-Fusion Antagonism ----
    # Under phase conflict, fracture should reduce crystal relative to
    # an engine where fracture is disabled (fusion-only).
    e_both = VoidResonanceV3(
        knot_threshold=0.3, crystal_threshold=0.1,
        fracture_threshold=0.05, fusion_rate=0.15)
    e_fuse_only = VoidResonanceV3(
        knot_threshold=0.3, crystal_threshold=0.1,
        fracture_threshold=100.0, fusion_rate=0.15)
    for _ in range(60):
        e_both.step(s_A, np.array([1.0, 0, 0, 0, 0, 0, 0, 0]))
        e_fuse_only.step(s_A, np.array([1.0, 0, 0, 0, 0, 0, 0, 0]))
    for _ in range(60):
        e_both.step(s_A, np.array([-2.0, 0, 0, 0, 0, 0, 0, 0]))
        e_fuse_only.step(s_A, np.array([-2.0, 0, 0, 0, 0, 0, 0, 0]))
    c_both = e_both.crystal[:, :, 0].sum()
    c_fuse_only = e_fuse_only.crystal[:, :, 0].sum()
    pass20 = c_both < c_fuse_only
    results.append(pass20)
    print(f"[PASS={pass20}] T20: Fracture-Fusion Antagonism "
          f"(frac+fuse={c_both:.2f} < fuse_only={c_fuse_only:.2f})")

    # ---- Summary ----
    elapsed = time.time() - t_start
    print("=" * 70)
    total = sum(results)
    n_core = sum(results[:6])
    n_phase = sum(results[6:12])
    n_ext = sum(results[12:])
    print(f"FINAL RESULT: {total}/{len(results)} PASS  "
          f"({elapsed:.2f}s)")
    print(f"  v2 core:     {n_core}/6")
    print(f"  v3 phase:    {n_phase}/6")
    print(f"  v3 extended: {n_ext}/{len(results) - 12}")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
