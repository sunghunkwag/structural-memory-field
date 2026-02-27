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
import math
from typing import Optional, Literal
import numpy as np
from collections import Counter

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
        if s_ph is None:
            s_ph = np.zeros_like(s_amp)
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

    # T1: Resonance Cascade
    e1 = VoidResonanceV3(resonance_range=0.5)
    e2 = VoidResonanceV3(resonance_range=0)
    for _ in range(50):
        e1.step(s_A); e2.step(s_A)
    pass1 = e1.V[5:, :, 0].mean() < e2.V[5:, :, 0].mean()
    results.append(pass1)
    print(f"[PASS={pass1}] T1: Resonance Cascade")

    # T2: Crystallization
    e3 = VoidResonanceV3(knot_threshold=0.3, crystal_threshold=0.1)
    for _ in range(80):
        e3.step(s_A)
    pass2 = e3.crystal[:, :, 0].sum() > 1.0
    results.append(pass2)
    print(f"[PASS={pass2}] T2: Crystallization")

    # T3: Scar Erosion
    e4 = VoidResonanceV3(knot_threshold=0.3, erosion_rate=0.5)
    e5 = VoidResonanceV3(knot_threshold=0.3, erosion_rate=0.0)
    for _ in range(50):
        e4.step(s_A); e5.step(s_A)
    k4 = e4.knots.sum(); k5 = e5.knots.sum()
    for _ in range(100):
        e4.step(empty); e5.step(empty)
    pass3 = (k4 - e4.knots.sum()) > (k5 - e5.knots.sum())
    results.append(pass3)
    print(f"[PASS={pass3}] T3: Scar Erosion")

    # T4: Wound Echo
    e_echo = VoidResonanceV3(echo_coupling=0.1)
    e_noecho = VoidResonanceV3(echo_coupling=0)
    for _ in range(30):
        e_echo.step(s_A); e_noecho.step(s_A)
    v_ec = []; v_ne = []
    for _ in range(40):
        e_echo.step(empty); e_noecho.step(empty)
        v_ec.append(e_echo.V[:, :, 0].mean())
        v_ne.append(e_noecho.V[:, :, 0].mean())
    pass4 = np.var(v_ec) > np.var(v_ne)
    results.append(pass4)
    print(f"[PASS={pass4}] T4: Wound Echo")

    # T5: Holographic Recall
    e_h = VoidResonanceV3()
    s_AB = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    for _ in range(50):
        e_h.step(s_AB, np.array([0.7, -0.7, 0, 0, 0, 0, 0, 0]))
    e_h.psi *= 0
    for _ in range(20):
        e_h.step(s_A, np.array([0.7, 0, 0, 0, 0, 0, 0, 0]))
    a = np.abs(e_h.psi)
    pass5 = a[:, :, 1].mean() > a[:, :, 2].mean() * 3
    results.append(pass5)
    print(f"[PASS={pass5}] T5: Holographic Recall")

    # T6: Behavioral Differentiation
    e_b = VoidResonanceV3(knot_threshold=1.2)
    aA = [e_b.step(s_A) for _ in range(80)]
    aB = [e_b.step(s_B) for _ in range(80)]
    pass6 = (Counter(aA).most_common(1)[0][0]
             != Counter(aB).most_common(1)[0][0])
    results.append(pass6)
    print(f"[PASS={pass6}] T6: Behavioral Differentiation "
          f"(A={Counter(aA).most_common(1)[0][0]} "
          f"B={Counter(aB).most_common(1)[0][0]})")

    print("-" * 70)
    print(" v3 Extension Tests: Crystal Phase Dynamics")
    print("-" * 70)

    # T7: Crystal Phase Imprinting
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
        pass7 = abs(mean_phase_a - mean_phase_b) > 0.1
    else:
        pass7 = False
    results.append(pass7)
    print(f"[PASS={pass7}] T7: Crystal Phase Imprinting")

    # T8: Boundary Tension Emergence
    e_bt = VoidResonanceV3(
        knot_threshold=0.3, crystal_threshold=0.1,
        fracture_threshold=10.0)
    for _ in range(60):
        e_bt.step(s_A, np.array([1.0, 0, 0, 0, 0, 0, 0, 0]))
    for _ in range(60):
        e_bt.step(s_A, np.array([-2.0, 0, 0, 0, 0, 0, 0, 0]))
    pass8 = e_bt.boundary[:, :, 0].max() > 0.01
    results.append(pass8)
    print(f"[PASS={pass8}] T8: Boundary Tension "
          f"(max={e_bt.boundary[:, :, 0].max():.4f})")

    # T9: Crystal Fracture
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

    # T10: Crystal Fusion
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

    # T11: Nucleation at Boundaries
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

    # T12: Fracture Wave Burst (majority-of-steps test)
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

    print("=" * 70)
    total = sum(results)
    print(f"FINAL RESULT: {total}/{len(results)} PASS")
    print(f"  v2 core:  {sum(results[:6])}/6")
    print(f"  v3 phase: {sum(results[6:])}/6")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
