"""
VOID RESONANCE v2: Physics-Native Emergent Intelligence
=========================================================
Extends the v1 unified engine with 4 new extensions that emerge
from the wound/void/knot physics itself.

NEW Extensions:
1. WOUND RESONANCE CASCADE
   Spatial stress propagation within channels. Nearby wounded regions
   generate "phantom wounds" due to manifold tension.

2. VOID CRYSTALLIZATION
   Local phase transition when knot density reaches critical saturation.
   Crystallized regions become rigid and reflect waves.

3. SCAR EROSION (Competitive Forgetting)
   Healthy void (High V) slowly erodes nearby knots. Memory is maintained
   only through repeated "attention" (re-wounding).

4. WOUND ECHO (Post-Healing Oscillation)
   The vacuum rings after a wound heals, creating a damped oscillation.
   Acts as a short-term sensory buffer or "afterimage".

Physics Features:
- Void Gravity: Global vacuum recovery toward resting state (V=1.0).
- Geodesic Holographic Diffusion: Curvature-aware information flow.
- Non-synaptic Memory: Topological defects (knots) and phase states (crystals).

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


class VoidResonanceV2:
    """
    Physics-native emergent intelligence engine.

    The field remembers through structural deformation, not learned parameters.
    Memory emerges from wound cascades, crystallization, erosion, and echo dynamics.
    """

    def __init__(self, H=32, W=32, K=8, **kwargs):
        self.H, self.W, self.K = H, W, K
        self.bk = HybridBackend()

        # --- Tuning parameters ---
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
        self.echo = self.bk.zeros_float((H, W, K))
        self.stress = self.bk.zeros_float((H, W))
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
            'crystal': self.crystal.copy(), 'echo': self.echo.copy(),
            'stress': self.stress.copy(),
        }

    def load_state(self, state):
        """Restore state fields from a dictionary produced by get_state()."""
        self.t = state['t']
        self.psi = state['psi'].copy()
        self.V = state['V'].copy()
        self.R = state['R'].copy()
        self.knots = state['knots'].copy()
        self.crystal = state['crystal'].copy()
        self.echo = state['echo'].copy()
        self.stress = state['stress'].copy()

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

        # 7. CRYSTALLIZATION + EROSION
        self.crystal = bk.clamp(
            self.crystal * 0.998
            + bk.maximum(self.knots - self.crystal_threshold, 0) * 0.1,
            0, 1)
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
    print("  VOID RESONANCE v2 FINAL: Physics-Native Extensions")
    print("=" * 70)

    s_A = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    s_B = np.array([0, 0, 1, 0, 0, 0, 0, 0])
    empty = np.zeros(8)
    results = []
    t_start = time.time()

    # ---- T1: Resonance Cascade ----
    e1 = VoidResonanceV2(resonance_range=0.5)
    e2 = VoidResonanceV2(resonance_range=0)
    for _ in range(50):
        e1.step(s_A); e2.step(s_A)
    v1_far = e1.V[5:, :, 0].mean()
    v2_far = e2.V[5:, :, 0].mean()
    pass1 = v1_far < v2_far
    results.append(pass1)
    print(f"[PASS={pass1}] T1: Resonance Cascade "
          f"(resonant={v1_far:.4f} < no_res={v2_far:.4f})")

    # ---- T2: Crystallization ----
    e3 = VoidResonanceV2(knot_threshold=0.3, crystal_threshold=0.1)
    for _ in range(80):
        e3.step(s_A)
    c_sum = e3.crystal[:, :, 0].sum()
    pass2 = c_sum > 1.0
    results.append(pass2)
    print(f"[PASS={pass2}] T2: Crystallization "
          f"(crystal_sum={c_sum:.2f} > 1.0)")

    # ---- T3: Scar Erosion ----
    e4 = VoidResonanceV2(knot_threshold=0.3, erosion_rate=0.5)
    e5 = VoidResonanceV2(knot_threshold=0.3, erosion_rate=0.0)
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
    e_echo = VoidResonanceV2(echo_coupling=0.1)
    e_noecho = VoidResonanceV2(echo_coupling=0)
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
    e_h = VoidResonanceV2()
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
    e_b = VoidResonanceV2(knot_threshold=1.2)
    aA = [e_b.step(s_A) for _ in range(80)]
    aB = [e_b.step(s_B) for _ in range(80)]
    act_A = Counter(aA).most_common(1)[0][0]
    act_B = Counter(aB).most_common(1)[0][0]
    pass6 = act_A != act_B
    results.append(pass6)
    print(f"[PASS={pass6}] T6: Behavioral Differentiation "
          f"(A={act_A} B={act_B})")

    print("-" * 70)
    print(" v2 Extended Validation Tests")
    print("-" * 70)

    # ---- T7: Timestep Counter ----
    e_t = VoidResonanceV2()
    for _ in range(25):
        e_t.step(s_A)
    pass7 = e_t.t == 25
    results.append(pass7)
    print(f"[PASS={pass7}] T7: Timestep Counter (t={e_t.t} == 25)")

    # ---- T8: State Reset ----
    e_r = VoidResonanceV2()
    for _ in range(30):
        e_r.step(s_A)
    e_r.reset()
    pass8 = (e_r.t == 0
             and e_r.V.mean() == 1.0
             and e_r.knots.sum() == 0.0
             and np.abs(e_r.psi).sum() == 0.0)
    results.append(pass8)
    print(f"[PASS={pass8}] T8: State Reset (t={e_r.t}, V={e_r.V.mean():.1f})")

    # ---- T9: Save / Load State ----
    e_sl = VoidResonanceV2(knot_threshold=0.3)
    for _ in range(40):
        e_sl.step(s_A)
    saved = e_sl.get_state()
    t_saved = e_sl.t
    knots_saved = e_sl.knots.sum()
    for _ in range(20):
        e_sl.step(s_B)
    e_sl.load_state(saved)
    pass9 = (e_sl.t == t_saved
             and abs(e_sl.knots.sum() - knots_saved) < 1e-6)
    results.append(pass9)
    print(f"[PASS={pass9}] T9: Save/Load State "
          f"(t_restored={e_sl.t} == {t_saved})")

    # ---- T10: Diagnostics Completeness ----
    e_d = VoidResonanceV2()
    for _ in range(10):
        e_d.step(s_A)
    diag = e_d.get_diagnostics()
    required_keys = {'t', 'psi_mean', 'psi_max', 'V_mean', 'V_min',
                     'knots_total', 'knots_max', 'crystal_total',
                     'crystal_max', 'stress_mean', 'stress_max',
                     'echo_energy', 'R_mean', 'R_max', 'field_energy'}
    pass10 = required_keys.issubset(set(diag.keys()))
    results.append(pass10)
    print(f"[PASS={pass10}] T10: Diagnostics "
          f"({len(diag)} fields, energy={diag['field_energy']:.2f})")

    # ---- T11: Zero-Input Stability ----
    e_z = VoidResonanceV2()
    for _ in range(200):
        e_z.step(empty)
    v_stable = e_z.V.mean()
    psi_stable = np.abs(e_z.psi).max()
    pass11 = v_stable > 0.99 and psi_stable < 1e-6
    results.append(pass11)
    print(f"[PASS={pass11}] T11: Zero-Input Stability "
          f"(V={v_stable:.4f}, psi_max={psi_stable:.2e})")

    # ---- T12: Multi-Channel Independence ----
    e_m = VoidResonanceV2(knot_threshold=0.3)
    s_ch0 = np.array([5, 0, 0, 0, 0, 0, 0, 0])
    for _ in range(60):
        e_m.step(s_ch0)
    ch0_crystal = e_m.crystal[:, :, 0].sum()
    ch3_crystal = e_m.crystal[:, :, 3].sum()
    pass12 = ch0_crystal > ch3_crystal * 2
    results.append(pass12)
    print(f"[PASS={pass12}] T12: Channel Independence "
          f"(ch0={ch0_crystal:.2f} >> ch3={ch3_crystal:.2f})")

    # ---- T13: Field Energy Monotonicity (idle decay) ----
    e_en = VoidResonanceV2()
    for _ in range(40):
        e_en.step(s_A)
    energies = []
    for _ in range(50):
        e_en.step(empty)
        energies.append(e_en.get_diagnostics()['field_energy'])
    decay_count = sum(1 for i in range(1, len(energies))
                      if energies[i] <= energies[i - 1] * 1.01)
    pass13 = decay_count > len(energies) * 0.7
    results.append(pass13)
    print(f"[PASS={pass13}] T13: Idle Energy Decay "
          f"({decay_count}/{len(energies) - 1} steps decaying)")

    # ---- T14: Stress Response Proportionality ----
    e_lo = VoidResonanceV2()
    e_hi = VoidResonanceV2()
    s_lo = np.array([0.5, 0, 0, 0, 0, 0, 0, 0])
    s_hi = np.array([3.0, 0, 0, 0, 0, 0, 0, 0])
    for _ in range(30):
        e_lo.step(s_lo); e_hi.step(s_hi)
    pass14 = e_hi.stress.mean() > e_lo.stress.mean()
    results.append(pass14)
    print(f"[PASS={pass14}] T14: Stress Proportionality "
          f"(hi={e_hi.stress.mean():.4f} > lo={e_lo.stress.mean():.4f})")

    # ---- Summary ----
    elapsed = time.time() - t_start
    print("=" * 70)
    total = sum(results)
    print(f"FINAL RESULT: {total}/{len(results)} PASS  "
          f"({elapsed:.2f}s)")
    print(f"  v2 core:     {sum(results[:6])}/6")
    print(f"  v2 extended: {sum(results[6:])}/{len(results) - 6}")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
