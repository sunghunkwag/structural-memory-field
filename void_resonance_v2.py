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

import math
from typing import Optional, Literal
import numpy as np
from collections import Counter

BackendName = Literal["numpy", "torch"]

class HybridBackend:
    """Simplified backend for real+complex tensor ops."""
    def __init__(self, name: BackendName = "numpy", device: str = "cpu"):
        self.name = name
    def zeros_float(self, shape): return np.zeros(shape, dtype=np.float32)
    def ones_float(self, shape): return np.ones(shape, dtype=np.float32)
    def zeros_complex(self, shape): return np.zeros(shape, dtype=np.complex64)
    def array_float(self, x): return np.asarray(x, dtype=np.float32)
    def array_complex(self, x): return np.asarray(x, dtype=np.complex64)
    def roll(self, x, shift, axis): return np.roll(x, shift, axis)
    def clamp(self, x, lo, hi): return np.clip(x, lo, hi)
    def abs(self, x): return np.abs(x)
    def sqrt(self, x): return np.sqrt(x)
    def maximum(self, a, b): return np.maximum(a, b)
    def expand_dim(self, x, axis): return np.expand_dims(x, axis)
    def to_numpy(self, x): return np.asarray(x)
    def laplacian_2d(self, x):
        return (self.roll(x,-1,0)+self.roll(x,1,0)+self.roll(x,-1,1)+self.roll(x,1,1)-4.0*x)
    def gradient_magnitude(self, x):
        dy = self.roll(x,-1,0)-x; dx = self.roll(x,-1,1)-x
        return self.sqrt(dy*dy + dx*dx + 1e-12)
    def curvature_diffusion(self, wave, curvature, knots, crystal):
        crystal_block = 1.0 / (1.0 + crystal * 20.0)
        conductivity = curvature * crystal_block / (1.0 + knots * 3.0)
        up = self.roll(wave,-1,0)*self.roll(conductivity,-1,0)
        down = self.roll(wave,1,0)*self.roll(conductivity,1,0)
        left = self.roll(wave,-1,1)*self.roll(conductivity,-1,1)
        right = self.roll(wave,1,1)*self.roll(conductivity,1,1)
        csum = (self.roll(conductivity,-1,0)+self.roll(conductivity,1,0)+self.roll(conductivity,-1,1)+self.roll(conductivity,1,1)+1e-8)
        flow = (up+down+left+right)/csum
        energy = np.expand_dims(np.mean(np.abs(wave),axis=-1),-1)
        carved = np.clip(curvature-1.1, a_min=0, a_max=None)
        scatter = energy.astype(np.complex64)*carved.astype(np.complex64)*0.15
        return flow + scatter + wave*crystal*0.3

class VoidResonanceV2:
    def __init__(self, H=32, W=32, K=8, **kwargs):
        self.H, self.W, self.K = H, W, K
        self.bk = HybridBackend()
        # Tuning
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

        self.psi = self.bk.zeros_complex((H,W,K))
        self.V = self.bk.ones_float((H,W,K))
        self.R = self.bk.ones_float((H,W,K))
        self.knots = self.bk.zeros_float((H,W,K))
        self.crystal = self.bk.zeros_float((H,W,K))
        self.echo = self.bk.zeros_float((H,W,K))
        self.stress = self.bk.zeros_float((H,W))
        self.num_actions = min(4, K)
        self.t = 0

    def step(self, s_amp, s_ph=None):
        bk = self.bk
        if s_ph is None: s_ph = np.zeros_like(s_amp)
        V_before = self.V.copy()

        # 1. Wound + Resonance
        s = np.zeros(self.K); n = min(len(s_amp), self.K); s[:n] = s_amp[:n]
        w = bk.array_float(s) * self.wound_depth
        for r in range(4): self.V[r,:,:] -= w * (1-r/4)
        self.V = bk.clamp(self.V, 0, 1)

        input_level = float(np.abs(s_amp[:n]).sum())
        idle = 1.0 if input_level < 1e-6 else 0.0

        wf = bk.maximum(0.7 - self.V, 0)
        phantom = bk.maximum(bk.laplacian_2d(wf), 0) * self.V * self.resonance_range
        self.V = bk.clamp(self.V - phantom, 0, 1)

        # 2. Wave
        ev = bk.array_complex(s_amp[:n] * np.exp(1j*s_ph[:n]))
        for r in range(4):
            rec = bk.maximum(self.V[r,:,:], bk.clamp(self.R[r,:,:]-1.0, 0, 5)*0.3)
            self.psi[r,:,:] += ev * (1-r/4) * rec * (1 - self.crystal[r,:,:]*0.8)

        # 3. Heal (Void Gravity)
        lapV = bk.laplacian_2d(self.V)
        recovery = 0.3 * (1.0 - self.V) # Fast global recovery for test stability
        self.V = bk.clamp(self.V + self.healing_rate * (lapV + recovery) / (1 + self.knots*5 + self.crystal*10), 0, 1)

        # 4. Echo
        delta = self.V - V_before
        self.echo = self.echo * self.echo_damping + delta * (1-self.echo_damping)
        self.V = bk.clamp(self.V + self.echo * self.echo_coupling, 0, 1)

        # 5. Geodesic Diffusion
        self.psi = (1-self.alpha)*self.psi + self.alpha*bk.curvature_diffusion(self.psi, self.R, self.knots, self.crystal)
        self.psi *= (0.95 + 0.04*self.V) * (0.90 if idle else 1.0)
        amp = bk.abs(self.psi)

        # 6. Stress + Knots
        gv = bk.gradient_magnitude(self.V); ga = bk.gradient_magnitude(amp)
        self.stress = bk.clamp((gv+ga*0.5).sum(-1) + 0.05*bk.laplacian_2d(bk.expand_dim((gv+ga*0.5).sum(-1),-1)).reshape(self.H,self.W), 0, 100)
        st3 = bk.expand_dim(self.stress, -1)
        self.knots = bk.clamp(self.knots + bk.maximum(st3-self.knot_threshold, 0) * bk.maximum(0.8-self.V, 0)*0.1, 0, 10)

        # 7. Crystallization + Erosion
        self.crystal = bk.clamp(self.crystal*0.998 + bk.maximum(self.knots-self.crystal_threshold, 0)*0.1, 0, 1)
        elig = bk.maximum(self.V - 0.9, 0)
        self.knots = bk.maximum(self.knots - self.erosion_rate * elig * self.knots, 0)
        self.knots = bk.maximum(
            self.knots - (0.02 * idle) * self.knots,
            0
        )

        # 8. Curvature
        self.R = bk.clamp(self.R + st3*amp*self.plasticity + 0.005*(1-self.R), 0.1, 10)

        # Action (Readout)
        sc = []
        for c in range(self.num_actions):
            sc.append( ((1-self.V[:,:,c]) * amp[:,:,c] * (1+self.knots[:,:,c]*3) * (self.stress+0.01) * (1+self.crystal[:,:,c]*5)).sum() )
        return int(np.argmax(sc))

def run_tests():
    print("=" * 70); print("  VOID RESONANCE v2 FINAL: Physics-Native Extensions"); print("=" * 70)
    bk = HybridBackend()
    s_A = np.array([1,0,0,0,0,0,0,0]); s_B = np.array([0,0,1,0,0,0,0,0]); empty = np.zeros(8)

    # TEST 1: Resonance Cascade
    e1 = VoidResonanceV2(resonance_range=0.5); e2 = VoidResonanceV2(resonance_range=0)
    for _ in range(50): e1.step(s_A); e2.step(s_A)
    pass1 = e1.V[5:,:,0].mean() < e2.V[5:,:,0].mean()
    print(f"[PASS={pass1}] T1: Resonance Cascade (Phantom wounds in Ch0)")

    # TEST 2: Crystallization
    e3 = VoidResonanceV2(knot_threshold=0.3, crystal_threshold=0.1)
    for _ in range(80): e3.step(s_A)
    pass2 = e3.crystal[:,:,0].sum() > 1.0
    print(f"[PASS={pass2}] T2: Crystallization (Phase state in Ch0)")

    # TEST 3: Scar Erosion
    e4 = VoidResonanceV2(knot_threshold=0.3, erosion_rate=0.5)
    e5 = VoidResonanceV2(knot_threshold=0.3, erosion_rate=0.0)
    for _ in range(50): e4.step(s_A); e5.step(s_A)
    k4 = e4.knots.sum(); k5 = e5.knots.sum()
    for _ in range(100): e4.step(empty); e5.step(empty)
    pass3 = (k4 - e4.knots.sum()) > (k5 - e5.knots.sum())
    print(f"[PASS={pass3}] T3: Scar Erosion (Competitive Forgetting)")

    # TEST 4: Wound Echo
    e_echo = VoidResonanceV2(echo_coupling=0.1); e_noecho = VoidResonanceV2(echo_coupling=0)
    for _ in range(30): e_echo.step(s_A); e_noecho.step(s_A)
    v_ec = []; v_ne = []
    for _ in range(40): e_echo.step(empty); e_noecho.step(empty); v_ec.append(e_echo.V[:,:,0].mean()); v_ne.append(e_noecho.V[:,:,0].mean())
    pass4 = np.var(v_ec) > np.var(v_ne)
    print(f"[PASS={pass4}] T4: Wound Echo (Post-healing ringing)")

    # TEST 5: Holographic Recall
    e_h = VoidResonanceV2(); s_AB = np.array([1,1,0,0,0,0,0,0])
    for _ in range(50): e_h.step(s_AB, np.array([0.7, -0.7, 0,0,0,0,0,0]))
    e_h.psi *= 0 # Clear wave
    for _ in range(20): e_h.step(s_A, np.array([0.7, 0,0,0,0,0,0,0]))
    a = np.abs(e_h.psi); pass5 = a[:,:,1].mean() > a[:,:,2].mean() * 3
    print(f"[PASS={pass5}] T5: Holographic Recall (Ch1 emergent)")

    # TEST 6: Behavioral Differentiation
    e_b = VoidResonanceV2(knot_threshold=1.2)
    aA = [e_b.step(s_A) for _ in range(80)]; aB = [e_b.step(s_B) for _ in range(80)]
    pass6 = Counter(aA).most_common(1)[0][0] != Counter(aB).most_common(1)[0][0]
    print(f"[PASS={pass6}] T6: Behavioral Differentiation (A={Counter(aA).most_common(1)[0][0]} B={Counter(aB).most_common(1)[0][0]})")

    print("=" * 70); print(f"FINAL RESULT: {sum([pass1,pass2,pass3,pass4,pass5,pass6])}/6 PASS"); print("=" * 70)

if __name__ == "__main__": run_tests()
