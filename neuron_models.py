import numpy as np
import matplotlib.pyplot as plt

class LIFNeuron:
    def __init__(
        self,
        V_th=1.0,
        V_reset=0.0,
        tau_mem=20.0,
        dt=1.0,
        leak=True,
    ):
        self.V_th = V_th
        self.V_reset = V_reset
        self.tau_mem = tau_mem
        self.dt = dt
        self.leak = leak
        self.V = 0.0

    def step(self, I):
        """
        LIF neuron update.
        """
        if self.leak:
            dV = (-self.V + I) * self.dt / self.tau_mem
        else:
            dV = I

        self.V += dV

        if self.V >= self.V_th:
            self.V = self.V_reset
            return 1
        else:
            return 0

class STLInspiredNeuron:
    def __init__(
        self,
        V_th=1.0,
        V_reset=0.0,
        tau_mem=20.0,
        dt=1.0,
        alpha_latch=8.0,
        latch_center=0.8,
        spike_threshold=1.2,
    ):
        self.V_th = V_th
        self.V_reset = V_reset
        self.tau_mem = tau_mem
        self.dt = dt
        self.alpha_latch = alpha_latch
        self.latch_center = latch_center
        self.spike_threshold = spike_threshold
        self.V = 0.0

    def latch_feedback(self, V):
        """
        threshold 근처에서 급격히 증가하는 positive feedback.
        sigmoid로 latch-like abrupt transition을 근사.
        """
        return 1.0 / (1.0 + np.exp(-self.alpha_latch * (V - self.latch_center)))

    def step(self, I):
        leak_term = -self.V / self.tau_mem
        input_term = I
        latch_term = self.latch_feedback(self.V) * 0.2

        self.V += self.dt * (leak_term + input_term + latch_term)

        if self.V >= self.spike_threshold:
            self.V = self.V_reset
            return 1

        return 0


class MultiTerminalSTLNeuron:
    def __init__(
        self,
        V_th0=1.0,
        V_reset=0.0,
        tau_mem=20.0,
        dt=1.0,
        beta=None,
        alpha_latch=10.0,
        latch_gain=0.25,
        spike_margin=0.3,
    ):
        """
        Multi-terminal STL-inspired neuron.

        I_in:
            main input current

        gates:
            multiple terminal voltages, shape (n_gates,)

        beta:
            gate-to-threshold coupling strength.
            beta[i]가 클수록 gate i가 threshold를 강하게 낮춤.
        """
        self.V_th0 = V_th0
        self.V_reset = V_reset
        self.tau_mem = tau_mem
        self.dt = dt

        self.alpha_latch = alpha_latch
        self.latch_gain = latch_gain
        self.spike_margin = spike_margin

        self.V = 0.0

        if beta is None:
            beta = np.array([0.15, 0.15, 0.15])

        self.beta = np.array(beta, dtype=float)

    def effective_threshold(self, gates):
        gates = np.array(gates, dtype=float)

        # gate input이 많아질수록 threshold가 낮아짐
        V_th_eff = self.V_th0 - np.dot(self.beta, gates)

        # 너무 낮아져서 말도 안 되는 firing이 생기는 것 방지
        V_th_eff = np.clip(V_th_eff, 0.2, 2.0)

        return V_th_eff

    def latch_feedback(self, V, V_th_eff):
        """
        threshold 근처에서 급격히 켜지는 latch-like positive feedback.
        """
        return 1.0 / (1.0 + np.exp(-self.alpha_latch * (V - V_th_eff)))

    def step(self, I_in, gates):
        V_th_eff = self.effective_threshold(gates)

        leak = -self.V / self.tau_mem
        latch = self.latch_gain * self.latch_feedback(self.V, V_th_eff)

        self.V += self.dt * (leak + I_in + latch)

        if self.V >= V_th_eff + self.spike_margin:
            self.V = self.V_reset
            return 1, V_th_eff

        return 0, V_th_eff