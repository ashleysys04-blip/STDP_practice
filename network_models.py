# network_models.py

import numpy as np

from neuron_models import LIFNeuron, STLInspiredNeuron, MultiTerminalSTLNeuron


def circular_distance(x, center):
    return ((x - center + 0.5) % 1.0) - 0.5


class SpikingReservoir:
    def __init__(
        self,
        n_neurons=50,
        neuron_type="lif",
        dt=1.0,
        seed=0,
        max_current=0.25,
        bias_current=0.02,
        tuning_width=0.08,
        n_gates=3,
    ):
        self.n_neurons = n_neurons
        self.neuron_type = neuron_type
        self.dt = dt

        self.max_current = max_current
        self.bias_current = bias_current
        self.tuning_width = tuning_width

        self.n_gates = n_gates

        self.rng = np.random.default_rng(seed)

        # main current용 preferred phase
        self.preferred_phases = self.rng.uniform(0.0, 1.0, size=n_neurons)

        # neuron별 gain / width
        self.gains = self.rng.uniform(0.7, 1.3, size=n_neurons)
        self.widths = self.rng.uniform(
            0.7 * tuning_width,
            1.3 * tuning_width,
            size=n_neurons,
        )

        # multi-terminal gate용 preferred phase
        # shape: (n_neurons, n_gates)
        self.gate_preferred_phases = self.rng.uniform(
            0.0,
            1.0,
            size=(n_neurons, n_gates),
        )

        # gate별 neuron sensitivity
        # shape: (n_neurons, n_gates)
        self.gate_gains = self.rng.uniform(
            0.6,
            1.4,
            size=(n_neurons, n_gates),
        )

        self.gate_widths = self.rng.uniform(
            0.08,
            0.20,
            size=(n_neurons, n_gates),
        )

        self.neurons = []
        self._build_neurons()

    def _build_neurons(self):
        self.neurons = []

        for i in range(self.n_neurons):
            if self.neuron_type == "lif":
                neuron = LIFNeuron(
                    V_th=self.rng.uniform(0.08, 0.18),
                    V_reset=0.0,
                    tau_mem=self.rng.uniform(8.0, 25.0),
                    dt=self.dt,
                    leak=True,
                )

            elif self.neuron_type == "stl":
                neuron = STLInspiredNeuron(
                    V_reset=0.0,
                    tau_mem=self.rng.uniform(12.0, 30.0),
                    dt=self.dt,
                    alpha_latch=self.rng.uniform(8.0, 14.0),
                    latch_center=self.rng.uniform(0.35, 0.75),
                    spike_threshold=self.rng.uniform(0.8, 1.2),
                )

            elif self.neuron_type == "multi_stl":
                beta = self.rng.uniform(0.05, 0.20, size=self.n_gates)

                neuron = MultiTerminalSTLNeuron(
                    V_th0=self.rng.uniform(0.8, 1.2),
                    V_reset=0.0,
                    tau_mem=self.rng.uniform(12.0, 30.0),
                    dt=self.dt,
                    beta=beta,
                    alpha_latch=self.rng.uniform(8.0, 14.0),
                    latch_gain=self.rng.uniform(0.15, 0.35),
                    spike_margin=self.rng.uniform(0.2, 0.4),
                )

            else:
                raise ValueError(f"Unknown neuron_type: {self.neuron_type}")

            self.neurons.append(neuron)

    def reset(self):
        for neuron in self.neurons:
            neuron.V = 0.0

    def phase_to_current(self, phase):
        """
        main source/drain input current 역할.
        """
        dist = circular_distance(phase, self.preferred_phases)

        current = (
            self.bias_current
            + self.max_current
            * self.gains
            * np.exp(-(dist ** 2) / (2 * self.widths ** 2))
        )

        return current

    def phase_to_gates(self, phase):
        """
        진짜 multi-terminal gate input 생성.

        return:
            gates: shape (n_neurons, n_gates)

        각 neuron의 각 gate가 서로 다른 preferred phase를 가짐.
        즉 neuron i의 gate 1, gate 2, gate 3가 모두 다른 timing에 반응 가능.
        """
        gates = np.zeros((self.n_neurons, self.n_gates))

        for g in range(self.n_gates):
            dist = circular_distance(phase, self.gate_preferred_phases[:, g])

            gates[:, g] = (
                self.gate_gains[:, g]
                * np.exp(-(dist ** 2) / (2 * self.gate_widths[:, g] ** 2))
            )

        # 0~1 범위로 대략 normalize
        gates = np.clip(gates, 0.0, 1.0)

        return gates

    def step(self, phase):
        currents = self.phase_to_current(phase)
        gates = self.phase_to_gates(phase)

        spikes = np.zeros(self.n_neurons)
        voltages = np.zeros(self.n_neurons)
        thresholds = np.full(self.n_neurons, np.nan)

        for i, neuron in enumerate(self.neurons):
            if self.neuron_type == "multi_stl":
                spike, Vth_eff = neuron.step(currents[i], gates[i])
                thresholds[i] = Vth_eff
            else:
                spike = neuron.step(currents[i])

            spikes[i] = spike
            voltages[i] = neuron.V

        return spikes, currents, voltages, gates, thresholds

    def run(self, phases):
        T = len(phases)

        spikes = np.zeros((T, self.n_neurons))
        currents = np.zeros((T, self.n_neurons))
        voltages = np.zeros((T, self.n_neurons))
        gates = np.zeros((T, self.n_neurons, self.n_gates))
        thresholds = np.full((T, self.n_neurons), np.nan)

        self.reset()

        for t in range(T):
            s, I, V, G, Vth = self.step(phases[t])

            spikes[t] = s
            currents[t] = I
            voltages[t] = V
            gates[t] = G
            thresholds[t] = Vth

        return spikes, currents, voltages, gates, thresholds