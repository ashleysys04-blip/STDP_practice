# experiments.py

import numpy as np

from neuron_models import LIFNeuron, STLInspiredNeuron, MultiTerminalSTLNeuron
from synapse_models import MemristiveSynapse


def run_single_synapse_experiment(
    neuron_type="lif",
    T=300,
    dt=1.0,
    seed=0,
):
    """
    Single pre-synaptic spike train -> one synapse -> one neuron 실험.

    neuron_type:
        "lif" : LIFNeuron 사용
        "stl" : STLInspiredNeuron 사용
    """

    np.random.seed(seed)

    # 입력 spike train
    pre_spikes = np.zeros(T)
    pre_spikes[[20, 40, 60, 80, 100, 140, 180, 220, 260]] = 1

    # synapse
    syn = MemristiveSynapse(G_init=0.6)

    # neuron 선택
    if neuron_type == "lif":
        neuron = LIFNeuron(
            V_th=0.15,
            tau_mem=10.0,
            dt=dt,
        )

    elif neuron_type == "stl":
        neuron = STLInspiredNeuron(
            tau_mem=20.0,
            alpha_latch=10.0,
            latch_center=0.6,
            spike_threshold=1.0,
            dt=dt,
        )

    else:
        raise ValueError(f"Unknown neuron_type: {neuron_type}")

    V_trace = []
    post_spikes = []
    G_trace = []
    I_trace = []

    for t in range(T):
        pre = pre_spikes[t]

        if pre == 1:
            syn.update_pre(t)

        I = syn.current(pre)
        post = neuron.step(I)

        if post == 1:
            syn.update_post(t)

        V_trace.append(neuron.V)
        post_spikes.append(post)
        G_trace.append(syn.G)
        I_trace.append(I)

    results = {
        "T": T,
        "dt": dt,
        "neuron_type": neuron_type,
        "pre_spikes": pre_spikes,
        "I_trace": np.array(I_trace),
        "V_trace": np.array(V_trace),
        "post_spikes": np.array(post_spikes),
        "G_trace": np.array(G_trace),
    }

    return results


def run_multi_terminal_experiment(
    T=300,
    seed=0,
):
    """
    MultiTerminalSTLNeuron 테스트용 실험.

    여러 gate 입력이 effective threshold를 낮추고,
    동시에 들어올 때 spike가 더 잘 발생하는지 확인.
    """

    np.random.seed(seed)

    I_input = np.zeros(T)
    I_input[[50, 100, 150, 200, 250]] = 0.2

    gates_trace = np.zeros((T, 3))

    # gate 1, 2, 3가 서로 다른 시간에 들어오는 상황
    gates_trace[40:80, 0] = 1.0
    gates_trace[90:130, 1] = 1.0
    gates_trace[140:180, 2] = 1.0

    # 세 gate가 동시에 들어오는 구간
    gates_trace[220:260, :] = 1.0

    neuron = MultiTerminalSTLNeuron(
        V_th0=1.0,
        beta=[0.15, 0.15, 0.15],
        tau_mem=20,
        alpha_latch=12,
    )

    V_trace = []
    spike_trace = []
    Vth_trace = []

    for t in range(T):
        spike, Vth_eff = neuron.step(I_input[t], gates_trace[t])

        V_trace.append(neuron.V)
        spike_trace.append(spike)
        Vth_trace.append(Vth_eff)

    results = {
        "T": T,
        "I_input": I_input,
        "gates_trace": gates_trace,
        "Vth_trace": np.array(Vth_trace),
        "V_trace": np.array(V_trace),
        "spike_trace": np.array(spike_trace),
    }

    return results