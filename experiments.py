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


# experiments.py 맨 아래에 추가

import numpy as np

from network_models import SpikingReservoir


def make_target_signal(kind="sin", T=1000, cycles=3):
    """
    target waveform 생성.

    kind:
        "sin"
        "rectified_sin"
        "sawtooth"
    """
    phase = (np.arange(T) / T * cycles) % 1.0

    if kind == "sin":
        y = np.sin(2 * np.pi * phase)

    elif kind == "rectified_sin":
        y = np.maximum(0.0, np.sin(2 * np.pi * phase))

    elif kind == "sawtooth":
        y = 2.0 * phase - 1.0

    else:
        raise ValueError(f"Unknown target kind: {kind}")

    return phase, y


def exponential_filter_spikes(spikes, tau=20.0, dt=1.0):
    """
    raw spike를 readout에 바로 쓰면 너무 sparse해서
    exponential trace로 바꿔줌.

    spikes: shape (T, N)
    return: traces, shape (T, N)
    """
    T, N = spikes.shape
    traces = np.zeros_like(spikes, dtype=float)

    decay = np.exp(-dt / tau)

    for t in range(1, T):
        traces[t] = decay * traces[t - 1] + spikes[t]

    return traces


def train_readout_ridge(X, y, reg=1e-3):
    """
    Ridge regression으로 readout weight 학습.

    X: shape (T, N)
    y: shape (T,)
    """
    # bias term 추가
    X_aug = np.column_stack([X, np.ones(X.shape[0])])

    I = np.eye(X_aug.shape[1])
    I[-1, -1] = 0.0  # bias는 regularization 안 함

    w = np.linalg.solve(X_aug.T @ X_aug + reg * I, X_aug.T @ y)

    return w


def predict_readout(X, w):
    X_aug = np.column_stack([X, np.ones(X.shape[0])])
    return X_aug @ w


def run_signal_generation_experiment(
    neuron_type="lif",
    target_type="sin",
    n_neurons=80,
    T=1000,
    cycles=3,
    dt=1.0,
    seed=0,
    train_ratio=0.7,
    trace_tau=25.0,
    reg=1e-3,
):
    """
    Spiking reservoir를 이용해서 target waveform을 fitting하는 task.

    학습되는 것:
        readout weight

    고정되는 것:
        reservoir neuron dynamics
    """

    # 1. target 생성
    phases, y_target = make_target_signal(
        kind=target_type,
        T=T,
        cycles=cycles,
    )

    # 2. reservoir 생성
    reservoir = SpikingReservoir(
        n_neurons=n_neurons,
        neuron_type=neuron_type,
        dt=dt,
        seed=seed,
    )

    # 3. reservoir 실행
    spikes, currents, voltages, gates, thresholds = reservoir.run(phases)

    # 4. spike filtering
    traces = exponential_filter_spikes(
        spikes,
        tau=trace_tau,
        dt=dt,
    )

    # 5. train/test split
    split = int(T * train_ratio)

    X_train = traces[:split]
    y_train = y_target[:split]

    X_test = traces[split:]
    y_test = y_target[split:]

    # 6. readout 학습
    w_out = train_readout_ridge(
        X_train,
        y_train,
        reg=reg,
    )

    # 7. prediction
    y_hat = predict_readout(traces, w_out)

    y_hat_train = y_hat[:split]
    y_hat_test = y_hat[split:]

    # 8. metric
    train_mse = np.mean((y_hat_train - y_train) ** 2)
    test_mse = np.mean((y_hat_test - y_test) ** 2)

    results = {
        "neuron_type": neuron_type,
        "target_type": target_type,
        "n_neurons": n_neurons,
        "T": T,
        "cycles": cycles,
        "dt": dt,
        "train_ratio": train_ratio,
        "split": split,
        "phases": phases,
        "y_target": y_target,
        "y_hat": y_hat,
        "spikes": spikes,
        "traces": traces,
        "currents": currents,
        "voltages": voltages,
        "gates": gates,
        "thresholds": thresholds,
        "w_out": w_out,
        "train_mse": train_mse,
        "test_mse": test_mse,
    }

    return results


def run_signal_task_comparison(
    target_types=None,
    neuron_types=None,
    n_neurons=80,
    T=1000,
    cycles=3,
    dt=1.0,
    seed=0,
):
    """
    LIF vs STL, target별 MSE 비교.
    """
    if target_types is None:
        target_types = ["sin", "rectified_sin", "sawtooth"]

    if neuron_types is None:
        neuron_types = ["lif", "stl"]

    all_results = []

    for target_type in target_types:
        for neuron_type in neuron_types:
            result = run_signal_generation_experiment(
                neuron_type=neuron_type,
                target_type=target_type,
                n_neurons=n_neurons,
                T=T,
                cycles=cycles,
                dt=dt,
                seed=seed,
            )
            all_results.append(result)

    return all_results