# plot_utils.py

import matplotlib.pyplot as plt


def plot_single_synapse_results(results):
    pre_spikes = results["pre_spikes"]
    I_trace = results["I_trace"]
    V_trace = results["V_trace"]
    post_spikes = results["post_spikes"]
    G_trace = results["G_trace"]
    neuron_type = results["neuron_type"]

    fig, axes = plt.subplots(5, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(pre_spikes)
    axes[0].set_ylabel("Pre spike")

    axes[1].plot(I_trace)
    axes[1].set_ylabel("I_syn")

    axes[2].plot(V_trace)
    axes[2].set_ylabel("Vmem")

    axes[3].plot(post_spikes)
    axes[3].set_ylabel("Post spike")

    axes[4].plot(G_trace)
    axes[4].set_ylabel("G")
    axes[4].set_xlabel("Time")

    fig.suptitle(f"Single Synapse Experiment: {neuron_type.upper()}")
    fig.tight_layout()

    return fig


def plot_multi_terminal_results(results):
    I_input = results["I_input"]
    gates_trace = results["gates_trace"]
    Vth_trace = results["Vth_trace"]
    V_trace = results["V_trace"]
    spike_trace = results["spike_trace"]

    fig, axes = plt.subplots(5, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(I_input)
    axes[0].set_ylabel("I_in")

    axes[1].plot(gates_trace[:, 0], label="gate 1")
    axes[1].plot(gates_trace[:, 1], label="gate 2")
    axes[1].plot(gates_trace[:, 2], label="gate 3")
    axes[1].set_ylabel("Gates")
    axes[1].legend()

    axes[2].plot(Vth_trace)
    axes[2].set_ylabel("Vth_eff")

    axes[3].plot(V_trace)
    axes[3].set_ylabel("Vmem")

    axes[4].plot(spike_trace)
    axes[4].set_ylabel("Spike")
    axes[4].set_xlabel("Time")

    fig.suptitle("Multi-terminal STL Neuron Experiment")
    fig.tight_layout()

    return fig


# plot_utils.py 맨 아래에 추가

import matplotlib.pyplot as plt
import numpy as np


def plot_signal_generation_results(results):
    """
    target waveform과 readout prediction 비교.
    """
    y_target = results["y_target"]
    y_hat = results["y_hat"]
    split = results["split"]

    neuron_type = results["neuron_type"]
    target_type = results["target_type"]
    train_mse = results["train_mse"]
    test_mse = results["test_mse"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(y_target, label="target")
    axes[0].plot(y_hat, label="prediction")
    axes[0].axvline(split, linestyle="--")
    axes[0].set_ylabel("Output")
    axes[0].legend()
    axes[0].set_title(
        f"{neuron_type.upper()} reservoir - {target_type} "
        f"(train MSE={train_mse:.4f}, test MSE={test_mse:.4f})"
    )

    # spike raster
    spikes = results["spikes"]
    spike_t, neuron_idx = np.where(spikes > 0)

    axes[1].scatter(spike_t, neuron_idx, s=2)
    axes[1].axvline(split, linestyle="--")
    axes[1].set_ylabel("Neuron index")
    axes[1].set_title("Spike raster")

    # filtered trace 일부만 표시
    traces = results["traces"]
    n_show = min(10, traces.shape[1])

    for i in range(n_show):
        axes[2].plot(traces[:, i], alpha=0.8)

    axes[2].axvline(split, linestyle="--")
    axes[2].set_ylabel("Filtered trace")
    axes[2].set_xlabel("Time")
    axes[2].set_title(f"Filtered spike traces, first {n_show} neurons")

    fig.tight_layout()

    return fig


def plot_signal_task_comparison(all_results):
    """
    target별, neuron type별 test MSE bar plot.
    """
    labels = []
    test_mse = []

    for result in all_results:
        label = f"{result['target_type']}\n{result['neuron_type']}"
        labels.append(label)
        test_mse.append(result["test_mse"])

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.bar(labels, test_mse)
    ax.set_ylabel("Test MSE")
    ax.set_title("Signal generation task comparison")

    fig.tight_layout()

    return fig


def plot_multiterminal_diagnostics(results, neuron_index=0):
    """
    multi_stl reservoir에서 특정 neuron 하나의
    current, gate inputs, threshold, voltage, spike를 확인.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    neuron_type = results["neuron_type"]

    if neuron_type != "multi_stl":
        print("plot_multiterminal_diagnostics only works for neuron_type='multi_stl'")
        return None

    currents = results["currents"]
    gates = results["gates"]
    thresholds = results["thresholds"]
    voltages = results["voltages"]
    spikes = results["spikes"]

    fig, axes = plt.subplots(5, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(currents[:, neuron_index])
    axes[0].set_ylabel("I_in")
    axes[0].set_title(f"Multi-terminal diagnostics: neuron {neuron_index}")

    n_gates = gates.shape[2]
    for g in range(n_gates):
        axes[1].plot(gates[:, neuron_index, g], label=f"gate {g}")
    axes[1].set_ylabel("Gate inputs")
    axes[1].legend()

    axes[2].plot(thresholds[:, neuron_index])
    axes[2].set_ylabel("Vth_eff")

    axes[3].plot(voltages[:, neuron_index])
    axes[3].set_ylabel("Vmem")

    axes[4].plot(spikes[:, neuron_index])
    axes[4].set_ylabel("Spike")
    axes[4].set_xlabel("Time")

    fig.tight_layout()

    return fig