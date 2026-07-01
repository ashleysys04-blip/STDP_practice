# plot_utils.py

import matplotlib.pyplot as plt


def plot_single_synapse_results(results):
    """
    Single synapse experiment 결과 plot.
    """

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
    plt.tight_layout()
    plt.show()


def plot_multi_terminal_results(results):
    """
    Multi-terminal STL neuron experiment 결과 plot.
    """

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
    plt.tight_layout()
    plt.show()