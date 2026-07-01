# main.py

from experiments import (
    run_single_synapse_experiment,
    run_multi_terminal_experiment,
)

from plot_utils import (
    plot_single_synapse_results,
    plot_multi_terminal_results,
)


def main():
    # 1. LIF neuron + memristive synapse 실험
    lif_results = run_single_synapse_experiment(
        neuron_type="lif",
        T=300,
        dt=1.0,
        seed=0,
    )
    plot_single_synapse_results(lif_results)

    # 2. STL-inspired neuron + memristive synapse 실험
    stl_results = run_single_synapse_experiment(
        neuron_type="stl",
        T=300,
        dt=1.0,
        seed=0,
    )
    plot_single_synapse_results(stl_results)

    # 3. Multi-terminal STL neuron 실험
    multi_results = run_multi_terminal_experiment(
        T=300,
        seed=0,
    )
    plot_multi_terminal_results(multi_results)


if __name__ == "__main__":
    main()