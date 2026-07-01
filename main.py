# main.py

import matplotlib.pyplot as plt

from experiments import (
    run_signal_generation_experiment,
    run_signal_task_comparison,
)

from plot_utils import (
    plot_signal_generation_results,
    plot_signal_task_comparison,
    plot_multiterminal_diagnostics,
)


def main():
    target_types = ["sin", "rectified_sin", "sawtooth"]
    neuron_types = ["lif", "stl", "multi_stl"]

    for target_type in target_types:
        for neuron_type in neuron_types:
            results = run_signal_generation_experiment(
                neuron_type=neuron_type,
                target_type=target_type,
                n_neurons=80,
                T=1000,
                cycles=3,
                seed=0,
            )

            print(
                f"{neuron_type.upper()} - {target_type}: "
                f"train MSE={results['train_mse']:.4f}, "
                f"test MSE={results['test_mse']:.4f}"
            )

            plot_signal_generation_results(results)

            if neuron_type == "multi_stl" and target_type == "sin":
                plot_multiterminal_diagnostics(results, neuron_index=0)

    comparison_results = run_signal_task_comparison(
        target_types=target_types,
        neuron_types=neuron_types,
        n_neurons=80,
        T=1000,
        cycles=3,
        seed=0,
    )

    plot_signal_task_comparison(comparison_results)

    plt.show()


if __name__ == "__main__":
    main()