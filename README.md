# STDP Practice: Spiking Neural Network Toy Models

A small Python project for learning how spiking neural networks (SNNs), STDP-like synapses, and device-inspired neuron models can be simulated at the code level.

This repository starts from a simple question:

> If neuromorphic devices can behave like neurons or synapses, can we express those device behaviors in Python and test them as an SNN model?

The answer here is **yes**. This project implements several simplified neuron and synapse models, builds a small spiking reservoir network, and trains a readout layer to reproduce temporal waveforms such as sine, rectified sine, and sawtooth signals.

---

## 1. What this project is about

Conventional artificial neural networks usually use continuous-valued activations such as ReLU or sigmoid. A **spiking neural network (SNN)** is different: neurons communicate using discrete events called **spikes**.

Instead of asking only:

```text
How large is the activation value?
```

an SNN also asks:

```text
When did the neuron spike?
How often did it spike?
Which neurons spiked together?
```

This project explores three related ideas:

1. **Spiking neuron models**  
   Simple neurons that accumulate input over time and generate spikes.

2. **Memristive STDP synapses**  
   Synapses whose conductance changes depending on the timing of pre- and post-synaptic spikes.

3. **Multi-terminal STL-inspired neurons**  
   Device-inspired neurons where multiple gate terminals modulate the effective firing threshold.

The project is not a full biological brain simulation. It is a compact educational simulation for understanding how device-level behavior can be abstracted into SNN-level computation.

---

## 2. Repository structure

```text
STDP_practice/
├── main.py
├── experiments.py
├── network_models.py
├── neuron_models.py
├── synapse_models.py
└── plot_utils.py
```

### `main.py`

Entry point of the project.

It runs the signal-generation experiments for different neuron types and target waveforms:

```python
neuron_types = ["lif", "stl", "multi_stl"]
target_types = ["sin", "rectified_sin", "sawtooth"]
```

For each combination, it trains a readout layer and plots the result.

---

### `neuron_models.py`

Contains neuron models.

#### `LIFNeuron`

A basic leaky integrate-and-fire neuron.

It accumulates input current as membrane potential. If the membrane potential crosses a threshold, the neuron emits a spike and resets.

Conceptually:

```text
input current → membrane potential increases → threshold crossing → spike
```

#### `STLInspiredNeuron`

A latch-like spiking neuron inspired by abrupt switching behavior.

Compared with a regular LIF neuron, this model includes a positive-feedback-like term that becomes strong near threshold. This makes the spike transition more abrupt.

Conceptually:

```text
input current → membrane potential rises → latch-like feedback turns on → spike
```

#### `MultiTerminalSTLNeuron`

A multi-terminal STL-inspired neuron.

This neuron receives:

```text
main input current: I_in
multiple gate inputs: Vg1, Vg2, Vg3, ...
```

The gate inputs modulate the effective firing threshold:

```text
Vth_eff = Vth0 - beta1*Vg1 - beta2*Vg2 - beta3*Vg3
```

If the gate inputs are active, the threshold becomes lower, and the neuron can spike more easily.

This is the main device-inspired idea in the project.

---

### `synapse_models.py`

Contains synapse models.

#### `StaticSynapse`

A fixed-weight synapse.

It simply converts a pre-synaptic spike into current:

```text
I = weight × pre_spike
```

The weight does not change.

#### `MemristiveSynapse`

A simplified memristive synapse with STDP-like plasticity.

It has a conductance value `G`. The synaptic current is:

```text
I = G × pre_spike
```

The conductance `G` can change depending on spike timing:

```text
pre before post → potentiation → G increases
post before pre → depression → G decreases
```

This is a simplified model of spike-timing-dependent plasticity, or STDP.

---

### `network_models.py`

Contains the network-level model.

#### `SpikingReservoir`

A small spiking reservoir network made of many spiking neurons.

The reservoir can use different neuron types:

```python
"lif"        # LIFNeuron reservoir
"stl"        # STLInspiredNeuron reservoir
"multi_stl"  # MultiTerminalSTLNeuron reservoir
```

Each neuron has its own preferred phase and responds differently over time. The reservoir produces a matrix of spikes:

```text
spikes: shape (time, number_of_neurons)
```

For the multi-terminal version, each neuron also receives multiple gate inputs:

```text
gates: shape (time, number_of_neurons, number_of_gates)
```

This makes each neuron behave like a slightly different multi-terminal device.

---

### `experiments.py`

Contains experiment functions.

Important functions include:

#### `make_target_signal()`

Generates target waveforms:

```python
"sin"
"rectified_sin"
"sawtooth"
```

#### `exponential_filter_spikes()`

Raw spikes are very sparse. This function converts spike trains into smoother traces using an exponential filter.

Conceptually:

```text
spike:          0 0 1 0 0 0 1 ...
filtered trace: 0 0 1 0.9 0.81 ...
```

The filtered traces are used as features for the readout layer.

#### `train_readout_ridge()`

Trains a linear readout using ridge regression.

The reservoir itself is fixed. The learned part is the readout weight.

```text
filtered spike traces → linear readout → predicted waveform
```

#### `run_signal_generation_experiment()`

Runs one waveform-generation task.

Example:

```python
run_signal_generation_experiment(
    neuron_type="multi_stl",
    target_type="sin",
    n_neurons=80,
    T=1000,
    cycles=3,
)
```

This asks the multi-terminal STL reservoir to generate a sine-like output through a trained readout.

---

### `plot_utils.py`

Contains plotting functions.

The plots show:

1. Target waveform vs predicted waveform
2. Spike raster plot
3. Filtered spike traces
4. Test MSE comparison
5. Multi-terminal diagnostic plots

The diagnostic plot is especially useful for the multi-terminal model. It shows:

```text
main input current
multiple gate inputs
effective threshold
membrane voltage
output spike
```

This helps verify whether the gate terminals actually modulate the firing threshold.

---

## 3. How the signal-generation task works

The main experiment is a temporal waveform fitting task.

The target can be:

```text
sine wave
rectified sine wave
sawtooth wave
```

The model does not directly output a continuous waveform. Instead, the process is:

```text
1. A spiking reservoir generates spikes over time.
2. Spikes are filtered into continuous traces.
3. A linear readout is trained to map those traces to the target waveform.
4. The predicted waveform is compared with the target waveform.
```

In diagram form:

```text
time / phase input
        ↓
spiking reservoir
        ↓
spike trains
        ↓
filtered spike traces
        ↓
trained linear readout
        ↓
predicted waveform
```

The learning happens in the **readout layer**, not inside the reservoir.

This is similar to reservoir computing: the internal dynamics are fixed, and only the output layer is trained.

---

## 4. Neuron types compared

### 1. LIF reservoir

Uses conventional leaky integrate-and-fire neurons.

This is the baseline.

### 2. STL-inspired reservoir

Uses neurons with latch-like positive feedback.

This tests whether abrupt switching dynamics can create useful temporal features.

### 3. Multi-terminal STL reservoir

Uses neurons with multiple gate-terminal inputs.

This tests whether threshold modulation through multiple terminals can enrich the reservoir dynamics.

The multi-terminal neuron receives:

```text
I_in(t): main input current
Vg1(t), Vg2(t), Vg3(t): gate-terminal inputs
```

and computes:

```text
Vth_eff(t) = Vth0 - beta · gates(t)
```

When gate inputs increase, the effective threshold decreases, making spikes more likely.

---

## 5. How to run

### 1. Clone the repository

```bash
git clone https://github.com/ashleysys04-blip/STDP_practice.git
cd STDP_practice
```

### 2. Install dependencies

The project only requires NumPy and Matplotlib.

```bash
pip install numpy matplotlib
```

### 3. Run the project

```bash
python main.py
```

This will run waveform-generation experiments for:

```text
neuron types: lif, stl, multi_stl
targets: sin, rectified_sin, sawtooth
```

and display several plots.

---

## 6. What to look for in the plots

### Target vs prediction

This plot shows whether the readout successfully learned the target waveform.

Good result:

```text
prediction overlaps well with target
```

Poor result:

```text
prediction is flat, noisy, or phase-shifted
```

---

### Spike raster

This shows when each neuron spikes.

Good reservoir behavior usually means that different neurons spike at different times, creating diverse temporal features.

---

### Filtered spike traces

These traces are the actual features used by the readout layer.

If the traces are too flat or too sparse, the readout will not have enough information to reconstruct the target waveform.

---

### Multi-terminal diagnostic plot

For `multi_stl`, this plot shows whether the gate terminals are doing something meaningful.

Expected behavior:

```text
gate inputs increase
        ↓
effective threshold decreases
        ↓
membrane voltage reaches firing condition more easily
        ↓
spike occurs
```

---

## 7. What is actually being learned?

This project contains two different learning-related ideas.

### STDP-like local plasticity

The `MemristiveSynapse` can update its conductance based on spike timing.

This is local synaptic adaptation.

### Readout learning

The waveform-generation task trains a linear readout using ridge regression.

This is the main task-level learning used in the current reservoir experiment.

So, in the current waveform task:

```text
reservoir neuron dynamics: fixed
readout weights: learned
```

---

## 8. Current limitations

This project is intentionally simple.

Current limitations include:

1. The reservoir is not yet a fully recurrent SNN.
2. The readout is linear.
3. The device models are compact toy models, not calibrated physical compact models.
4. The multi-terminal behavior is modeled as threshold modulation, not full TCAD or SPICE simulation.
5. The task is waveform fitting, not large-scale classification.

These limitations are also useful starting points for future work.

---

## 9. Possible next steps

Potential extensions:

1. Add recurrent connections inside the reservoir.
2. Compare spike count and energy proxy across neuron types.
3. Add noise and test robustness.
4. Use more realistic device non-idealities.
5. Train on real temporal datasets.
6. Implement multi-terminal synaptic plasticity.
7. Export results automatically to a `results/` folder.
8. Add command-line arguments for neuron type and target type.

Example future command:

```bash
python main.py --neuron_type multi_stl --target sawtooth
```

---

## 10. Why this project matters

The main point of this project is not to claim that a physical device has already been fabricated.

Instead, the goal is to ask:

> If a device has latch-like switching or multi-terminal threshold modulation, what kind of SNN-level computation could it support?

By expressing device-inspired behavior in Python, we can test ideas quickly before moving to heavier tools such as SPICE, Verilog-A, or TCAD.

This makes the project a bridge between:

```text
device-level intuition
        ↓
compact computational model
        ↓
SNN-level task behavior
```

---

## 11. Minimal conceptual summary

For a beginner, the whole project can be summarized as:

```text
We make artificial spiking neurons in Python.
Some neurons behave like ordinary LIF neurons.
Some neurons behave more like latch-based devices.
Some neurons have multiple gate inputs that change their firing threshold.
We connect many such neurons into a reservoir.
Then we train a simple readout to reconstruct waveforms such as sine and sawtooth signals.
```

