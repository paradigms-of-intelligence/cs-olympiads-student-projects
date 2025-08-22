
# Project Overview

`circuit` is a small experimental framework for generating, training, converting and evaluating boolean-gate style neural networks for classification tasks (such as MNIST). The repository contains generator components that write a network architecture, a training engine built on JAX + Optax, helpers to convert the trained network into an AIG-like binary format, and evaluators for both the original 16-gate representation and a converted 2-gate/AIG representation.


---

## Repo layout

```
circuit
├── config.py                  # global parameters
├── converter
│   ├── convert_abc.py         # converts 16-gate binary -> 2-gate AIG-like file
│   ├── converter.py  # (other converter into a different and-inverter format)
├── data
│   ├── extractor.py           # build training/test .txt files from MNIST gz
│   ├── *.gz                   # raw MNIST images and labels
│   ├── training.txt / testdata.txt
├── evaluator
│   ├── evaluator_abc.py       # evaluates the 2-gate/AIG file
│   └── evaluator.py           # (analogous to the standard converter)
├── generator
│   └── generator.py           # creates the network architecture
├── network
│   ├── network.py             # training & inference engine 
├── exec
│   └── (runtime files, outputs)
├── run_network.py
└── README.md                  # this file
```

---

## How to run

1. Create a Python environment with the following (minimum) dependencies:

   * Python 3.10+ (repo shows Python 3.13 bytecode but the code is standard Python 3)
   * jax, jaxlib (with CPU/GPU backend as desired)
   * optax
   * numpy
   * matplotlib (optional, for extractor)
   * coloredlogs

2. Populate `data/` using `data/extractor.py` (it reads MNIST `.gz` files). Example (interactive):

```bash
python data/extractor.py
# choose y/N for prompts if needed
```

3. Run the main entrypoint:

```bash
python run_network.py
```

`run_network.py` does the following (based on `config.py`):

* `initialize_config()` sets environment variables and logging
* call one of the generator functions (by default `generate_network_probabilistic_model()`)
* run training network step (`run_training_sequence()` in `network.py`)
* run converter & evaluator depending on `ABC_FORMAT` flag

---

## Configuration (`config.py`)

Important options you will likely tweak:

* `INPUT_NODES` — defaults to `784` (28×28 images)
* `OUTPUT_NODES` — defaults to `1000` (the code later groups outputs into 10 categories)
* `LAYERS` — list of layer sizes 

Hyperparameters: `EPOCH_COUNT`, `TOTAL_SIZE`, `BATCH_SIZE`, `LEARNING_RATE`, `WEIGHT_DECAY`, `MAX_TEMPERATURE`.

---

## Component responsibilities and I/O (detailed)

Below is a concise table describing the input and output format between each major component. Where the code is ambiguous or inconsistent, the `Notes / Fix` column explains what to change or expect.

### Legend

* `TXT` = ASCII text file
* `BIN` = binary file with a recorded structure
* `Numpy` = `.npy` files saved with `numpy.save` / `jnp.save`

---

### 1) generator → network (file: `exec/network_architecture.txt`)

**Producing component**: `generator/generator.py`.

**Consumed by**: `network.input_network()` in `network/network.py`.

**Expected format (by `network.input_network`)**:

* The file is read sequentially. For each non-input node the code executes: `left, right, prob_id = map(int, file.readline().strip().split())`.
* That means **exactly three** integers per line separated by whitespace, representing `left_node_id right_node_id probability_gate_id` for that gate (the last integer can be used to model convolutional structures).

**Examples** (one gate per line):

```
23 45 0
10 987 0
784 1100 2
```

---

### 2) network → trained network file (`exec/trained_network_16gates.bin`)

**Producing component**: `network.print_network()` writes the trained network binary. **Note:** the code that writes the file writes, for each gate, four 32-bit little-endian signed integers (via `.to_bytes(4, byteorder='little', signed=True)`) in this order:

* gate\_index (an integer selecting one of 16 possible gate types) — 4 bytes
* node id  — 4 bytes
* left connection id — 4 bytes
* right connection id — 4 bytes

The **abc** files work in binary AIGER format.

**Simplified spec for an example AIG**:

```
Header (ASCII):       aig M I L O A\n
Outputs (ASCII):      one integer per line (literal ids) x O
AND-gate data (BIN):  For each AND gate, 2 varints: delta0, delta1
```
Further information: https://fmv.jku.at/aiger/FORMAT.aiger

---

### 4) evaluator\_abc.py (AIG evaluator)

**Inputs**:

The evaluators take the input format produced by the two converters respectively and evalutate in both cases an AIG network.

**Output**: 
The evaluator prints out the final accuracy of the AIG network printing out the percentage of the testdata correctly predicted.

**Debug evaluation**:
A DEBUG parameter is present the ```network/network.py``` when set to ```True``` it enables continual accuracy printing over training and test data throughout the various training epochs. 

---

## Example end-to-end sequence (recommended)

1. `python data/extractor.py` → produces `data/training.txt` and `data/testdata.txt`.
2. `python run_network.py` does:

   * `initialize_config()`
   * `generate_network_probabilistic_model()` (or the other generator)
   * run `network.run_training_sequence()` (provided)
   * run converter+evaluator, transforming the logic gates into AIG 

> To debug, run steps individually in REPL and inspect intermediate files:
>
> * Open `exec/network_architecture.txt` and verify line counts / format.
> * After training, inspect `exec/trained_network_16gates.bin` header and size.
> * Run `python converter/convert_abc.py` manually and check that `exec/trained_network_2gates.bin` is produced.
---