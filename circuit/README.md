# circuit — README

## Project Overview

`circuit` is a small experimental framework for generating, training, converting and evaluating boolean-gate style neural networks (a hybrid probabilistic / gate-based architecture). The repository contains generator components that write a network architecture, a training engine built on JAX + Optax, helpers to convert the trained network into an AIG-like binary/text format, and evaluators for both the original 16-gate representation and a converted 2-gate/AIG representation.


---

## Repo layout

```
circuit
├── config.py
├── converter
│   ├── convert_abc.py         # converts 16-gate binary -> 2-gate AIG-like file
│   ├── converter_standard.py  # (other converter, not shown here)
├── data
│   ├── extractor.py           # build training/test .txt files from MNIST gz
│   ├── *.gz                   # raw MNIST images and labels
│   ├── training.txt / testdata.txt
│   └── *.values.npy / *.answers.npy
├── evaluator
│   ├── evaluator_abc.py       # evaluates the 2-gate/AIG file
│   └── evaluator.py           # (evaluator for 16-gate format)
├── generator
│   └── generator.py           # writes network_architecture.txt
├── network
│   ├── network.py             # training & inference engine (JAX + Optax)
│   └── network_alternative.py  # alternative procedures referenced from run_network
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
* run alternative network step (`an()` in `network_alternative.py`)
* run converter & evaluator depending on `ABC_FORMAT` flag

---

## Configuration (`config.py`)

Important options you will likely tweak:

* `INPUT_NODES` — defaults to `784` (28×28 images)
* `OUTPUT_NODES` — defaults to `1000` (the code later groups outputs into 10 categories)
* `LAYERS` — list of layer sizes (defaults in your file are `[784, 3000, 3000, 1000]`)
* `NETWORK_ARCHITECTURE_FILE` — path where the generator writes `network_architecture.txt`
* `TRAINING_DATA_FILE` / `TESTDATA_DATA_FILE` — paths to training/test text files
* `TRAINED_NETWORK_16GATES_FILE` / `TRAINED_NETWORK_2GATES_FILE` — binary output paths

Hyperparameters: `EPOCH_COUNT`, `TOTAL_SIZE`, `BATCH_SIZE`, `LEARNING_RATE`, `WEIGHT_DECAY`, `MAX_TEMPERATURE`.

Call `initialize_config()` at program start (already done in `run_network.py`).

---

## Component responsibilities and I/O (detailed)

Below is a concise table describing the *exact* input and output format between each major component. Where the code is ambiguous or inconsistent, the `Notes / Fix` column explains what to change or expect.

### Legend

* `TXT` = ASCII text file
* `BIN` = binary file with a recorded structure
* `Numpy` = `.npy` files saved with `numpy.save` / `jnp.save`

---

### 1) generator → network (file: `exec/network_architecture.txt`)

**Producing component**: `generator.generator.py` (two functions exist: `generate_network_custom_distribution()` and `generate_network_probabilistic_model()`).

**Consumed by**: `network.input_network()` in `network/network.py`.

**Expected format (by `network.input_network`)**:

* The file is read sequentially. For each non-input node the code executes: `left, right = map(int, file.readline().strip().split())`.
* That means **exactly two** integers per line separated by whitespace, representing `left_node_id right_node_id` for that gate.

**Examples** (one gate per line):

```
23 45
10 987
784 1100
```

**Generator behavior**:

* `generate_network_custom_distribution()` writes lines of the form `"{l} {r}\n"` — matches `network.input_network`.
* `generate_network_probabilistic_model()` (as provided) writes `"{l} {r} {cnt}\n"` (three columns). **This will break** `network.input_network` because it expects two integers. See *Gotchas & suggested fixes*.

**Suggested fix**: modify `generate_network_probabilistic_model()` to write only the first two integers per line, or change `network.input_network()` to `tokens = file.readline().strip().split(); left = int(tokens[0]); right = int(tokens[1])` to be robust against extra columns.

---

### 2) network → trained network file (`exec/trained_network_16gates.bin`)

**Producing component**: `network.print_network()` writes the trained network binary. **Note:** the code that writes the file writes, for each gate, four 32-bit little-endian signed integers (via `.to_bytes(4, byteorder='little', signed=True)`) in this order:

* gate\_index (an integer selecting one of 16 possible gate types) — 4 bytes
* aus\[current\_layer]\[i] (node id) — 4 bytes
* left\_nodes\[current\_layer + 1]\[i] (left node id) — 4 bytes
* right\_nodes\[current\_layer + 1]\[i] (right node id) — 4 bytes

However, **`convert_abc.py` expects the first 4 bytes of the file to be a 32-bit `node_count`** and then `node_count - INPUT_NODES` consecutive records of four int32 fields (type, id, left, right). That implies that a valid `trained_network_16gates.bin` must start with an initial `int32` count value before the gate records.

**Conclusion / Fix**: Ensure the writer writes an initial `node_count` (as a 32-bit signed little-endian integer) before emitting the per-gate quadruples. If you do not, `convert_abc.py` will read garbage for `node_count` and fail.

**Binary layout (recommended, matches converter expectations)**:

```
Offset 0   (4 bytes)   : int32 node_count (M)
Next      (4 bytes x N) : for each node: int32 gate_type (0..15)
                           int32 node_id
                           int32 left_node
                           int32 right_node
```

Where `N == node_count - INPUT_NODES`.

---

### 3) convert\_abc.py — 16-gate binary -> 2-gate AIG file (`trained_network_2gates.bin`)

**Input (as `convert_abc.py` expects)**: The 16-gate binary (see format above).

**What convert\_abc does**:

* Reads `node_count` (first int32) and then `node_count - INPUT_NODES` quadruples of int32.
* Each record is interpreted as `(type, id, left, right)` and stored in `input_nodes`.
* Performs a topological sort and logic-rewrite (replacing 16 gate types with combinations of 2-input AND + negations) producing a `final_nodes` list.
* Writes out an **AIG-like ASCII header** `aig M I L O A\n` where fields are: M (#nodes), I (#inputs = INPUT\_NODES), L (#latches = 0), O (#outputs = OUTPUT\_NODES), A (#AND-gates).
* Writes `O` output lines (the mapped literal IDs).
* Then writes the AND-gate list encoded as varint deltas (7-bit groups with MSB continuation) — this is a compact binary AIG delta stream.

**Output (file `TRAINED_NETWORK_2GATES_FILE`)**: Binary file containing an ASCII header and then varint-encoded delta list as shown. This is compatible with the logic expected by `evaluator_abc.py` (which uses `decode()` varint helper to read these two deltas per and-gate).

**Simplified spec for an example AIG**:

```
Header (ASCII):       aig M I L O A\n
Outputs (ASCII):      one integer per line (literal ids) x O
AND-gate data (BIN):  For each AND gate, 2 varints: delta0, delta1
```

---

### 4) evaluator\_abc.py (AIG evaluator)

**Inputs**:

* `TRAINED_NETWORK_2GATES_FILE` — AIG-like file produced by `convert_abc.py`.
* `TESTDATA_DATA_FILE` — text test set, formatted as in Data section below.

**Behavior**:

* `AndNot_network.init(path)` reads header and outputs, then decodes the varint gate deltas to reconstruct the gate graph.
* `evaluate_abc_format()` iterates over `TEST_SIZE` test samples from `TESTDATA_DATA_FILE`, runs the network, and prints: number tested and `% accuracy`.

**Output**: printed lines to stdout such as `Tested 10000/10000` and `12.3% accuracy`.

---

### 5) network.train/test & evaluator (16-gate path)

`network.train_network()` expects training/test sets as `TRAINING_DATA_FILE` / `TESTDATA_DATA_FILE` and produces/returns the learned `prob` arrays in memory, then `print_network()` writes the trained network binary file described earlier.

`evaluator.evaluator.py` (not displayed here) likely consumes `TRAINED_NETWORK_16GATES_FILE` and `TESTDATA_DATA_FILE` and reports accuracy. Follow the patterns used by `evaluator_abc.py` for the 2-gate path.

---

## Data file formats (training/test files)

The pipeline expects a very compact ASCII test/train format produced by `data/extractor.py`. The format is:

For each sample:

1. One line with 28×28 = `784` characters `0` or `1` with no separators, representing thresholded pixels (row-major left-to-right, top-to-bottom).
2. Next line: a single ASCII digit `0`..`9` followed by newline — the label.

Example for one sample:

```
000100000... (784 chars total)
7
```

Tools will cache pre-parsed NumPy arrays in the same folder when reading these files by saving `training.txt.values.npy` and `training.txt.answers.npy` (code uses `jnp.save(file + ".values.npy", values)`). When present, `network.read_values()` will load `.values.npy` and `.answers.npy` instead of reparsing the text.

---

## Function / in-memory interfaces (Python signatures & types)

Below are the main functions and the expected types they accept/return (for programmatic integration):

* `generator.generate_network_custom_distribution(rnd_seed: int = 0) -> None`

  * writes `NETWORK_ARCHITECTURE_FILE` text (two ints per line: `left right`).

* `generator.generate_network_probabilistic_model(rnd_seed: int = 0) -> None`

  * writes `NETWORK_ARCHITECTURE_FILE`. **(See format mismatch note.)**

* `network.input_network(left_nodes: list, right_nodes: list, prob: list, aus: list) -> None`

  * Reads `NETWORK_ARCHITECTURE_FILE` and appends initialized JAX arrays for `left_nodes`, `right_nodes`, and `prob`, and builds `aus` list-of-layers.
  * After call: `left_nodes`, `right_nodes`, `prob` are lists of JAX arrays.

* `network.train_network(prob: list, left_nodes: list, right_nodes: list) -> prob`:

  * Trains using JAX and returns updated `prob` (network parameters).

* `network.print_network(aus: list, prob: list, left_nodes: list, right_nodes: list) -> None`:

  * Writes `TRAINED_NETWORK_16GATES_FILE` binary file. **Make sure file header `node_count` is written**.

* `converter.convert_abc_format() -> None` (in `converter/convert_abc.py`)

  * Reads `TRAINED_NETWORK_16GATES_FILE` and writes `TRAINED_NETWORK_2GATES_FILE` AIG-like file.

* `evaluator.evaluate_abc_format() -> None` (in `evaluator/evaluator_abc.py`)

  * Reads `TRAINED_NETWORK_2GATES_FILE` and `TESTDATA_DATA_FILE` and prints accuracy.

---

## Gotchas & suggested fixes (you should address these for a working end-to-end pipeline)

1. **`generator_probabilistic_model` writes 3 columns per line** (`l r cnt`) while `network.input_network()` expects two (`l r`).

   * Fix: change the generator to `f.write(f"{l} {r}\n")` (omit `cnt`) or make `network.input_network` robust to extra columns.

2. **`print_network()` does not write an initial `node_count` header** but `convert_abc.py` expects one.

   * Fix: prepend `node_count` (int32 little-endian) at the start of `TRAINED_NETWORK_16GATES_FILE`. `node_count` should equal `NETWORK_SIZE` or whichever total you intend (be consistent with what `convert_abc` expects — it computes `first_output_id = node_count - OUTPUT_NODES + 1`).

3. **Indexing conventions** (1-based vs 0-based) are important throughout. The code expects input IDs to be `1..INPUT_NODES` and gates to be `INPUT_NODES+1 .. node_count`. Keep generator and writer consistent.

4. **Byte-order and signedness:** `convert_abc.py` uses `<i` little-endian signed ints and `print_network()` writes using `int.to_bytes(..., byteorder='little', signed=True)` — that part is consistent if the header is added.

5. **`OUTPUT_NODES` grouping assumption:** `network.inference` groups `OUTPUT_NODES` into 10 categories: `category_size = OUTPUT_NODES // 10`. Make sure `OUTPUT_NODES` is divisible by 10 or adjust code.

---

## Example end-to-end sequence (recommended)

1. `python data/extractor.py` → produces `data/training.txt` and `data/testdata.txt`.
2. `python run_network.py` does:

   * `initialize_config()`
   * `generate_network_probabilistic_model()` (or the other generator)
   * run `network_alternative.an()` (provided)
   * when `ABC_FORMAT == 1` the script calls `convert_network()` then `evaluate_network()`; otherwise it runs the ABC conversion/evaluator path.

> To debug, run steps individually in REPL and inspect intermediate files:
>
> * Open `exec/network_architecture.txt` and verify line counts / format.
> * After training, inspect `exec/trained_network_16gates.bin` header and size.
> * Run `python converter/convert_abc.py` manually and check that `exec/trained_network_2gates.bin` is produced.

---

## Troubleshooting tips

* If `convert_abc.py` fails reading the 16-gate bin: check that the first 4 bytes encode a reasonable `node_count` (e.g. not an insanely large number). Use a small Python snippet to `int.from_bytes(open(file,'rb').read(4), 'little', signed=True)` to validate.

* If parsing `network_architecture.txt` fails with `ValueError: too many values to unpack`, open the file and check whether lines contain 2 or 3 values.

* If accuracy is unexpectedly poor: verify that `training.txt`/`testdata.txt` encoding matches `network.read_values()` expectations (i.e., repeated calls to `readline()` read input bits line followed by a label line). Also check `OUTPUT_NODES // 10` is an integer divisor.

---

## Recommended small fixes (patch suggestions)

1. In `generator.generate_network_probabilistic_model()` change the write line from

```py
f.write(f"{l} {r} {cnt}\n")
```

to

```py
f.write(f"{l} {r}\n")
```

2. In `network.print_network()` write the `node_count` header before gate records. Example:

```py
with open(TRAINED_NETWORK_16GATES_FILE, "wb") as f:
    # Example node_count: the total number of nodes (NETWORK_SIZE)
    f.write(int(NETWORK_SIZE).to_bytes(4, byteorder='little', signed=True))
    # then the gate quadruples as currently implemented
    ...
```

3. Make `network.input_network()` robust to extra columns by reading `tokens = file.readline().strip().split()` and using `left = int(tokens[0]); right = int(tokens[1])`.
