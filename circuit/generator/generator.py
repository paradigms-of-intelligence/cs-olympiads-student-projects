from random import seed, randint, shuffle, choices
import numpy as np

from config import *


def id_to_pos(node_index_zero_based: int):
    """Map a 0-based node index to (row, col) in a 28x28 grid."""
    return (node_index_zero_based // 28, node_index_zero_based % 28)


def comp_prob(x: int, y: int):
    """Compute a weight that favors nodes closer to the center (example)."""
    return (min(x, 28 - x) + min(y, 28 - y)) ** 3


# ---------- Generator 1: custom distribution ----------
def generate_network_custom_distribution(rnd_seed: int = 0):
    """
    Writes a network architecture using the custom distribution strategy.
    Keeps input pixel horizontal pair connections and alternating vertical connections,
    then builds the subsequent layers. Matches the style in the first script.
    """
    square = 28
    seed(rnd_seed)
    np.random.seed(rnd_seed)

    # This probability list was in your first file (14 values repeated * 56 -> 784)
    prob = [
        0.03 / 112,
        0.06 / 112,
        0.11 / 112,
        0.2 / 112,
        0.2 / 112,
        0.2 / 112,
        0.2 / 112,
        0.2 / 112,
        0.2 / 112,
        0.2 / 112,
        0.2 / 112,
        0.11 / 112,
        0.06 / 112,
        0.03 / 112,
    ] * 56

    with open(NETWORK_ARCHITECTURE_FILE, "w") as f:
        # --- Input pixel adjacency (horizontal pairs and alternating vertical)
        # Input nodes are assumed 1..INPUT_NODES
        for i in range(1, INPUT_NODES, 2):
            f.write(f"{i} {i+1}\n")
        for i in range(1, INPUT_NODES + 1):
            # connect to the node `square` below for alternating rows
            if ((i - 1) // square) % 2 == 0:
                below = i + square
                if below <= INPUT_NODES:
                    f.write(f"{i} {below}\n")

        # --- Inter-layer connections
        # First layer node id for the first non-input layer should be INPUT_NODES+1
        first_layer_node = INPUT_NODES + 1

        for layer in range(1, len(LAYERS)):
            prev_size = LAYERS[layer - 1]
            next_size = LAYERS[layer]

            # Left-side nodes (nl): deterministic first min(...) then fill to next_size
            nl = [
                x
                for x in range(
                    first_layer_node, first_layer_node + min(prev_size, next_size)
                )
            ]
            cp = nl.copy()

            while len(nl) < next_size:
                if layer == 2:
                    # second hidden layer uses the precomputed `prob` array (from the first file)
                    choice_node = int(np.random.choice(cp, p=prob))
                    nl.append(choice_node)
                else:
                    # uniform random pick among previous layer nodes
                    nl.append(
                        randint(first_layer_node, first_layer_node + prev_size - 1)
                    )

            shuffle(nl)

            # Right-side nodes: pair with random previous-layer nodes (uniform)
            nr = [
                randint(first_layer_node, first_layer_node + prev_size - 1)
                for _ in range(next_size)
            ]

            # advance first_layer_node by prev_size (we move to indexing for next layer)
            first_layer_node += prev_size

            # write edges left->right
            for l, r in zip(nl, nr):
                f.write(f"{l} {r}\n")

        # optional footer
        f.write(f"# Total output nodes: {OUTPUT_NODES}\n")


# ---------- Generator 2: probabilistic model ----------
def generate_network_probabilistic_model(rnd_seed: int = 0):
    """
    Writes a network architecture using the probabilistic model from the second script.
    This function treats node indexing with the input layer starting at 1 (1..INPUT_NODES).
    The first hidden layer selection uses a spatial probability based on comp_prob.
    """
    seed(rnd_seed)
    np.random.seed(rnd_seed)

    with open(NETWORK_ARCHITECTURE_FILE, "w") as f:
        first_layer_node = 1  # in the second script main(), indexing started at 1

        for layer in range(1, len(LAYERS)):
            prev_size = LAYERS[layer - 1]
            next_size = LAYERS[layer]

            # nl: deterministic first min(...) then fill to next_size
            nl = [
                x
                for x in range(
                    first_layer_node, first_layer_node + min(prev_size, next_size)
                )
            ]

            while len(nl) < next_size:
                nl.append(randint(first_layer_node, first_layer_node + prev_size - 1))
            shuffle(nl)

            # nr depends on whether this is the first hidden layer (layer == 1)
            if layer == 1:
                # build spatial probability weights for previous-layer nodes
                prob_weights = []
                for node in range(first_layer_node, first_layer_node + prev_size):
                    # id_to_pos requires 0-based index
                    x, y = id_to_pos(node - 1)
                    prob_weights.append(comp_prob(x, y))

                total = sum(prob_weights)
                if total <= 0:
                    # fallback to uniform if something weird happens
                    prob_weights = None
                else:
                    prob_weights = [p / total for p in prob_weights]

                # choose next_size nodes from previous layer with weights
                nr = choices(
                    population=[
                        i for i in range(first_layer_node, first_layer_node + prev_size)
                    ],
                    weights=prob_weights,
                    k=next_size,
                )
                # advance
                first_layer_node += prev_size
            else:
                nr = [
                    randint(first_layer_node, first_layer_node + prev_size - 1)
                    for _ in range(next_size)
                ]
                first_layer_node += prev_size

            # write edges left->right
            cnt = 785
            for l, r in zip(nl, nr):
                f.write(f"{l} {r} {cnt}\n")
                cnt += 1
