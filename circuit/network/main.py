import random
import math
import functools
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad

# ---------------- Hyperparameters ----------------
EPOCH_COUNT = 1
INPUT_SIZE = 784
TEST_CASE_COUNT = 3

ALPHA = 0.001
BETA2 = 0.999
BETA1 = 0.9
EPSILON = 1e-8
LEARNING_RATE = 0.01

# ---------------- Global state ----------------
NETWORK_SIZE = 0  # Number of nodes (inputs + internal + outputs)
OUTPUT_NODES = []
nodes = []   # list[Gate]

# Adam bias-correction timestamps (updated in main loop)
BETA1_TIMESTAMP = 0.0
BETA2_TIMESTAMP = 0.0


# ---------------- Model primitives ----------------
def inference_function(a: float, b: float, p: jnp.ndarray) -> jnp.ndarray:
    """
    Pure JAX, vectorized features; p is softmax'd; returns scalar output.
    """
    p = jax.nn.softmax(p)
    pr = a * b
    feats = jnp.array([
        0.0,                # 0
        pr,                 # 1
        a - pr,             # 2
        a,                  # 3
        b - pr,             # 4
        b,                  # 5
        a + b - 2.0 * pr,   # 6
        a + b - pr,         # 7
        1.0 - a - b + pr,   # 8
        1.0 - a - b + 2.0*pr,# 9
        1.0 - b,            # 10
        1.0 - b + pr,       # 11
        1.0 - a,            # 12
        1.0 - a + pr,       # 13
        1.0 - pr,           # 14
        1.0                 # 15
    ], dtype=p.dtype)
    return jnp.dot(p, feats)


# Build once: JITted value+grads w.r.t. (a, b, p)
inference_vg = jit(value_and_grad(inference_function, argnums=(0, 1, 2)))


class Gate:
    """
    A single gate/node in the graph.
    NOTE: Use instance attributes, not class attributes.
    """
    def __init__(self, a: int, b: int) -> None:
        self.a = a
        self.b = b
        # Parameters and optimizer state
        self.p = jnp.array([random.uniform(0.0, 1.0) for _ in range(16)], dtype=jnp.float32)
        self.m = jnp.zeros_like(self.p)
        self.v = jnp.zeros_like(self.p)
        # Forward value and backprop error (scalars)
        self.value = jnp.array(0.0, dtype=jnp.float32)
        self.error = jnp.array(0.0, dtype=jnp.float32)

    def update_momentum(self, nabla: jnp.ndarray, t1: float, t2: float):
        """
        Optim #1: fully vectorized Adam with bias correction.
        t1, t2 are running BETA1^t and BETA2^t (pass in, do not read globals).
        """
        self.m = BETA1 * self.m + (1.0 - BETA1) * nabla
        self.v = BETA2 * self.v + (1.0 - BETA2) * (nabla * nabla)

        # Bias correction (avoid division by zero at t=0 by ensuring t1,t2>0 from caller)
        m_hat = self.m / (1.0 - t1)
        v_hat = self.v / (1.0 - t2)

        self.p = self.p - ALPHA * m_hat / (jnp.sqrt(v_hat) + EPSILON)

        # propagate backward to inputs (accumulate error)
        if self.a >= 0:
            nodes[self.a].error = nodes[self.a].error + self.error
        if self.b >= 0:
            nodes[self.b].error = nodes[self.b].error + self.error

        # reset own error
        self.error = jnp.array(0.0, dtype=jnp.float32)


# ---------------- Inference / Backprop ----------------
def inference():
    """
    Feed-forward over all non-input nodes in topological order.
    """
    global nodes, NETWORK_SIZE, INPUT_SIZE
    # Nodes 0..INPUT_SIZE are inputs (assuming 1-based inputs in original; normalize here)
    # We'll assume nodes are appended in topo order.
    for i in range(INPUT_SIZE + 1, NETWORK_SIZE + 1):
        g = nodes[i]
        v1 = nodes[g.a].value if g.a >= 0 else jnp.array(0.0, dtype=jnp.float32)
        v2 = nodes[g.b].value if g.b >= 0 else jnp.array(0.0, dtype=jnp.float32)
        g.value = inference_function(v1, v2, g.p)


def backpropagate(answer: int, t1: float, t2: float):
    """
    Backprop through every node.
    Optim #2: reuse prebuilt, jit'ed value_and_grad on inference_function.
    """
    global nodes, OUTPUT_NODES, NETWORK_SIZE, INPUT_SIZE, LEARNING_RATE

    # Evaluate a simple "cost" into node.error at outputs (keeping original intent)
    # (Original code used node.error slots for both value and error—kept structure minimal.)
    # Here, we interpret g.value as logits-ish scalar per output; compute squared error vs one-hot.
    K = len(OUTPUT_NODES)
    if K > 0:
        correct_bucket = answer  # expecting 0..9
        # Assign errors: (target - value)^2, accumulated in .error
        for idx, node_id in enumerate(OUTPUT_NODES):
            target = 1.0 if (idx // max(K // 10, 1)) == correct_bucket else 0.0
            pred = nodes[node_id].value
            nodes[node_id].error = (target - pred) * (target - pred)

    # Backward over gates (reverse topo)
    for i in range(NETWORK_SIZE, INPUT_SIZE, -1):
        g = nodes[i]
        a_val = nodes[g.a].value if g.a >= 0 else jnp.array(0.0, dtype=jnp.float32)
        b_val = nodes[g.b].value if g.b >= 0 else jnp.array(0.0, dtype=jnp.float32)
        delta_out = g.error

        # Optim #2: single jit'ed call for val and grads wrt (a, b, p)
        val, (df_da, df_db, df_dp) = inference_vg(a_val, b_val, g.p)

        # Propagate error backwards to inputs (SGD-scaled)
        nodes[g.a].error = nodes[g.a].error + delta_out * df_da * LEARNING_RATE if g.a >= 0 else jnp.array(0.0, dtype=jnp.float32)
        nodes[g.b].error = nodes[g.b].error + delta_out * df_db * LEARNING_RATE if g.b >= 0 else jnp.array(0.0, dtype=jnp.float32)

        # Parameter gradient for Adam (SGD-scaled)
        nabla = delta_out * df_dp * LEARNING_RATE

        # Optim #1: vectorized Adam update (pass timestamps)
        g.update_momentum(nabla, t1, t2)

        # Reset this node's backprop error after use (already done in update_momentum for symmetry)
        g.error = jnp.array(0.0, dtype=jnp.float32)


# ---------------- Training driver ----------------
def main():
    global NETWORK_SIZE, INPUT_SIZE, OUTPUT_NODES, nodes, EPOCH_COUNT
    global BETA1_TIMESTAMP, BETA2_TIMESTAMP

    with open("network_architecture.txt", 'r') as file:
        NETWORK_SIZE = int(file.readline().strip())

        nodes = []
        # Reserve indices 0..INPUT_SIZE for inputs
        for _ in range(0, INPUT_SIZE + 1):
            nodes.append(Gate(-1, -1))

        # Read network architecture for nodes INPUT_SIZE+1 .. NETWORK_SIZE
        # (Fix off-by-one: include NETWORK_SIZE)
        for _ in range(INPUT_SIZE + 1, NETWORK_SIZE + 1):
            a, b = map(int, file.readline().strip().split())
            nodes.append(Gate(a, b))

        # Read number of outputs, assume they are the last N nodes
        number_outputs = int(file.readline().strip())
        OUTPUT_NODES = [x for x in range(NETWORK_SIZE - number_outputs + 1, NETWORK_SIZE + 1)]

    # Initialize bias-correction timestamps at 1.0 (so first step uses 1 - BETA^1)
    BETA1_TIMESTAMP = 1.0
    BETA2_TIMESTAMP = 1.0

    # Training loop
    for epoch in range(0, EPOCH_COUNT):
        # Update timestamps (these represent BETA^t)
        BETA1_TIMESTAMP *= BETA1
        BETA2_TIMESTAMP *= BETA2

        for test_case in range(0, TEST_CASE_COUNT):
            with open(f"../data/training/img_{test_case}.txt", 'r') as file:
                print("Reading image", test_case)

                # Read training input: expecting a line of digits '0'/'1' or floats without spaces
                # Original code did: list(map(float, file.readline().strip()))
                # Keep behavior but cast to float array via jnp.array for speed.
                line_str = file.readline().strip()
                line = jnp.array(list(map(float, line_str)), dtype=jnp.float32)

                # Load inputs into nodes[1..INPUT_SIZE]
                for i in range(1, INPUT_SIZE + 1):
                    nodes[i].value = line[i - 1]

                # Forward
                inference()

                # Read label
                answer = int(file.readline().strip())

                # Backward (pass current timestamps for Adam bias correction)
                backpropagate(answer, BETA1_TIMESTAMP, BETA2_TIMESTAMP)

        print("Epoch", epoch + 1)

    # Save the network
    with open("trained_network.bin", "wb") as f:
        # Network size -> 32 bits/ 4 bytes
        f.write(NETWORK_SIZE.to_bytes(4, byteorder='little', signed=False))

        for idx in range(INPUT_SIZE + 1, NETWORK_SIZE + 1):
            gate = nodes[idx]
            gate_type = int(jnp.argmax(gate.p))
            f.write(gate_type.to_bytes(4, byteorder='little', signed=True))
            f.write(idx.to_bytes(4, byteorder='little', signed=True))
            f.write(int(gate.a).to_bytes(4, byteorder='little', signed=True))
            f.write(int(gate.b).to_bytes(4, byteorder='little', signed=True))

        for idx in range(NETWORK_SIZE - 9, NETWORK_SIZE + 1):
            f.write(idx.to_bytes(4, byteorder='little', signed=True))


if __name__ == "__main__":
    main()
