import random
import jax.numpy as jnp
from jax import grad, jit
import math
import functools

global network_size
network_size = 0  # Number of gates in the network
OUTPUT_NODES: list["Gate"] = []
nodes: list["Gate"] = []   #Gates

EPOCH_COUNT = 2
INPUT_SIZE = 784
TEST_CASE_COUNT = 10

# TODO: Simplyfy This all. With the Idea of the 4 bools instead of the Gates.
ALPHA = 0.001
BETA2 = .999
BETA1 = .9
EPSILON = 1e-8
LEARNING_RATE = 0.01

# This should be multiplied by BETA1 and BETA2
# and be updated for each iteration
BETA1_TIMESTAMP = 0
BETA2_TIMESTAMP = 0

class Gate:
    p = jnp.array([random.uniform(0, 1) for _ in range(0, 4)])  # Parameters for the gate
    value = 0
    
    v = jnp.array([0 for _ in range (0, 4)])
    m = jnp.array([0 for _ in range (0, 4)])
    error = 0

    def __init__(self, a : int, b : int) -> None:
        self.a = a
        self.b = b

    def update_momentum(self, nabla):
        #self.m = self.m*BETA1 + (1)
        for i in range (0, 4):
            m_i = self.m.at[i].get()
            m_i = BETA1 * m_i + (1 - BETA1) * nabla.at[i].get()
            self.m.at[i].set(m_i)

            v_i = self.v.at[i].get()
            v_i = BETA2 * v_i + (1 - BETA2) * pow(nabla.at[i].get(), 2)
            self.v.at[i].set(v_i)
            
            i_m = self.m.at[i].get()/(1-BETA1_TIMESTAMP)
            i_v = (self.v.at[i].get())/(1-BETA2_TIMESTAMP)
            self.p.at[i].set(self.p[i] - ALPHA * i_m / (math.sqrt(i_v) + EPSILON))
        
        # propagate_backward
        nodes[self.a].error += self.error
        nodes[self.b].error += self.error
        # reset error
        error = 0

def softmax(x):
    e_x = jnp.exp(x - jnp.max(x))
    return e_x / jnp.sum(e_x)

def inference_function(a: float, b: float, p):
    #SUUUUUUUUUUUUUUUS
    p = softmax(p)

    return p[0] * a * b + p[1] * a * (1 - b) + p[2] * (1 - a) * b + p[3] * (1 - a) * (1 - b)

def get_gate(index: int, p) -> int:
    num: int = 0

    for i in range(4):
        num += (round(p[index][i]) << i)

    return num
    
def activation(node: Gate):
    global nodes, OUTPUT_NODES, network_size, INPUT_SIZE
    v1, v2 = nodes[node.a].value, nodes[node.b].value
    node.value = function(v1, v2, node.p)    
    pass

    # ?

@jit
def inference():
    global nodes, OUTPUT_NODES, network_size, INPUT_SIZE
    # feedforward
    # ...
    # must be toposorted
    for i in range (INPUT_SIZE+1, network_size):
        Gate = nodes[i]
        v1, v2 = nodes[Gate.a].value, nodes[Gate.b].value
        Gate.value = inference_function(v1, v2, Gate.p)

@functools.partial(jit, static_argnames=["answer"])
def backpropagate(answer):
    global nodes, OUTPUT_NODES, network_size, INPUT_SIZE, LEARNING_RATE
    # backpropagate through every node
    # TODO: de-sus this: de-marago this
    nabla = [[0 for _ in range (0, 4)] for _ in range (0, network_size)]

    # evaluating cost function
    for i in range (0, len(OUTPUT_NODES)):
        if (i//(len(OUTPUT_NODES)/10) == answer): nodes[OUTPUT_NODES[i]].error = pow((1-nodes[OUTPUT_NODES[i]].error), 2)
        else: nodes[OUTPUT_NODES[i]].error = pow((nodes[OUTPUT_NODES[i]].error), 2)

    for i in range (network_size, INPUT_SIZE, -1):
        gate = nodes[i]
        a_val, b_val = nodes[gate.a].value, nodes[gate.b].value
        delta_out = gate.error

        # local derivatives
        df_da = grad(inference_function, argnums=0)(a_val, b_val, gate.p)
        df_db = grad(inference_function, argnums=1)(a_val, b_val, gate.p)
        df_dp = grad(inference_function, argnums=2)(a_val, b_val, gate.p)

        # propagate error backwards
        nodes[gate.a].error += delta_out * df_da * LEARNING_RATE
        nodes[gate.b].error += delta_out * df_db * LEARNING_RATE

        # parameter gradient for Adam
        nabla = delta_out * df_dp * LEARNING_RATE
        gate.update_momentum(nabla)

        # reset error
        gate.error = 0

def main():
    with open("network_architecture.txt", 'r') as file:
        global network_size, INPUT_SIZE, OUTPUT_NODES, nodes, EPOCH_COUNT
        network_size = int(file.readline().strip())
        nodes = []
        # Store input values
        for _ in range (0, INPUT_SIZE+1):
            nodes.append(Gate(-1,-1))

        # Read network architecture
        for _ in range (INPUT_SIZE, network_size):
            a,b = map(int, file.readline().strip().split())
            nodes.append(Gate(a,b))

        # assumed to be the last N
        number_outputs = int(file.readline().strip())
        OUTPUT_NODES = [x for x in range(network_size-number_outputs+1, network_size+1)]

    BETA1_TIMESTAMP = 1
    BETA2_TIMESTAMP = 1
    # Start training routine
    for epoch in range (0, EPOCH_COUNT):
        BETA1_TIMESTAMP *= BETA1
        BETA2_TIMESTAMP *= BETA2

        #read data
        for test_case in range(0, TEST_CASE_COUNT):
            with open("../data/training/img_" + str(test_case) + ".txt", 'r') as file:

                #read training input
                line = list(map(float, file.readline().strip()))

                for i in range (1, INPUT_SIZE+1):
                    nodes[i].value = line[i-1]
                inference()
                
                #read result
                answer = int(file.readline().strip())
                backpropagate(answer)
            
        print("Epoch " + str(epoch+1))
    
    # Print the network
    with open("trained_network.bin", "wb") as f:
        # Network size -> 32 bits/ 4 bytes
        f.write(network_size.to_bytes(4, byteorder = 'little'))

        for id in range(INPUT_SIZE+1, network_size+1):
            gate_type = get_gate(id, nodes)
            f.write(gate_type.to_bytes(4, byteorder = 'little', signed=True))
            f.write(id.to_bytes(4, byteorder='little', signed=True))
            f.write(int(nodes[id].a).to_bytes(4, byteorder='little', signed=True))
            f.write(int(nodes[id].b).to_bytes(4, byteorder='little', signed=True))

            #if (id < 786): print("Debug: id is " + str(id) + "  £  gate type is "+ str(int(nodes[id].p.index(max(nodes[id].p)))))

        for id in range (network_size-9, network_size+1):
            f.write(id.to_bytes(4, byteorder='little', signed=True))

if __name__ == "__main__":
    main()