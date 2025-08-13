import random
import jax.numpy as jnp
from jax import grad
import math

NETWORK_SIZE = 0 
OUTPUT_NODES = []
nodes = 0            #Gate

EPOCH_COUNT = 10
INPUT_SIZE = 784
INPUT_NAME = "../"
TEST_CASE_COUNT = 10000

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
    p = [random.gauss() for _ in range (0, 16)]
    value = 0
    
    v = [0 for _ in range (0, 16)]
    m = [0 for _ in range (0, 16)]
    error = 0

    def __init__(self, a, b) -> None:
        self.a = a
        self.b = b

    def update_momentum(self, nabla):
        for i in range (0, 16):
            self.m[i] = BETA1 * self.m[i] + (1-BETA1) * nabla
            self.v[i] = BETA2 * self.v[i] + (1-BETA2) * nabla * nabla
            i_m = self.m[i]/(1-BETA1_TIMESTAMP)
            i_v = self.v[i]/(1-BETA2_TIMESTAMP)
            self.p[i] = self.p[i] - ALPHA * i_m[i] / (math.sqrt(i_v[i]) + EPSILON)
        
        # propagate_backward
        nodes[self.a].error += error
        nodes[self.b].error += error
        # reset error
        error = 0
        

def softmax(x):
    e_x = jnp.exp(x - jnp.max(x))
    return e_x / jnp.sum(e_x)

# def f_todo(x):
#     def a(a, b):
#         return (x & 1) * a * b + (x & 2) * a * (1 - b) + (x & 4) * (1 - a) * b + (x & 8) * (1 - a) * (1 - b)

#     return a

def inference_function(a: float, b: float, p):
    pr = a*b
    #SUUUUUUUUUUUUUUUS
    softmax(p)
    f = []
    f[0] = 0 
    p = softmax(p)

    f = [0 for _ in range (0, 16)]
    f[0] = 0
    f[1] = pr*p[1]
    f[2] = (a-pr)*p[2]
    f[3] = a*p[3]
    f[4] = (b-pr)*p[4]
    f[5] = b*p[5]
    f[6] = (a+b-2*pr)*p[6]
    f[7] = (a+b-pr)*p[7]
    f[8] = (1-a-b+pr)*p[8]
    f[9] = (1-a-b+2*pr)*p[9]
    f[10] = (1-b)*p[10]
    f[11] = (1-b+pr)*p[11]
    f[12] = (1-a)*p[12]
    f[13] = (1-a+pr)*p[13]
    f[14] = (1-pr)*p[14]
    f[15] = p[15]
    
    sum = 0
    for l in f: 
        sum += l

    return sum

def activation(node: Gate):
    v1, v2 = nodes[Gate.a].v, nodes[Gate.b].v
    Gate.v = function(v1, v2, Gate.p)    
    pass


def inference():
    # feedforward
        # ...
    # must be toposorted
    for i in range (INPUT_SIZE+1, NETWORK_SIZE):
        Gate = nodes[i]
        v1, v2 = nodes[Gate.a].value, nodes[Gate.b].value
        Gate.value = inference_function(v1, v2, Gate.p)

def backpropagate(answer):
    # backpropagate through every node
    # TODO: de-sus this: de-marago this
    nabla = [[0 for _ in range (0,16)] for _ in range (0, NETWORK_SIZE)]

    for i in range (0, len(OUTPUT_NODES)):
        if (i//(OUTPUT_NODES/10) == answer): nodes[OUTPUT_NODES[i]].error = pow((1-nodes[OUTPUT_NODES[i]].error), 2)
        else: nodes[OUTPUT_NODES[i]].error = pow((nodes[OUTPUT_NODES[i]].error), 2)

    for i in range (NETWORK_SIZE, INPUT_SIZE, -1):
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
    NETWORK_SIZE = int(input().strip())
    nodes = []
    # Store input values
    for _ in range (0, INPUT_SIZE+1):
        nodes.append(Gate(-1,-1))

    # Read network architecture
    for _ in range (INPUT_SIZE, NETWORK_SIZE):
        a,b = map(int, input().strip().split())
        nodes.append(Gate(a,b))

    # assumed to be the last N
    number_outputs = int(input().strip())
    OUTPUT_NODES = [x for x in range(NETWORK_SIZE-number_outputs+1, NETWORK_SIZE+1)]

    BETA1_TIMESTAMP = 1
    BETA2_TIMESTAMP = 1
    # Start training routine
    for epoch in range (0, EPOCH_COUNT):
        BETA1_TIMESTAMP *= BETA1
        BETA2_TIMESTAMP *= BETA2

        #read data
        for test_case in range(0, TEST_CASE_COUNT):
            with open("../../data/training/img_" + str(test_case) + ".txt", 'r') as file:

                #read training input
                line = list(file.readline().strip())
                map(int, line)

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
        f.write(NETWORK_SIZE.to_bytes(4, byteorder = 'little'))

        for id in range(INPUT_SIZE+1, NETWORK_SIZE+1):
            f.write(int(nodes[id].p.index(max(nodes[id].p))).to_bytes(4, byteorder = 'little', signed=True))
            f.write(id.to_bytes(4, byteorder='little', signed=True))
            f.write(int(nodes[id].a).to_bytes(4, byteorder='little', signed=True))
            f.write(int(nodes[id].b).to_bytes(4, byteorder='little', signed=True))

            if (id < 786): print("Debug: id is " + str(id) + "  £  gate type is "+ str(int(nodes[id].p.index(max(nodes[id].p)))))

        for id in range (NETWORK_SIZE-9, NETWORK_SIZE+1):
            f.write(id.to_bytes(4, byteorder='little', signed=True))






if __name__ == "__main__":
    main()