import random
import jax 
import jax.numpy as jnp

global NETWORK_SIZE 
global nodes                            #Gate

EPOCH_COUNT = 10
INPUT_SIZE = 784
INPUT_NAME = "../"

# TODO: Simplyfy This all. With the Idea of the 4 bools instead of the Gates.

class Gate:
    p = [random.gauss() for x in range (0, 16)]
    v = 0

    def __init__(self, a, b) -> None:
        self.a = a
        self.b = b


def softmax(x):
    e_x = jnp.exp(x - jnp.max(x))
    return e_x / jnp.sum(e_x)

def f(x):
    def a(a, b):
        return (x & 1) * a * b + (x & 2) * a * (1 - b) + (x & 4) * (1 - a) * b + (x & 8) * (1 - a) * (1 - b)

    return a


def function(a: float, b: float, p):
    pr = a*b
    #SUUUUUUUUUUUUUUUS
    softmax(p)
    f = []
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

    if sum > 1.0:
        sum = softmax(sum)

    return sum

def activation(node: Gate):
    v1, v2 = nodes[Gate.a].v, nodes[Gate.b].v
    Gate.v = function(v1, v2, Gate.p)    
    pass

def inference():
    #read

    with open(INPUT_NAME, 'r') as file:
        line = map(int, file.readline().strip().split())
        for i in range (1, INPUT_SIZE+1):
            nodes[i].v = line[i-1]


    # feedforward
        # ...
    # must be toposorted
    for i in range (INPUT_SIZE+1, NETWORK_SIZE):
        activation(i)


def backpropagate():
    pass

def main():
    NETWORK_SIZE = map(int, input().strip().split())

    # Store input values
    for _ in range (0, INPUT_SIZE+1):
        nodes.append(Gate(-1,-1))

    # Read network architecture
    for _ in range (INPUT_SIZE+1, NETWORK_SIZE):
        nodes.append(Gate(map(int, input().strip().split())))

    
    # Start training routine
    for i in range (0, EPOCH_COUNT):
        inference()

        backpropagate()
        print("Epoch " + str(i+1))



if __name__ == "__main__":
    main()