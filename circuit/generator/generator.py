from config import *
from random import *
import numpy as np

def generate_network():
    square = 28
    seed(0)
    with open(NETWORK_ARCHITECTURE_FILE, "w") as f:
        first_layer_node = INPUT_NODES+1

        for i in range (1, INPUT_NODES, 2): 
            f.write(str(i) + " " + str(i+1) + "\n")
        for i in range(1, INPUT_NODES+1, 1):
            if ((i-1)//square) %2 == 0: 
                f.write(str(i) + " " + str(i+square) + "\n")

        prob = [0.03/112, 0.06/112, 0.11/112, 0.2/112, 0.2/112, 0.2/112, 0.2/112, 0.2/112, 0.2/112, 0.2/112, 0.2/112, 0.11/112, 0.06/112, 0.03/112] *56

        for layer in range(1, len(LAYERS)):
            prev_size = LAYERS[layer-1]
            next_size = LAYERS[layer]
            nl = [x for x in range(first_layer_node, first_layer_node + min(prev_size, next_size))]
            cp = nl.copy()
            

            while len(nl) < next_size:
                if (layer == 2): 
                    nl.append(np.random.choice(cp, p=prob))
                else: nl.append(randint(first_layer_node, first_layer_node + prev_size-1))
            shuffle(nl)

            nr = [randint(first_layer_node, first_layer_node + prev_size-1) for _ in range(0, next_size)]
            first_layer_node += prev_size

            for [l,r] in zip(nl,nr):
                f.write(str(l) + " " + str(r) + "\n")
