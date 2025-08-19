from random import *
import numpy as np

INPUT_NODES = 784
OUTPUT_NODES = 1000
square = 28

LAYERS = [INPUT_NODES, INPUT_NODES, 4500, 3000, 2000, 1500,  OUTPUT_NODES]

seed(0)

print(sum(LAYERS))
first_layer_node = INPUT_NODES+1

for i in range (1, INPUT_NODES, 2): 
    print(i, i+1)
for i in range(1, INPUT_NODES+1, 1):
    if ((i-1)//square) %2 == 0: 
        print(i, i+square)

prob = [0.03/112, 0.06/112, 0.11/112, 0.2/112, 0.2/112, 0.2/112, 0.2/112, 0.2/112, 0.2/112, 0.2/112, 0.2/112, 0.11/112, 0.06/112, 0.03/112] *56

for layer in range(2, len(LAYERS)):
    prev_size = LAYERS[layer-1]
    next_size = LAYERS[layer] # SUS SUS SUS SUS SUS SUS SUS SUS TRIBUTO GORMITA SUS SUS SUS SUS
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
        print(str(l) + " " + str(r))

print(OUTPUT_NODES)
