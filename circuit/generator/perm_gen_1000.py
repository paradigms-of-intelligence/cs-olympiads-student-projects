from random import *

LAYER_NUMBER = 5
INPUT_NODES = 784
OUTPUT_NODES = 1000

LAYERS = [INPUT_NODES, 2000, 2000, 2000, OUTPUT_NODES]



print(sum(LAYERS))
first_layer_node = 1

for layer in range(1, len(LAYERS)):
    prev_size = LAYERS[layer-1]
    next_size = LAYERS[layer] # SUS SUS SUS SUS SUS SUS SUS SUS TRIBUTO GORMITA SUS SUS SUS SUS
    nl = [randint(first_layer_node, first_layer_node + prev_size-1) for _ in range(0, next_size)]
    nr = [randint(first_layer_node, first_layer_node + prev_size-1) for _ in range(0, next_size)]
    first_layer_node += prev_size

    for [l,r] in zip(nl,nr):
        print(str(l) + " " + str(r))

print(OUTPUT_NODES)
