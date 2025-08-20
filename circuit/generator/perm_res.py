from random import *

INPUT_NODES = 784
OUTPUT_NODES = 1000

LAYERS = [INPUT_NODES, 3000, 2000, OUTPUT_NODES]


def id_to_pos(node):
    return (node//28, node%28)

def comp_prob(x, y):
    return pow((min(x, 28-x) + min(y, 28-y)), 3)

def main():
    print(sum(LAYERS))
    first_layer_node = 1

    for layer in range(1, len(LAYERS)):
        prev_size = LAYERS[layer-1]
        next_size = LAYERS[layer] # SUS SUS SUS SUS SUS SUS SUS SUS TRIBUTO GORMITA SUS SUS SUS SUS
        nl = [x for x in range(first_layer_node, first_layer_node + min(prev_size, next_size))]


        while len(nl) < next_size:
            nl.append(randint(1,784))
        shuffle(nl)

        
        nr = []
        if layer == 1:
            prob = []
            for node in range(first_layer_node, first_layer_node + prev_size):
                x, y = id_to_pos(node - 1)   # subtract 1 if your node indexing starts at 1
                prob.append(comp_prob(x, y))

            _sum = sum(prob)
            prob = [p/_sum for p in prob]

            nr = choices(
                population=[i for i in range(first_layer_node, first_layer_node + prev_size)],
                weights=prob,
                k=next_size
            )

            first_layer_node += prev_size
        else:
            nr = [randint(first_layer_node, first_layer_node + prev_size-1) for _ in range(0, next_size)]
            first_layer_node += prev_size

        for [l,r] in zip(nl,nr):
            print(str(l) + " " + str(r))

    print(OUTPUT_NODES)


if __name__ == "__main__":
    main()