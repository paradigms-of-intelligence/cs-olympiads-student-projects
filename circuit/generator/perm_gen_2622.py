from random import *

# dimensioni corrette per MNIST con feature estese
PIXEL_INPUT = 784
EXTRA_FEATURES = 729 + 625 + 484  # 1838
INPUT_NODES = PIXEL_INPUT + EXTRA_FEATURES  # 2622

OUTPUT_NODES = 4000
LAYERS = [INPUT_NODES, 5000, 4000, OUTPUT_NODES]


def id_to_pos(node):
    """Mappa i primi 784 nodi (pixel) a coordinate 28x28.
       I nodi extra non hanno posizione."""
    if node < PIXEL_INPUT:
        return (node // 28, node % 28)
    return None  # feature extra


def comp_prob(x, y):
    return pow((min(x, 28-x) + min(y, 28-y)), 2.5)


def main():
    print(sum(LAYERS))
    first_layer_node = 1

    for layer in range(1, len(LAYERS)):
        prev_size = LAYERS[layer-1]
        next_size = LAYERS[layer]

        nl = [x for x in range(first_layer_node, first_layer_node + min(prev_size, next_size))]

        if(next_size > prev_size):
            while len(nl) < next_size:
                nl.append(randint(first_layer_node, first_layer_node + prev_size-1))
            shuffle(nl)

            nr = []
            if layer == 1:
                prob = []
                for node in range(first_layer_node, first_layer_node + prev_size):
                    pos = id_to_pos(node - 1)  # node-1 perché parte da 1
                    if pos is None:
                        p = 1.0  # feature extra → probabilità uniforme
                    else:
                        x, y = pos
                        p = comp_prob(x, y)
                    prob.append(p)

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
        else:
            nl = [randint(first_layer_node, first_layer_node + prev_size-1) for _ in range(0, next_size)]
            nr = [randint(first_layer_node, first_layer_node + prev_size-1) for _ in range(0, next_size)]
            first_layer_node += prev_size


        for l, r in zip(nl, nr):
            print(str(l) + " " + str(r))

    print(OUTPUT_NODES)


if __name__ == "__main__":
    main()
