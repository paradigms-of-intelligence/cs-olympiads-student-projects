import random 
import jax.numpy as jnp
import jax, math, functools, optax

#ඞඞඞඞඞ SUUUUUUUS ඞඞඞඞඞඞඞඞඞඞඞ
# Network constants
NETWORK_SIZE = 0  # number of gates (set from file)
OUTPUT_NODES = []
LEFT_NODES = []
RIGHT_NODES = []

# Training input parameters
EPOCH_COUNT = 200
INPUT_SIZE = 784
BATCH_SIZE = 10000

# Training constants
ALPHA = 0.001
BETA2 = .999
BETA1 = .9
EPSILON = 1e-8
LEARNING_RATE = 0.25
# This should be multiplied by BETA1 and BETA2
# and be updated for each iteration
BETA1_TIMESTAMP = 0
BETA2_TIMESTAMP = 0
    
# basic inference function for probabilities
@jax.jit
def inference_function(p, left, right, values):
    '''Compute gate output from inputs a, b and 16-element prob vector p.'''
    pr = values[left]*values[right]
    # TODO: Test if using an Hadamard product is optimized better than this
    sum = (
    pr * p[1]
    + (values[left] - pr) * p[2]
    + values[left] * p[3]
    + (values[right] - pr) * p[4]
    + values[right] * p[5]
    + (values[left] + values[right] - 2 * pr) * p[6]
    + (values[left] + values[right] - pr) * p[7]
    + (1 - values[left] - values[right] + pr) * p[8]
    + (1 - values[left] - values[right] + 2 * pr) * p[9]
    + (1 - values[right]) * p[10]
    + (1 - values[right] + pr) * p[11]
    + (1 - values[left]) * p[12]
    + (1 - values[left] + pr) * p[13]
    + (1 - pr) * p[14]
    + p[15]
    )
    #jax.debug.print("{}", sum)
    return sum

layer_inference = jax.jit(jax.vmap(inference_function, in_axes=(0, 0, 0, None)))


@jax.jit
def loss_function(prob, values, correct_answer, left_nodes, right_nodes):
    '''Run forward pass and return MSE between outputs and correct_answer.'''
    start_of_current_layer = 785

    for c in range(1, len(prob)):
        end_of_current_layer = start_of_current_layer+len(prob[c])

        aus = layer_inference(prob[c], left_nodes[c], right_nodes[c], values)

        values = values.at[start_of_current_layer:end_of_current_layer].set(aus)
        #jax.debug.print("{}", values)
        start_of_current_layer = end_of_current_layer

    outputs = jnp.array([values[node] for node in OUTPUT_NODES])

    return jnp.mean(jnp.sqrt(jnp.abs(outputs - correct_answer)))


@jax.jit
def fitting_function(a):

    SUS = jnp.array([0.02, 0.5, 0.5, 0.6, 0.5, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.02])    
    # SUS = jnp.array([0.02, 0.5, 0.5, 0.6, 0.5, 0.6, 0, 0, 0, 0, 0.5, 0, 0.5, 0, 0, 0.02])    

    return jnp.multiply(a, SUS)

@jax.jit
def scalar_loss(prob, values, correct_answer, left_nodes, right_nodes):
    '''Vectorize loss_function over the batch and return mean loss.'''

    batch_fitting_function = jax.vmap(fitting_function, in_axes=(0))

    for i in range(len(prob)):
        prob[i] = batch_fitting_function(prob[i])
        prob[i] = jax.nn.softmax(prob[i], 1)

    batch_loss = jax.vmap(loss_function, in_axes=(None, 0, 0, None, None)) 
    loss = batch_loss(prob, values, correct_answer, left_nodes, right_nodes)
    return jnp.sum(loss)

def main():
    '''Load architecture, train for EPOCH_COUNT epochs, and save network.'''

    #Cache jit compilation
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

    global BETA1_TIMESTAMP, BETA2_TIMESTAMP, BATCH_SIZE, EPSILON, LEARNING_RATE, INPUT_SIZE, OUTPUT_NODES
    global NETWORK_SIZE, LEFT_NODES, RIGHT_NODES, EPOCH_COUNT

    left_nodes = []
    right_nodes = []
    prob = []
    aus = []
    # Load network architecture
    with open("network_architecture.txt", 'r') as file:

        # Initialize network size 
        NETWORK_SIZE = int(file.readline().strip())

        # Initialize null connections of input nodes
        LEFT = [-1 for x in range(0, INPUT_SIZE+1)]
        RIGHT = [-1 for x in range(0, INPUT_SIZE+1)]    

        # Initialize all other nodes
        for _ in range (INPUT_SIZE+1, NETWORK_SIZE+1):
            left, right = map(int, file.readline().strip().split())
            LEFT.append(left)
            RIGHT.append(right)

        # Initialize output nodes (assumed to be the last N)
        number_outputs = int(file.readline().strip())
        OUTPUT_NODES = [x for x in range(NETWORK_SIZE-number_outputs+1, NETWORK_SIZE+1)]

        # Calculate the layers
        l = [0 for _ in range(0, NETWORK_SIZE+1)]
        att = 0
        for id in range(INPUT_SIZE+1, NETWORK_SIZE+1):
            if l[LEFT[id]] == att or l[RIGHT[id]] == att: 
                att+=1
                aus.append([])
            aus[-1].append(id)
            l[id] = att


        # Initialize layers
        left_nodes.append(jnp.zeros((INPUT_SIZE+1), dtype=jnp.int32))
        right_nodes.append(jnp.zeros((INPUT_SIZE+1), dtype=jnp.int32))
        prob.append(jnp.zeros((INPUT_SIZE+1, 16), dtype=jnp.float32))
        for x in aus:
            left = []
            right = []
            p = []
            for id in x:
                left.append(LEFT[id])
                right.append(RIGHT[id])
                p.append([0.1 for index in range(16)])
                p[-1][3] = 1

            left_nodes.append(jnp.array(left))
            right_nodes.append(jnp.array(right))
            prob.append(jnp.array(p))


    values = jnp.zeros((BATCH_SIZE, NETWORK_SIZE+1), dtype=jnp.float32)
    answer = []
    values_list = []
    answers_list = []

    with open("../data/training.txt", 'r') as file:
        for test_case in range(BATCH_SIZE):
            # read training input
            line = list(map(float, file.readline().strip()))

            # one row: pad with zero in col 0, then inputs
            row = [0.0] + line + ([0.0] * 2340)
            values_list.append(row)

            # Setting the correct answer
            ans = int(file.readline().strip())
            one_hot = [0] * len(OUTPUT_NODES)
            one_hot[ans] = 1
            answers_list.append(one_hot)

            print("Test case " + str(test_case))

    # Convert to JAX arrays once at the end
    values = jnp.array(values_list, dtype=jnp.float32)
    answer = jnp.array(answers_list, dtype=jnp.float32)
        

    correct_answer = jnp.array(answer)

    BETA1_TIMESTAMP = 1
    BETA2_TIMESTAMP = 1
    # Start training routine.append
    for epoch in range (0, EPOCH_COUNT):
        print("Epoch " + str(epoch+1))
        # Update timestamps (these represent BETA^t)
        BETA1_TIMESTAMP *= BETA1
        BETA2_TIMESTAMP *= BETA2

        # Initialize optimizer
        optimizer  = optax.adam(learning_rate=LEARNING_RATE, b1=BETA1_TIMESTAMP, b2=BETA2_TIMESTAMP, eps=EPSILON) 
        opt_state = optimizer.init(prob)

        # Forward pass
        loss_value = scalar_loss(prob, values, correct_answer, left_nodes, right_nodes)
        print("Mean value: " + str(loss_value))

        # Backward pass
        # TODO: CRITICAL AND SUS, Optimize this please, it takes minutes per epoch
        gradients = jax.grad(scalar_loss)(prob, values, correct_answer, left_nodes, right_nodes)
        
        # Update parameters
        updates, opt_state = optimizer.update(gradients, opt_state)
        prob = optax.apply_updates(prob, updates)

    # # Print the network
    with open("trained_network.bin", "wb") as f:

        
        # Write the network size -> 32 bits/ 4 bytes
        f.write(NETWORK_SIZE.to_bytes(4, byteorder = 'little'))
        for current_layer in range(0, len(aus)):
            for i in range(0, len(aus[current_layer])):

                gate_index = int(jnp.argmax(prob[current_layer + 1][i]))
                f.write(gate_index.to_bytes(4, byteorder='little', signed=True))

                f.write(int(aus[current_layer][i]).to_bytes(4, byteorder='little', signed=True))
                f.write(int(left_nodes[current_layer + 1][i]).to_bytes(4, byteorder='little', signed=True))
                f.write(int(right_nodes[current_layer + 1][i]).to_bytes(4, byteorder='little', signed=True))

        for id in range (NETWORK_SIZE-9, NETWORK_SIZE+1):
            f.write(id.to_bytes(4, byteorder='little', signed=True))
    
    for i in range(len(prob)):
        prob[i] = jax.nn.softmax(prob[i], 1)

    with open("sus.txt", "w") as f:
        # Write the network size -> 32 bits/ 4 bytes

        f.write(str(NETWORK_SIZE) + "\n\n")
        for current_layer in range(0, len(aus)):
            for i in range(0, len(aus[current_layer])):

                gate_index = int(jnp.argmax(prob[current_layer + 1][i]))
                f.write(str(gate_index) + " ")

                f.write(str(int(aus[current_layer][i])) + " ")
                f.write(str(int(left_nodes[current_layer + 1][i]))+ " ")
                f.write(str(int(right_nodes[current_layer + 1][i])) + " \n")

        f.write("\n");
        for id in range (NETWORK_SIZE-9, NETWORK_SIZE+1):
            f.write(str(id)+ " ")


        
        # Write the network gates
        # for id in range(INPUT_SIZE+1, NETWORK_SIZE+1):
        #     # f.write(jnp.argmax(prob[id])[0].to_bytes(4, byteorder = 'little', signed=True))
        #     gate_index = int(jnp.argmax(prob[id]))  # int(...) also works
        #     f.write(gate_index.to_bytes(4, byteorder='little', signed=True))

        #     f.write(id.to_bytes(4, byteorder='little', signed=True))
        #     f.write(int(LEFT_NODES[id]).to_bytes(4, byteorder='little', signed=True))
        #     f.write(int(RIGHT_NODES[id]).to_bytes(4, byteorder='little', signed=True))

        # for id in range (NETWORK_SIZE-9, NETWORK_SIZE+1):
        #     f.write(id.to_bytes(4, byteorder='little', signed=True))


if __name__ == "__main__":
    main()
