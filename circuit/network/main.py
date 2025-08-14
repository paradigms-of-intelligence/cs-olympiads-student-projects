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
EPOCH_COUNT = 20
INPUT_SIZE = 784
BATCH_SIZE = 16

# Training constants
ALPHA = 0.001
BETA2 = .999
BETA1 = .9
EPSILON = 1e-4
LEARNING_RATE = 0.01
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
    return sum


@jax.jit
def loss_function(prob, values, correct_answer, left_nodes, right_nodes):
    '''Run forward pass and return MSE between outputs and correct_answer.'''
    layer_inference = jax.vmap(inference_function, in_axes=(0, 0, 0, None))
     

    start_of_current_layer = 1
    for c in range(len(prob)):
        end_of_current_layer = start_of_current_layer+len(prob[c])

        aus = layer_inference(prob[c], left_nodes[c], right_nodes[c], values)

        values = values.at[start_of_current_layer:end_of_current_layer].set(aus)
        end_of_current_layer = start_of_current_layer

    outputs = jnp.array([values[node] for node in OUTPUT_NODES])

    return jnp.mean(jnp.square(outputs - correct_answer))

@jax.jit
def scalar_loss(prob, values, correct_answer, left_nodes, right_nodes):
    '''Vectorize loss_function over the batch and return mean loss.'''
    prob = jax.nn.softmax(prob, 1)
    batch_loss = jax.vmap(loss_function, in_axes=(None, 0, 0, None)) 
    loss = batch_loss(prob, values, correct_answer, left_nodes, right_nodes)
    return jnp.mean(loss)


def main():
    '''Load architecture, train for EPOCH_COUNT epochs, and save network.'''

    #Cache jit compilation
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

    global BETA1_TIMESTAMP, BETA2_TIMESTAMP, BATCH_SIZE, EPSILON, LEARNING_RATE, INPUT_SIZE, OUTPUT_NODES

    left_nodes = []
    right_nodes = []
    prob = []

    # Load network architecture
    with open("network_architecture.txt", 'r') as file:
        global NETWORK_SIZE, LEFT_NODES, RIGHT_NODES, EPOCH_COUNT

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

        l = [0 for _ in range(0, NETWORK_SIZE+1)]
        aus = []
        att = 0
        for id in range(INPUT_SIZE+1, NETWORK_SIZE+1):
            if l[LEFT[id]] == att or l[RIGHT[id]] == att: 
                att+=1
                l.append([])
            l[-1].append(id)

        
        for x in aus:
            left_nodes.append([])
            right_nodes.append([])
            prob.append([])
            for id in x:
                left_nodes[-1].append(LEFT[id])
                right_nodes[-1].append(RIGHT[id])
                prob[-1].append([random.random() for _ in range(16)])


    LEFT_NODES = jnp.array(left_nodes)
    RIGHT_NODES = jnp.array(right_nodes)
    PROBS = jnp.array(prob)

    prob = jnp.array([[random.random() for _ in range (16)] for _ in range(NETWORK_SIZE)], dtype=jnp.float16)
    
    BETA1_TIMESTAMP = 1
    BETA2_TIMESTAMP = 1
    # Start training routine
    for epoch in range (0, EPOCH_COUNT):
        print("Epoch " + str(epoch+1))
        # Update timestamps (these represent BETA^t)
        BETA1_TIMESTAMP *= BETA1
        BETA2_TIMESTAMP *= BETA2

        # Initialize optimizer
        optimizer  = optax.adam(learning_rate=LEARNING_RATE, b1=BETA1_TIMESTAMP, b2=BETA2_TIMESTAMP, eps=EPSILON) 
        opt_state = optimizer.init(PROBS)

        # Initialize values to 0 for the batch
        values = jnp.zeros((BATCH_SIZE, NETWORK_SIZE), dtype=jnp.float16)
        answer = []

        # For each image in the batch read training data
        # TODO: Stop reading from BATCH_SIZE files, just read a single file with every input
        
        with open("../data/testdata.txt", 'r') as file:
            for test_case in range(0, BATCH_SIZE):

                # read training input
                line = list(map(float, file.readline().strip()))

                for id in range (1, INPUT_SIZE+1):
                    values = values.at[test_case, id].set(line[id-1])

                # Setting the correct answer
                ans = int(file.readline().strip())

                answer.append([0 for _ in range(len(OUTPUT_NODES))])
                answer[test_case][ans] = 1
                print("Test case " + str(test_case))
            
        correct_answer = jnp.array(answer)

        # Forward pass
        loss_value = scalar_loss(PROBS, values, correct_answer, LEFT_NODES, RIGHT_NODES)
        print("Loss value: " + str(loss_value))

        # Backward pass
        # TODO: CRITICAL AND SUS, Optimize this please, it takes minutes per epoch
        gradients = jax.grad(scalar_loss, argnums=0)(PROBS, values, correct_answer, LEFT_NODES, RIGHT_NODES)
        
        # Update parameters
        updates, opt_state = optimizer.update(gradients, opt_state)
        PROBS = optax.apply_updates(PROBS, updates)
        
        


    # Print the network
    with open("trained_network.bin", "wb") as f:
        # Write the network size -> 32 bits/ 4 bytes
        f.write(NETWORK_SIZE.to_bytes(4, byteorder = 'little'))

        # Write the network gates
        for id in range(INPUT_SIZE+1, NETWORK_SIZE+1):
            # f.write(jnp.argmax(prob[id])[0].to_bytes(4, byteorder = 'little', signed=True))
            gate_index = int(jnp.argmax(prob[id]))  # int(...) also works
            f.write(gate_index.to_bytes(4, byteorder='little', signed=True))

            f.write(id.to_bytes(4, byteorder='little', signed=True))
            f.write(int(LEFT_NODES[id]).to_bytes(4, byteorder='little', signed=True))
            f.write(int(RIGHT_NODES[id]).to_bytes(4, byteorder='little', signed=True))

        for id in range (NETWORK_SIZE-9, NETWORK_SIZE+1):
            f.write(id.to_bytes(4, byteorder='little', signed=True))


if __name__ == "__main__":
    main()
