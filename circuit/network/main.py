import random 
import jax.numpy as jnp
import jax, math, functools, optax


# Network constants
NETWORK_SIZE = 0  # number of gates (set from file)
OUTPUT_NODES = []
LEFT_NODES = []
RIGHT_NODES = []

# Training input parameters
EPOCH_COUNT = 1
INPUT_SIZE = 784
BATCH_SIZE = 16

# Training constants
ALPHA = 0.001
BETA2 = .999
BETA1 = .9
EPSILON = 1e-4
LEARNING_RATE = 0.01
# timestamps used in the script
BETA1_TIMESTAMP = 0
BETA2_TIMESTAMP = 0

# basic inference function for probabilities

@jax.jit
def inference_function(a: float, b: float, p):
    '''Compute gate output from inputs a, b and 16-element prob vector p.'''
    pr = a*b

    # f = [0 for _ in range (0, 16)]
    # f[0] = 0
    # f[1] = pr*p[1]
    # f[2] = (a-pr)*p[2]
    # f[3] = a*p[3]
    # f[4] = (b-pr)*p[4]
    # f[5] = b*p[5]
    # f[6] = (a+b-2*pr)*p[6]
    # f[7] = (a+b-pr)*p[7]
    # f[8] = (1-a-b+pr)*p[8]
    # f[9] = (1-a-b+2*pr)*p[9]
    # f[10] = (1-b)*p[10]
    # f[11] = (1-b+pr)*p[11]
    # f[12] = (1-a)*p[12]
    # f[13] = (1-a+pr)*p[13]
    # f[14] = (1-pr)*p[14]
    # f[15] = p[15]

    # TODO: Test if using an Hadamard product is optimized better than this
    sum = (
    pr * p[1]
    + (a - pr) * p[2]
    + a * p[3]
    + (b - pr) * p[4]
    + b * p[5]
    + (a + b - 2 * pr) * p[6]
    + (a + b - pr) * p[7]
    + (1 - a - b + pr) * p[8]
    + (1 - a - b + 2 * pr) * p[9]
    + (1 - b) * p[10]
    + (1 - b + pr) * p[11]
    + (1 - a) * p[12]
    + (1 - a + pr) * p[13]
    + (1 - pr) * p[14]
    + p[15]
    )
    return sum

@jax.jit
def loss_function(prob, values, correct_answer):
    '''Run forward pass and return MSE between outputs and correct_answer.'''

    prob_copy = []
    values_copy = []
    for i in range(INPUT_SIZE+1,NETWORK_SIZE+1):
        prob_copy.append(jax.nn.softmax(prob[i]))
        values_copy.append(inference_function(values[LEFT_NODES[i]], values[RIGHT_NODES[i]], prob_copy[-1]))

    prob = jnp.array(prob_copy)
    values = jnp.array(values_copy)

    outputs = jnp.array([values[node] for node in OUTPUT_NODES])

    return jnp.mean(jnp.square(outputs - correct_answer))

@jax.jit
def scalar_loss(prob, values, correct_answer):
    '''Vectorize loss_function over the batch and return mean loss.'''
    batch_loss = jax.vmap(loss_function, in_axes=(None, 0, 0)) 
    loss = batch_loss(prob, values, correct_answer)
    return jnp.mean(loss)


def main():
    '''Load architecture, train for EPOCH_COUNT epochs, and save network.'''

    #Cache jit compilation
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

    global BETA1_TIMESTAMP, BETA2_TIMESTAMP, BATCH_SIZE, EPSILON, LEARNING_RATE, INPUT_SIZE, OUTPUT_NODES

    # Load network architecture
    with open("network_architecture.txt", 'r') as file:
        global NETWORK_SIZE, LEFT_NODES, RIGHT_NODES, EPOCH_COUNT

        # Initialize network size 
        NETWORK_SIZE = int(file.readline().strip())

        # Initialize null connections of input nodes
        LEFT_NODES = [-1 for x in range(0, INPUT_SIZE+1)]
        RIGHT_NODES = [-1 for x in range(0, INPUT_SIZE+1)]    

        # Initialize all other nodes
        for _ in range (INPUT_SIZE+1, NETWORK_SIZE+1):
            left, right = map(int, file.readline().strip().split())
            LEFT_NODES.append(left)
            RIGHT_NODES.append(right)

        # Initialize output nodes (assumed to be the last N)
        number_outputs = int(file.readline().strip())
        OUTPUT_NODES = [x for x in range(NETWORK_SIZE-number_outputs+1, NETWORK_SIZE+1)]

    # Initialize the network with random probabilities


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
        opt_state = optimizer.init(prob)

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
        loss_value = scalar_loss(prob, values, correct_answer)
        print("Loss value: " + str(loss_value))

        # Backward pass
        # TODO: CRITICAL AND SUS, Optimize this please, it takes minutes per epoch
        gradients = jax.grad(scalar_loss, argnums=0)(prob, values, correct_answer)
        
        # Update parameters
        updates, opt_state = optimizer.update(gradients, opt_state)
        prob = optax.apply_updates(prob, updates)
        
        


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
