import random, os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"   # don't grab all GPU memory at startup
import jax.numpy as jnp
import jax, math, functools, optax

#    ඞඞඞඞඞ SUUUUUUUS ඞඞඞඞඞ

# IDEAS:
# Instead of mapping connections with randint, try 
# mapping central pixels more often
#
# Instead of having a static input size of 784, try
# passing other statistics such as density of smaller
# squares and similar stuff. If you want to give more
# weight to a statistic, just feed the statistic nodes 
# to the network multiple times
#
# Simulate the first 10 epochs and re-map empty connections
# in the first layer to other input nodes
#
# Convolution sus
# 
# Try connecting nodes from earlier layers. For instance, suppose
# having a 8 layer network. At layer 5 connect the input nodes as 
# well. This allows the network to build some internal statistics
# 
#
# Extract images from a standard mnist to get a better idea of which
# statistics are more important.
# Network constants
NETWORK_SIZE = 0  # number of gates (set from file)
OUTPUT_SIZE = 0 # output size (set from file)
INPUT_SIZE = 784
OUTPUT_NODES = []
# Training input parameters
EPOCH_COUNT = 55
TOTAL_SIZE = 7400
BATCH_SIZE = 200

# Training constants
BETA2 = .99
BETA1 = .9
EPSILON = 1e-5
LEARNING_RATE = 0.04
LEARNING_INCREASE = 1
TEMPERATURE = 1

# This should be multiplied by BETA1 and BETA2
# and be updated for each iteration
    
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
    return 1/(1+jnp.exp(-10*(sum-0.5)))

@jax.jit
def fitting_function(a):
    global TEMPERATURE, LEARNING_INCREASE
    SUS = jnp.array([0.1, 0.1, 0.1, 0.11, 0.1, 0.11, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])    
    SUS = SUS + TEMPERATURE
    TEMPERATURE *= LEARNING_INCREASE
    return jnp.multiply(a, SUS)

layer_inference = jax.jit(jax.vmap(inference_function, in_axes=(0, 0, 0, None)))
batch_fitting_function = jax.jit(jax.vmap(fitting_function, in_axes=(0)))

@jax.jit
def inference(prob, left_nodes, right_nodes, values):
    global INPUT_SIZE, OUTPUT_NODES, OUTPUT_SIZE
    start_of_current_layer = INPUT_SIZE + 1
    
    # Convert values from bool to float.32
    values = values.astype(jnp.float32)

    # Layer loop
    for c in range(1, len(prob)):
        end_of_current_layer = start_of_current_layer + len(prob[c])
        aus = layer_inference(prob[c], left_nodes[c], right_nodes[c], values)
        values = values.at[start_of_current_layer:end_of_current_layer].set(aus)
        start_of_current_layer = end_of_current_layer
    
    # Compute category means (MNIST: 10 classes)
    category_size = OUTPUT_SIZE // 10
    outputs = []   # ← correct variable name

    for cat in range(10):
        node_ids = jnp.array(OUTPUT_NODES[cat * category_size : (cat + 1) * category_size])
        category_values = values[node_ids]
        outputs.append(jnp.mean(category_values))

    outputs = jnp.array(outputs, jnp.float32)

    return outputs


@jax.jit
def loss_function(prob, values, correct_answer, left_nodes, right_nodes):
    '''Run forward pass and return loss between outputs and correct_answer.'''
    return optax.softmax_cross_entropy(inference(prob, left_nodes, right_nodes, values), correct_answer.astype(jnp.float32))

def layer_normalize(prob):
    '''Normalize the probabilities to all 0 and a 1'''
    max_idx = jnp.argmax(prob)
    return jnp.eye(prob.shape[0])[max_idx]


@jax.jit
def accuracy_function(prob, values, correct_answer, left_nodes, right_nodes):
    '''Run forward pass and return accuracy between outputs and correct_answer.'''
    helper = jnp.array(inference(prob, left_nodes, right_nodes, values))
    predicted = (jnp.argmax(helper)).astype(jnp.int32)
    correct = jnp.argmax(correct_answer).astype(jnp.int32)
    return (predicted == correct).astype(jnp.float32)


@jax.jit
def accuracy_function(prob, values, correct_answer, left_nodes, right_nodes):
    '''Run forward pass and return accuracy between outputs and correct_answer.'''
    helper = jnp.array(inference(prob, left_nodes, right_nodes, values))
    predicted = (jnp.argmax(helper)).astype(jnp.int32)
    correct = jnp.argmax(correct_answer).astype(jnp.int32)
    return (predicted == correct).astype(jnp.float32)

@jax.jit
def scalar_loss(prob, values, correct_answer, left_nodes, right_nodes):
    '''Vectorize loss_function over the batch and return mean loss.'''

    for i in range(len(prob)):
        prob[i] = batch_fitting_function(prob[i])
        prob[i] = jax.nn.softmax(prob[i], 1)

    batch_loss = jax.vmap(loss_function, in_axes=(None, 0, 0, None, None)) 
    loss = batch_loss(prob, values, correct_answer, left_nodes, right_nodes)
    return jnp.mean(loss)

def input_network(left_nodes, right_nodes, prob, aus):
    global INPUT_SIZE, NETWORK_SIZE, OUTPUT_NODES, OUTPUT_SIZE

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
        OUTPUT_SIZE = int(file.readline().strip())
        OUTPUT_NODES = [x for x in range(NETWORK_SIZE-OUTPUT_SIZE+1, NETWORK_SIZE+1)]

        # Calculate the layers
        l = [0 for _ in range(0, NETWORK_SIZE+1)]
        att = 0
        for id in range(INPUT_SIZE+1, NETWORK_SIZE+1):
            cl = max(l[LEFT[id]], l[RIGHT[id]])
            if cl >= att: 
                att+=1
                aus.append([])
            aus[cl].append(id)
            l[id] = cl+1


        # Initialize layers
        left_nodes.append(jnp.zeros((INPUT_SIZE+1), dtype=jnp.float32))
        right_nodes.append(jnp.zeros((INPUT_SIZE+1), dtype=jnp.float32))
        prob.append(jnp.zeros((INPUT_SIZE+1, 16), dtype=jnp.float32))
        for x in aus:
            left = []
            right = []
            p = []
            for id in x:
                left.append(LEFT[id])
                right.append(RIGHT[id])
                p.append([random.uniform(0, 1) for index in range(16)])

            left_nodes.append(jnp.array(left))
            right_nodes.append(jnp.array(right))
            prob.append(jnp.array(p))
        
    print(len(aus))

def read_values(file, values, answers):
    with open(file, 'r') as file:
        for test_case in range(BATCH_SIZE):
            # read training inputanswer
            line = list(map(float, file.readline().strip()))

            # one row: pad with zero in col 0, then inputs
            row = [0.0] + line + ([0.0] * (NETWORK_SIZE - INPUT_SIZE))
            values.append(row)

            # Setting the correct answer
            ans = int(file.readline().strip())
            one_hot = [0] * 10
            one_hot[ans] = 1
            answers.append(one_hot)
    
    return values, answers
    
#@jax.jit
def train_network(prob, left_nodes, right_nodes):
    global LEARNING_RATE, BETA1, BETA2, EPSILON, EPOCH_COUNT, OUTPUT_NODES, INPUT_SIZE
    # Initialize for testing
    values_list_testing = []
    answers_list_testing = []
    read_values("../data/testdata.txt", values_list_testing, answers_list_testing)
    values_testing = jnp.array(values_list_testing, dtype=jnp.float32)
    correct_answer_testing = jnp.array(answers_list_testing, dtype=jnp.float32)

    batch_accuracy_testing = jax.vmap(accuracy_function, in_axes=(None, 0, 0, None, None)) 


    # Read training data
    values_list = [[] for _ in range(round(TOTAL_SIZE/BATCH_SIZE))]
    answers_list = [[] for _ in range(round(TOTAL_SIZE/BATCH_SIZE))]

    print("Reading values")       
    for i in range (0, round(TOTAL_SIZE/BATCH_SIZE)): 
        values_list[i],answers_list[i] = read_values("../data/training.txt", values_list[i], answers_list[i])
    print("Values read")       


    optimizer  = optax.adamw(LEARNING_RATE) 
    opt_state = optimizer.init(prob)


    values_list = jnp.array(values_list, dtype=jnp.bool)
    answers_list = jnp.array(answers_list, dtype=jnp.bool)

    # Start training routine
    for epoch in range (0, EPOCH_COUNT):
        # Forward pass

        values_list = jax.random.permutation(jax.random.PRNGKey(epoch), values_list)
        answers_list = jax.random.permutation(jax.random.PRNGKey(epoch), answers_list)


        loss_sum = 0
        for i in range (0, round(TOTAL_SIZE/BATCH_SIZE)):
            loss_sum  += scalar_loss(prob, values_list[i], answers_list[i], left_nodes, right_nodes)

            # Backward pass
            gradients = jax.grad(scalar_loss)(prob, values_list[i], answers_list[i], left_nodes, right_nodes)
            
            # Update parameters
            updates, opt_state = optimizer.update(gradients, opt_state, params=prob)
            prob = optax.apply_updates(prob, updates)

        print("Epoch " + str(epoch+1) + " Loss: " + str(loss_sum * BATCH_SIZE / TOTAL_SIZE))

        
        if (epoch + 1) % 20 == 0:
            prob_testing  = [[] for _ in range(len(prob))]
            # Test the network on test data
            for i in range (len(prob)):
                prob_testing[i] = jax.nn.softmax(prob[i])
            acc = batch_accuracy_testing(prob_testing, values_testing, correct_answer_testing, left_nodes, right_nodes)
            print("Accuracy: " + str(float(jnp.mean(acc))))
    return prob

@jax.jit
def test_network(prob, left_nodes, right_nodes):
    global BATCH_SIZE, OUTPUT_NODES, INPUT_SIZE
    values_list = []
    answers_list = []
    read_values("../data/testdata.txt", values_list, answers_list)
    values = jnp.array(values_list, dtype=jnp.float32)
    correct_answer = jnp.array(answers_list, dtype=jnp.float32)

    batch_layer_normalize = jax.vmap(layer_normalize, in_axes=(0,))
    for i in range (len(prob)): 
        prob[i] = batch_layer_normalize(prob[i])

    batch_accuracy = jax.vmap(accuracy_function, in_axes=(None, 0, 0, None, None)) 
    acc = batch_accuracy(prob, values, correct_answer, left_nodes, right_nodes)
    return jnp.mean(acc)


def print_network(aus, prob, left_nodes, right_nodes):
    global NETWORK_SIZE, OUTPUT_NODES
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

        # Write the OUTPUT_SIZE and the output node ids (UNCOMMENTED)
        f.write(OUTPUT_SIZE.to_bytes(4, byteorder='little', signed=True))
        for id in (OUTPUT_NODES):
            f.write(int(id).to_bytes(4, byteorder='little', signed=True))


    with open("sus.txt", "w") as f:
        f.write("Newline\n")
        # Write the network size -> 32 bits/ 4 bytes
        f.write(str(NETWORK_SIZE) + "\n")
        for current_layer in range(0, len(aus)):
            for i in range(0, len(aus[current_layer])):
                gate_index = int(jnp.argmax(prob[current_layer + 1][i]))
                f.write(str(gate_index) + " ")
                f.write(str(int(aus[current_layer][i])) + " ")
                f.write(str(int(left_nodes[current_layer + 1][i])) + " ")
                f.write(str(int(right_nodes[current_layer + 1][i])) + "\n")

        f.write(str(OUTPUT_SIZE) + "\n")
        for id in (OUTPUT_NODES):
            f.write(str(id) + " ")


def main():
    '''Load architecture, train for EPOCH_COUNT epochs, and save network.'''

    #Cache jit compilation
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

    # Setup network architecture
    left_nodes = []
    right_nodes = []
    prob = []
    aus = []

    input_network(left_nodes, right_nodes, prob, aus)

    #Train network
    prob = train_network(prob, left_nodes, right_nodes)

    # Test network on testadata
    print(test_network(prob, left_nodes, right_nodes))

    # Print the network to binary file
    print_network(aus, prob, left_nodes, right_nodes)
    

if __name__ == "__main__":
    main()
