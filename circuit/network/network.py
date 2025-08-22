import random, os
from config import *
import jax.numpy as jnp
import jax, math, functools, optax

#Cache jit compilation
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


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

OUTPUT_NODES_LIST = []

# Training input parameters


# This should be multiplied by BETA1 and BETA2
# and be updated for each iteration
    
@jax.jit
def inference_function(p, left, right, values):
    '''Compute gate output from inputs a, b and 16-element prob vector p.'''
    pr = values[left]*values[right]
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
def fitting_function(a):
    SUS = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) + 1.0
    return jnp.multiply(a, SUS)

layer_inference = jax.jit(jax.vmap(inference_function, in_axes=(0, 0, 0, None)))
batch_fitting_function = jax.jit(jax.vmap(fitting_function, in_axes=(0)))

@jax.jit
def inference(prob, left_nodes, right_nodes, inputs):
    global OUTPUT_NODES_LIST
    start_of_current_layer = INPUT_NODES + 1

    values = jnp.zeros((NETWORK_SIZE + 1,))
    values = values.at[1:INPUT_NODES+1].set(inputs)

    # Layer loop
    for c in range(1, len(prob)):
        end_of_current_layer = start_of_current_layer + len(prob[c])
        aus = layer_inference(prob[c], left_nodes[c], right_nodes[c], values)
        values = values.at[start_of_current_layer:end_of_current_layer].set(aus)
        start_of_current_layer = end_of_current_layer
    
    # Compute category means (MNIST: 10 classes)
    category_size = OUTPUT_NODES // 10
    outputs = []   # ← correct variable name

    for cat in range(10):
        node_ids = jnp.array(OUTPUT_NODES_LIST[cat * category_size : (cat + 1) * category_size])
        category_values = values[node_ids]
        outputs.append(jnp.mean(category_values))

    outputs = jnp.array(outputs, jnp.float32)

    return outputs


@jax.jit
def loss_function(prob, values, correct_answer, left_nodes, right_nodes):
    '''Run forward pass and return loss between outputs and correct_answer.'''
    result = inference(prob, left_nodes, right_nodes, values)
    onehotresult = jax.nn.one_hot(jnp.argmax(result), 10)
    return optax.softmax_cross_entropy(result, correct_answer.astype(jnp.float32)), jnp.all(onehotresult == correct_answer).astype(jnp.float32)

def layer_normalize(prob):
    '''Normalize the probabilities to all 0 and a 1'''
    max_idx = jnp.argmax(prob)
    return jnp.eye(prob.shape[0])[max_idx]

batch_layer_normalize = jax.jit(jax.vmap(layer_normalize, in_axes=(0,)))


@jax.jit
def accuracy_function(prob, values, correct_answer, left_nodes, right_nodes):
    '''Run forward pass and return accuracy between outputs and correct_answer.'''
    helper = jnp.array(inference(prob, left_nodes, right_nodes, values))
    predicted = (jnp.argmax(helper)).astype(jnp.int32)
    correct = jnp.argmax(correct_answer).astype(jnp.int32)
    return (predicted == correct).astype(jnp.float32)


@jax.jit
def scalar_loss(prob, values, correct_answer, left_nodes, right_nodes, temperature):
    '''Vectorize loss_function over the batch and return mean loss.'''

    for i in range(len(prob)):
        prob[i] = batch_fitting_function(prob[i]) * temperature
        prob[i] = jax.nn.softmax(prob[i], 1)

    batch_loss = jax.vmap(loss_function, in_axes=(None, 0, 0, None, None)) 
    loss, accuracy = batch_loss(prob, values, correct_answer, left_nodes, right_nodes)
    jax.debug.print("{}", jnp.mean(accuracy))
    return jnp.mean(loss)

def input_network(left_nodes, right_nodes, prob, aus):
    global OUTPUT_NODES_LIST

    random.seed(0)

    with open(NETWORK_ARCHITECTURE_FILE, 'r') as file:        
        # Initialize network size 
        # Initialize null connections of input nodes
        LEFT = [-1 for x in range(0, INPUT_NODES+1)]
        RIGHT = [-1 for x in range(0, INPUT_NODES+1)]    

        # Initialize all other nodes
        for l in range (INPUT_NODES+1, NETWORK_SIZE+1):
            left, right = map(int, file.readline().strip().split())
            LEFT.append(left)
            RIGHT.append(right)

        # Initialize output nodes (assumed to be the last N)
        OUTPUT_NODES_LIST = [x for x in range(NETWORK_SIZE-OUTPUT_NODES+1, NETWORK_SIZE+1)]

        # Calculate the layers
        l = [0 for _ in range(0, NETWORK_SIZE+1)]
        att = 0
        for id in range(INPUT_NODES+1, NETWORK_SIZE+1):
            cl = max(l[LEFT[id]], l[RIGHT[id]])
            if cl >= att: 
                att+=1
                aus.append([])
            aus[cl].append(id)
            l[id] = cl+1


        # Initialize layers
        left_nodes.append(jnp.zeros((INPUT_NODES+1), dtype=jnp.float32))
        right_nodes.append(jnp.zeros((INPUT_NODES+1), dtype=jnp.float32))
        prob.append(jnp.zeros((INPUT_NODES+1, 16), dtype=jnp.float32))
        for x in aus:
            left = []
            right = []
            p = []
            for id in x:
                left.append(LEFT[id])
                right.append(RIGHT[id])
                p.append([0.06 for index in range(16)])
                p[-1][random.randint(1,2)*2+1] = 2

            left_nodes.append(jnp.array(left))
            right_nodes.append(jnp.array(right))
            prob.append(jnp.array(p))
        
    print("Layers:", len(aus))

def read_values(file):
    if os.path.exists(file + ".values.npy"):
        return jnp.load(file + ".values.npy"), jnp.load(file + ".answers.npy")

    values = []
    answers = []
    
    with open(file, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break

            # read training inputanswer
            line = list(map(float, line.strip()))

            # one row: pad with zero in col 0, then inputs
            row = line
            values.append(row)

            # Setting the correct answer
            ans = int(f.readline().strip())
            one_hot = [0] * 10
            one_hot[ans] = 1
            answers.append(one_hot)
    
    values = jnp.array(values)
    answers = jnp.array(answers)
    jnp.save(file + ".values.npy", values)
    jnp.save(file + ".answers.npy", answers)

    return values, answers
    
#@jax.jit
def train_network(prob, left_nodes, right_nodes):
    global OUTPUT_NODES_LIST
    # Initialize for testing
    values_list_testing, answers_list_testing = read_values(TESTDATA_DATA_FILE)
    values_testing = jnp.array(values_list_testing, dtype=jnp.float32)
    correct_answer_testing = jnp.array(answers_list_testing, dtype=jnp.float32)

    batch_accuracy_testing = jax.vmap(accuracy_function, in_axes=(None, 0, 0, None, None)) 


    # Read training data
    values_list = [[] for _ in range(round(TOTAL_SIZE/BATCH_SIZE))]
    answers_list = [[] for _ in range(round(TOTAL_SIZE/BATCH_SIZE))]

    STEPS_PER_EPOCH  = TOTAL_SIZE // BATCH_SIZE
    TOTAL_STEPS = STEPS_PER_EPOCH * EPOCH_COUNT

    print("Reading values")       
    values_list, answers_list = read_values(TRAINING_DATA_FILE)


    optimizer  = optax.adamw(
        optax.exponential_decay(LEARNING_RATE,
                                transition_steps = TOTAL_STEPS,
                                decay_rate = 0.1),
        weight_decay=WEIGHT_DECAY
    )
    opt_state = optimizer.init(prob)

    value_and_grad = jax.jit(jax.value_and_grad(scalar_loss))

    assert TOTAL_SIZE % BATCH_SIZE == 0

    # Start training routine
    for epoch in range (0, EPOCH_COUNT):
        # Forward pass

        values_list = jax.random.permutation(jax.random.PRNGKey(epoch), values_list)
        answers_list = jax.random.permutation(jax.random.PRNGKey(epoch), answers_list)

        values = jnp.reshape(values_list, (STEPS_PER_EPOCH, BATCH_SIZE, -1))
        answers = jnp.reshape(answers_list, (STEPS_PER_EPOCH, BATCH_SIZE, -1))


        loss_sum = 0
        accuracy_sum = 0
        for i in range(STEPS_PER_EPOCH):
            temperature = MAX_TEMPERATURE**((epoch * STEPS_PER_EPOCH + i)/TOTAL_STEPS)


            (loss, gradients) = value_and_grad(prob, values[i], answers[i], left_nodes, right_nodes, temperature)
            loss_sum += loss

            # Update parameters
            updates, opt_state = optimizer.update(gradients, opt_state, params=prob)
            prob = optax.apply_updates(prob, updates)

        print("Epoch " + str(epoch+1) + " Loss: " + str(loss_sum * BATCH_SIZE / TOTAL_SIZE))

        
        if (epoch + 1) % 20 == 0:
            prob_testing  = [[] for _ in range(len(prob))]
            # Test the network on test data
            for i in range (len(prob)):
                prob_testing[i] = batch_layer_normalize(prob[i])
            acc = batch_accuracy_testing(prob_testing, values_testing, correct_answer_testing, left_nodes, right_nodes)
            print("Accuracy: " + str(float(jnp.mean(acc))))
    return prob

@jax.jit
def test_network(prob, left_nodes, right_nodes):
    global OUTPUT_NODES_LIST
    values_list, answers_list = read_values(TESTDATA_DATA_FILE)
    values = jnp.array(values_list, dtype=jnp.float32)
    correct_answer = jnp.array(answers_list, dtype=jnp.float32)

    for i in range (len(prob)): 
        prob[i] = batch_layer_normalize(prob[i])

    batch_accuracy = jax.vmap(accuracy_function, in_axes=(None, 0, 0, None, None)) 
    acc = batch_accuracy(prob, values, correct_answer, left_nodes, right_nodes)
    return jnp.mean(acc)


def print_network(aus, prob, left_nodes, right_nodes):
    global OUTPUT_NODES_LIST
    # # Print the network
    with open(TRAINED_NETWORK_16GATES_FILE, "wb") as f:
        # Write the network size -> 32 bits/ 4 bytes
        for current_layer in range(0, len(aus)):
            for i in range(0, len(aus[current_layer])):
                gate_index = int(jnp.argmax(prob[current_layer + 1][i]))
                f.write(gate_index.to_bytes(4, byteorder='little', signed=True))
                f.write(int(aus[current_layer][i]).to_bytes(4, byteorder='little', signed=True))
                f.write(int(left_nodes[current_layer + 1][i]).to_bytes(4, byteorder='little', signed=True))
                f.write(int(right_nodes[current_layer + 1][i]).to_bytes(4, byteorder='little', signed=True))


def run_training_sequence():
    '''Load architecture, train for EPOCH_COUNT epochs, and save network.'''

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
    


