import random, os
import jax.numpy as jnp
import jax, math, functools, optax
from config import *

#Cache jit compilation
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

#Initialization of probabilities
def initialize_probabilities(index):
    p = [0.06 for index in range(16)]
    p[random.randint(1,2)*2+1] = float(2)
    return p


#Inference probability fitting
FITTING = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) + 1.0
batch_fitting_function = jax.jit(jax.vmap(lambda a: jnp.multiply(a, FITTING), in_axes = (0)))


DEBUG = 0

def inference_function(id, prob, prob_id, left, right, values):
    p = prob[prob_id[id]]
    l = values[left[id]]
    r = values[right[id]]
    sum = (
    l*r * p[1]
    + (l - l*r) * p[2]
    + l * p[3]
    + (r - l*r) * p[4]
    + r * p[5]
    + (l + r - 2 * l*r) * p[6]
    + (l + r - l*r) * p[7]
    + (1 - l - r + l*r) * p[8]
    + (1 - l - r + 2 * l*r) * p[9]
    + (1 - r) * p[10]
    + (1 - r + l*r) * p[11]
    + (1 - l) * p[12]
    + (1 - l + l*r) * p[13]
    + (1 - l*r) * p[14]
    + p[15]
    )
    return sum

layer_inference = jax.jit(jax.vmap(inference_function, in_axes = (0, None, None, None, None, None)))

@jax.jit
def binarizing_function(prob):
    # Returns a binarized probability distribution (one hot)
    max_id = jnp.argmax(prob)
    return jnp.eye(prob.shape[0])[max_id]

@jax.jit
def inference(layered_id, prob, prob_id, left_nodes, right_nodes, inputs):
    #Counter for the node range
    start_of_current_layer = INPUT_NODES+1
    #Setting input gate values
    values = jnp.zeros((NETWORK_SIZE + 1,))
    values = values.at[1:INPUT_NODES+1].set(inputs)

    #Iterate through layers
    for c in range(1, len(layered_id)):
        end_of_current_layer = start_of_current_layer + len(layered_id[c])
        aus = layer_inference(layered_id[c], prob, prob_id, left_nodes, right_nodes, values)
        values = values.at[start_of_current_layer:end_of_current_layer].set(aus)
        start_of_current_layer = end_of_current_layer

    # Compute category means (MNIST: 10 classes)
    output_values = values[-OUTPUT_NODES:]
    output_values = jnp.reshape(output_values, (10, -1))
    return jnp.mean(output_values, axis=-1)

@jax.jit
def loss_function(layered_id, prob, prob_id, left_nodes, right_nodes, values, answer):
    result = inference(layered_id, prob, prob_id, left_nodes, right_nodes, values)
    onehotresult = jax.nn.one_hot(jnp.argmax(result), 10)
    return optax.softmax_cross_entropy(result, answer.astype(jnp.float32)), jnp.all(onehotresult == answer).astype(jnp.float32)

@jax.jit 
def scalar_loss(layered_id, prob, prob_id, left_nodes, right_nodes, values, answers, temperature):
    prob = batch_fitting_function(prob) * temperature
    prob = jax.nn.softmax(prob, -1)

    batch_loss = jax.vmap(loss_function, in_axes=(None, None, None, None, None, 0, 0)) 
    loss, accuracy = batch_loss(layered_id, prob, prob_id, left_nodes, right_nodes, values, answers)
    if (DEBUG):
        jax.debug.print("{}", jnp.mean(accuracy))
    return jnp.mean(loss)

@jax.jit
def accuracy_function(layered_id, prob, prob_id, left_nodes, right_nodes, values, correct_answer):
    '''Run forward pass and return accuracy between outputs and correct_answer.'''
    predicted = jnp.argmax(jnp.array(inference(layered_id, prob, prob_id, left_nodes, right_nodes, values)))
    correct = jnp.argmax(correct_answer).astype(jnp.float32)
    return (predicted == correct).astype(jnp.float32)

def input_network():
    random.seed(0)

    # Input the connections and output nodes
    with open(NETWORK_ARCHITECTURE_FILE, "r") as network:
        #Initialize null connections of input nodes
        LEFT = [-1 for x in range(0, INPUT_NODES+1)] # left connection at each node
        RIGHT = [-1 for x in range(0, INPUT_NODES+1)] # right connection at each node
        PROB_ID = [-1 for x in range(0, INPUT_NODES+1)] # probability index associated to each node

        for _ in range (INPUT_NODES+1, NETWORK_SIZE+1):
            left, right, prob_id = map(int, network.readline().strip().split())
            LEFT.append(left)
            RIGHT.append(right)
            PROB_ID.append(prob_id)
    
    PROB_ID = jnp.array(PROB_ID)
    LEFT = jnp.array(LEFT)
    RIGHT = jnp.array(RIGHT)

    # Calculate the layers
    l = [0 for _ in range(0, NETWORK_SIZE+1)] # for each node save its layer
    layered_id = [[i for i in range (1, INPUT_NODES+1)]] # for each layer save node ids
    for id in range(INPUT_NODES+1, NETWORK_SIZE+1):
        cl = max(l[LEFT[id]], l[RIGHT[id]]) + 1
        while cl >= len(layered_id): 
            layered_id.append([])
        layered_id[cl].append(id)
        l[id] = cl

    for i in range (len(layered_id)):
        layered_id[i] = jnp.array(layered_id[i]) 
    
    return layered_id, LEFT, RIGHT, PROB_ID

def read_values(file):
    #Autosaving
    if os.path.exists(file + ".values.npy"):
        return jnp.load(file + ".values.npy"), jnp.load(file + ".answers.npy")

    values = []
    answers = []
    
    with open(file, 'r') as image:
        while True:
            line = image.readline()
            if not line:

                break

            # read training inputanswer
            line = list(map(float, line.strip()))

            # one row: pad with zero in col 0, then inputs
            row = line
            values.append(row)

            # Setting the correct answer
            ans = int(image.readline().strip())
            one_hot = [0] * 10
            one_hot[ans] = 1
            answers.append(one_hot)
    
    values = jnp.array(values)
    answers = jnp.array(answers)
    jnp.save(file + ".values.npy", values)
    jnp.save(file + ".answers.npy", answers)

    return values, answers

def test_network(layered_id, prob, prob_id, left_nodes, right_nodes, values, answers):
    batch_accuracy_testing = jax.vmap(accuracy_function, in_axes = (None, None, None, None, None, 0, 0))

    # Testing with binarization
    batch_binarizing = jax.jit(jax.vmap(binarizing_function, in_axes=(0)))
    prob_testing  = batch_binarizing(prob) #duplicate probabilities for testing
    acc = batch_accuracy_testing(layered_id, prob_testing, prob_id, left_nodes, right_nodes, values, answers)

    # Testing without binarization
    prob_testing = jax.nn.softmax(prob, -1)
    acc_nobin = batch_accuracy_testing(layered_id, prob_testing, prob_id, left_nodes, right_nodes, values, answers)

    print("Accuracy: " + str(float(jnp.mean(acc))*100) + "% (no bin: " + str(float(jnp.mean(acc_nobin))*100) + "%)")

def train_network(layered_id, prob, prob_id, left_nodes, right_nodes):
    values_list_testing, answers_list_testing = read_values(TESTDATA_DATA_FILE)
    
    STEPS_PER_EPOCH  = TOTAL_SIZE // BATCH_SIZE
    TOTAL_STEPS = STEPS_PER_EPOCH * EPOCH_COUNT

    values_list, answers_list = read_values(TRAINING_DATA_FILE)
    optimizer = optax.adamw(optax.exponential_decay(LEARNING_RATE,
                                transition_steps = TOTAL_STEPS,
                                decay_rate = 0.1),
                weight_decay=WEIGHT_DECAY)
    opt_state = optimizer.init(prob)

    value_and_grad = jax.jit(jax.value_and_grad(scalar_loss,argnums=1))
    assert TOTAL_SIZE % BATCH_SIZE == 0

    #Training routine
    for epoch in range (0, EPOCH_COUNT):
        values_list = jax.random.permutation(jax.random.PRNGKey(epoch), values_list)
        answers_list = jax.random.permutation(jax.random.PRNGKey(epoch), answers_list)

        values = jnp.reshape(values_list, (STEPS_PER_EPOCH, BATCH_SIZE, -1))
        answers = jnp.reshape(answers_list, (STEPS_PER_EPOCH, BATCH_SIZE, -1))

        loss_sum = 0
        for i in range (STEPS_PER_EPOCH):
            temperature = MAX_TEMPERATURE**((epoch * STEPS_PER_EPOCH + i)/TOTAL_STEPS)
            (loss, gradients) = value_and_grad(layered_id, prob, prob_id, left_nodes, right_nodes, values[i], answers[i], temperature)
            loss_sum += loss

            # Update parameters
            updates, opt_state = optimizer.update(gradients, opt_state, params=prob)
            prob = optax.apply_updates(prob, updates)
        
        if DEBUG:
            print("Gradient:", jnp.linalg.norm(gradients))

        print("Epoch " + str(epoch+1) + " Loss: " + str(loss_sum * BATCH_SIZE / TOTAL_SIZE))

        if (DEBUG): 
            test_network(layered_id, prob, prob_id, left_nodes, right_nodes, values_list_testing, answers_list_testing)
    return prob

def print_network(prob, prob_id, left_nodes, right_nodes):
    # Print the network
    with open(TRAINED_NETWORK_16GATES_FILE, "wb") as f:
        # Write the trained network
        for i in range (INPUT_NODES+1, NETWORK_SIZE+1):
            gate_index = int(jnp.argmax(prob[prob_id[i]]))
            f.write(gate_index.to_bytes(4, byteorder='little', signed=True))
            f.write(i.to_bytes(4, byteorder='little', signed=True))
            f.write(int(left_nodes[i]).to_bytes(4, byteorder='little', signed=True))
            f.write(int(right_nodes[i]).to_bytes(4, byteorder='little', signed=True))
    

def run_training_sequence():
    # Setup network architecture

    left_nodes = []
    right_nodes = []

    #Input the network architecture 
    layered_id, left_nodes, right_nodes, prob_id =  input_network()
    prob = jnp.array([initialize_probabilities(x) for x in range (NETWORK_SIZE+1)])

    #Train network
    prob = train_network(layered_id, prob, prob_id, left_nodes, right_nodes)

    # Print the network to binary file
    print_network(prob, prob_id, left_nodes, right_nodes)