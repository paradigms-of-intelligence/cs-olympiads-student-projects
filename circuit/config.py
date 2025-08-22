from dataclasses import dataclass
import logging, coloredlogs
import os

ENV = [
    ["XLA_FLAGS", "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=12"],
    ["TF_GPU_ALLOCATOR", "cuda_malloc_async"],
    ["XLA_PYTHON_CLIENT_PREALLOCATE", "false"],
]

# Filepaths (relative to exec folder)
NETWORK_ARCHITECTURE_FILE = "./exec/network_architecture.txt"
TRAINING_DATA_FILE = "./data/training.txt"
TESTDATA_DATA_FILE = "./data/testdata.txt"
TRAINED_NETWORK_16GATES_FILE = "./exec/trained_network_16gates.bin"
TRAINED_NETWORK_2GATES_FILE = "./exec/trained_network_2gates.bin"

ABC_FORMAT = 1
VERBOSE = True

INPUT_NODES = 784
OUTPUT_NODES = 1000
LAYERS = [INPUT_NODES, 3000, 3000, OUTPUT_NODES]
NETWORK_SIZE = sum(LAYERS)

ALWAYS_TRUE = 0x7FFFFFFF
ALWAYS_FALSE = 0x7FFFFFFE

# Hyper-parameters
EPOCH_COUNT = 10
TOTAL_SIZE = 60000
BATCH_SIZE = 500
TEST_SIZE = 10000

LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0005
MAX_TEMPERATURE = 3


log: logging.Logger | None = None


def update_input(new_input):
    INPUT_NODES = new_input
    LAYERS[0] = INPUT_NODES
    NETWORK_SIZE = sum(LAYERS)


def initialize_config():
    global logger
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG if VERBOSE else logging.INFO)

    coloredlogs.install(
        level="DEBUG" if VERBOSE else "INFO",
        fmt="%(asctime)s.%(msecs)03d %(hostname)s %(name)s[%(process)d] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    for env_var in ENV:
        log.debug(f"Setting environment variable {env_var[0]} to {env_var[1]}")
        os.environ[env_var[0]] = env_var[1]
