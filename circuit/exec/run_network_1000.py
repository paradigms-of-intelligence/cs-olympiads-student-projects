import os
import coloredlogs, logging

logger = logging.getLogger(__name__)

def cleanup():
    pass

def main():
    coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s.%(msecs)03d %(hostname)s %(name)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
    )

    # Set ENV_VARIABILES
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=12"
    # # Generator
    logger.debug("Generating a network architecture")
    os.system("python3 ../generator/perm_gen_1000.py > ./network_architecture.txt")
    logger.debug("Generated")

    # # Network
    logger.debug("Setup the network gates")
    os.system("python3 ../network/main.py") #prints a trained_network.bin
    logger.debug("16-gate network set up")

    # # Converter
    logger.debug("Converting network to and-not")
    os.system("g++ ../converter/convert_network.cpp ../circuit.h ../circuit.cpp  -Wall -Wextra -std=gnu++17 -static -o convert_network")
    os.system("./convert_network trained_network.bin 2gate_trained_network.bin")  
    logger.debug("Finished converting")   
     
    # Evaluator

    logger.debug("Evaluating the network")
    os.system("g++ ../evaluator/evaluator.cpp ../circuit.cpp ../circuit.h -Wall -Wextra -std=gnu++17 -static -o evaluate_network")
    os.system("./evaluate_network ../data/testdata.txt 2gate_trained_network.bin")
    logger.debug("Evaluated")
 
    # Flush all logs before exiting
    logging.shutdown()


if __name__ == "__main__":
    main()