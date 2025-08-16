import sys
import os
import coloredlogs, logging
import subprocess

logger = logging.getLogger(__name__)

def run_test():
    logger.debug("Generating a network architecture")
    os.system("python3 ../generator/convolution.py > ./network_architecture.txt")
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
    os.system("g++ ../evaluator/evaluator_test.cpp ../circuit.cpp ../circuit.h -Wall -Wextra -std=gnu++17 -static -o evaluate_network")
    
    cmd = ["./evaluate_network", "../data/testdata.txt", "2gate_trained_network.bin"]
    output = subprocess.Popen( cmd, stdout=subprocess.PIPE ).communicate()[0]
    logger.debug("Evaluated")

    print(f"Output: {output}")
    return float(output.strip())

TEST_COUNT = 2
def main():
    label = sys.argv[1]

    coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s.%(msecs)03d %(hostname)s %(name)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
    )

    # Set ENV_VARIABILES
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=12"
    # # Generator
    sum = 0
    for _ in range(0, TEST_COUNT):
        sum += run_test()

    sum /= TEST_COUNT
    
    with open("plot_data.txt", "a") as f:
        f.write(f"{label} {sum}\n")
    


    #Cleaning the directory
    logger.debug("Cleaning the workspace")
    os.system("rm 2gate_trained_network.bin convert_network evaluate_network" 
              + " network_architecture.txt trained_network.bin")
              
    # Flush all logs before exiting
    logging.shutdown()

if __name__ == "__main__":
    main()