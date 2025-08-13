import os
import coloredlogs, logging, colorama

logger = logging.getLogger(__name__)

def cleanup():
    pass

def main():

    coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s.%(msecs)03d %(hostname)s %(name)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
    )

    
    # Generator
    logger.debug("Generating a network architecture")
    os.system("python3 ../generator/perm_gen.py > ./network_architecture.txt")
    # log.info("Generated\n")

    # Network
    # log.info("Setup the network gates")
    # os.system("python3 main.py < ./network_architecture.txt") #prints a trained_network.bin
    # log.info("16-gate network set up")

    # Converter
    # log.info("Converting network to and-not")
    # os.system("g++ ../converter/convert_network.cpp ../circuit.cpp ../circuit.h -Wall -Wextra -std=gnu++17 -static -o convert_network")
    # os.system("./convert_network trained_network.bin 2gate_trained_network.bin")  
    # log.info("Finished converting")   
     
    # Evaluator

    # os.system("g++ ../evaluator/evaluator.cpp ../circuit.cpp ../circuit.h -Wall -Wextra -std=gnu++17 -static evaluate_network")
    # os.system("./evaluate_network 2gate_trained_network.bin")
 
    # Flush all logs before exiting
    logging.shutdown()


if __name__ == "__main__":
    main()