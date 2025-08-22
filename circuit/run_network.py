import config
from generator.generator import generate_network
from network.network import run_training_sequence
from converter.converter import convert_network
from evaluator.evaluator import evaluate_network
# this code assumes that the data was already extracted
def main():
    logger = config.logger or config.initialize_config()

    generate_network()

    run_training_sequence()

    convert_network()

    evaluate_network()

if __name__ == "__main__":
    main()