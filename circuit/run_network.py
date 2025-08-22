from config import *
from generator.generator import (
    generate_network_custom_distribution,
    generate_network_probabilistic_model,
)
from network.network import run_training_sequence

from converter.converter import convert_network
from evaluator.evaluator import evaluate_network

from converter.convert_abc import convert_abc_format
from evaluator.evaluator_abc import evaluate_abc_format


def main():
    initialize_config()

    generate_network_probabilistic_model()

    run_training_sequence()

    if ABC_FORMAT:
        convert_network()
        evaluate_network()
    else:
        convert_abc_format()
        evaluate_abc_format()


if __name__ == "__main__":
    main()
