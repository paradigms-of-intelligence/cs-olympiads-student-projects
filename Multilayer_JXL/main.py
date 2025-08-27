# Implements the file to call that takes in the arguments, calls create_image and calls the cpp to save it as jxl

import os
# Suppress XLA/CUDA warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow/XLA logs
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
os.environ['JAX_LOG_COMPILES'] = '0'  # Suppress JAX compilation logs

import sys
import argparse
import train as tr
import jax.numpy as jnp
import jax
# Additional JAX logging suppression
jax.config.update('jax_log_compiles', False)

from PIL import Image
import codex as cdx
import faster_wasserstein_vgg16 as fastW
import subprocess
import logging

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# TODO add flags for these
dct_mult = [1024.,256.,128.]
xyb_mult = [1025.,256.,128.]
rgb_mult = 50

def parse_arguments():
    parser = argparse.ArgumentParser(description='Create layered JXL images from input files')
    
    # Positional arguments
    parser.add_argument('input_path', 
                       help='Path to input file or directory (ending with / for directory)')
    parser.add_argument('number_of_layers', type=int, nargs='?', default=2,
                       help='Number of layers (default: 2)')
    
    # Optional arguments
    parser.add_argument('-l', '--lambda', dest='lambd', type=float, default=1.,
                       help='Lambda value (default: 1)')
    parser.add_argument('-g', '--gamma', type=float, default=63.,
                       help='Gamma value (default: 63)')
    parser.add_argument('-s', '--sigma', dest='log2_sigma_value', type=float, default=0.,
                       help='Log2 sigma value (default: 0)')
    parser.add_argument('-l2', '--l2-turns', dest='l2_turns', type=int, default=90000,
                       help='L2 turns (default: 90000)')
    parser.add_argument('-ws', '--ws-turns', dest='ws_turns', type=int, default=10000,
                       help='WS turns (default: 10000)')
    parser.add_argument('-o', '--output', dest='output_path', default="",
                       help='Output path')
    parser.add_argument('-space', '--variable-space', dest='var_space', default="rgb",
                       help='Color space (default: rgb)')
    parser.add_argument('-i', '--intermediate', default="out/",
                       help='Folder to store intermediate files in (default: out/)')
    parser.add_argument('-c', '--use-settings', dest='take_settings', action='store_true',
                       help='Use settings from settings.conf')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Determine if input is file or folder
    filename = ""
    foldername = ""
    if args.input_path.endswith("/"):
        foldername = args.input_path
    else:
        filename = args.input_path
    
    # settings
    if args.take_settings:
        settings = {}
        for i in range(args.number_of_layers):
            settings[f"{i}: compression_strength"] = 3.0
        f = open("settings.conf", "r")
        for line in f:
            key, value = line.strip().split(": ")
            settings[key] = value
        f.close()
    
    # load VGG16 model
    cdx.loss.load_vgg16_model()
    
    # Create file list
    filelist = []
    if foldername != "":
        filelist = [(foldername + f) for f in os.listdir(foldername)]
        if args.output_path and not args.output_path.endswith("/"):
            args.output_path = "out/"
    elif filename == "":
        # Handle case where no input was provided - should not happen with argparse
        print("Error: No input file or directory specified")
        sys.exit(1)
    else:
        filelist = [filename]
        if args.output_path == "":
            split_filename = filename.split(".")
            args.output_path = "out/" + (split_filename[-2] if len(split_filename) > 1 and split_filename[-2] != "" else split_filename[-1]) + ".jxl"

    print(filelist)
    for f in filelist:
        print(bcolors.OKGREEN + f + bcolors.ENDC)
        target = []
        try:
            with Image.open(f, 'r') as file:
                target = jnp.asarray(file, dtype=jnp.float32) / 255.
                min_multiple = (2**(2+args.number_of_layers))
                target = jnp.array(target[:(target.shape[0]//min_multiple)*min_multiple, :(target.shape[1]//min_multiple)*min_multiple, :])
                ff = f.replace('/', '_').split('.')[0]
                try:
                    target_features = fastW.get_features(target)
                    tr.create_image_split(
                        target,
                        target_features,
                        args.lambd,
                        args.gamma,
                        args.log2_sigma_value,
                        args.l2_turns,
                        args.ws_turns,
                        dct_mult,
                        xyb_mult,
                        rgb_mult,
                        args.number_of_layers,
                        args.intermediate + ff,
                        args.var_space.lower()
                    )
                    command = [f'./build/jxl_layered_encoder']
                    command.append(f'{args.number_of_layers}')
                    for i in range(args.number_of_layers):
                        command.append(f"{args.intermediate}{ff}_{i}.txt")
        
                    if(foldername != ""):
                        command.append(args.output_path + f"{ff}.jxl")
                    else:
                        command.append(args.output_path)
                    if args.take_settings:
                        for key, value in settings.items():
                            command.append(f"{value}")
                    print("Running command: " + " ".join(command))
                    subprocess.run(command, check=True)
                except Exception as e:
                    print(bcolors.FAIL + f"Error creating image:" + bcolors.ENDC + f"{e}")
        except FileNotFoundError:
            print(bcolors.FAIL + f"File not found:" + bcolors.ENDC + f"{f}")
        except Exception as e:
            print(bcolors.FAIL + f"Error reading file:" + bcolors.ENDC + f"{e}")
            
if __name__ == "__main__":
    main()