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
jax.config.update('jax_compilation_cache_dir', '/tmp/jax_cache')

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
    parser.add_argument('layers', type=int, nargs='?', default=2,
                       help='Number of layers (default: 2)')
    
    # Optional arguments
    parser.add_argument('-l', '--lambda', dest='lambd', type=float, default=0,
                       help='Lambda value (default: 0)')
    parser.add_argument('-g', '--gamma', type=float, default=0,
                       help='Gamma value (default: 0)')
    parser.add_argument('-s', '--sigma', dest='log2_sigma_value', type=float, default=0,
                       help='Log2 sigma value (default: 0)')
    parser.add_argument('-l2', '--l2-turns', dest='l2_turns', type=int, default=90000,
                       help='L2 turns (default: 90000)')
    parser.add_argument('-ws', '--ws-turns', dest='ws_turns', type=int, default=10000,
                       help='WS turns (default: 10000)')
    parser.add_argument('-o', '--output', dest='output_path', default="",
                       help='Output path')
    parser.add_argument('-space', '--color-space', dest='color', default="rgb",
                       help='Color space (default: rgb)')
    parser.add_argument('-prefix', '--prefix', default="out/",
                       help='Prefix for output files (default: out/)')
    parser.add_argument('-c', '--use-settings', dest='take_settings', action='store_true',
                       help='Use settings from settings.conf')
    
    return parser.parse_args()

# Parse arguments
args = parse_arguments()

# Extract values from args
filename = ""
foldername = ""
lambd = args.lambd
gamma = args.gamma
log2_sigma_value = args.log2_sigma_value
layers = args.layers
l2_turns = args.l2_turns
ws_turns = args.ws_turns
patience = l2_turns
delta = 1e-6
take_settings = args.take_settings
color = args.color.lower()
prefix = args.prefix
output_path = args.output_path

# Determine if input is file or folder
if args.input_path.endswith("/"):
    foldername = args.input_path
else:
    filename = args.input_path

print(f"Parsed arguments: {vars(args)}")

# settings
settings = {}
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
    if output_path and not output_path.endswith("/"):
        output_path = "out/"
elif filename == "":
    # Handle case where no input was provided - should not happen with argparse
    print("Error: No input file or directory specified")
    sys.exit(1)
else:
    filelist = [filename]
    if output_path == "":
        output_path = "out/" + filename.split(".")[0] + ".jxl"



print(filelist)
for f in filelist:
    print(bcolors.OKGREEN + f + bcolors.ENDC)
    target = []
    try:
        with Image.open(f, 'r') as file:
            target = jnp.asarray(file, dtype=jnp.float32) / 255.
            target = jnp.array(target[:(target.shape[0]//(2**(2+layers)))*(2**(2+layers)), :(target.shape[1]//(2**(2+layers)))*(2**(2+layers)), :])
            ff = f.replace('/', '_').split('.')[0]
            try:
                target_features = fastW.get_features(target)
                tr.create_image_split(
                    target,
                    target_features,
                    lambd,
                    gamma,
                    log2_sigma_value,
                    l2_turns,
                    ws_turns,
                    dct_mult,
                    xyb_mult,
                    rgb_mult,
                    layers,
                    prefix + ff,
                    color
                )
                command = [f'./build/jxl_layered_encoder']
                command.append(f'{layers}')
                for i in range(layers):
                    command.append(f"{prefix}{ff}_{i}.txt")
    
                if(foldername != ""):
                    command.append(output_path + f"{ff}.jxl")
                else:
                    command.append(output_path)
                if take_settings:
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