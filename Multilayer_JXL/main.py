# Implements the file to call that takes in the arguments, calls create_image and calls the cpp to save it as jxl

import os
import sys
import argparse
import train as tr
import jax.numpy as jnp
from PIL import Image
import codex as cdx
import faster_wasserstein_vgg16 as fastW
import subprocess

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
                       help='Path to input file or directory (ending with / for directory). If directory, all image files in the directory will be processed')
    parser.add_argument('number_of_layers', type=int, nargs='?', default=2,
                       help='Number of layers to be encoded (default: 2). At the moment only 2 is supported')
    
    # Optional arguments
    parser.add_argument('-l', '--lambda', dest='lambd', type=float, default=1.,
                       help='Lambda value: weight of compression loss compared to wasserstein/l2 loss (default: 1)')
    parser.add_argument('-g', '--gamma', type=float, default=63.,
                       help='Gamma value: the weight of coefficient loss compared to context loss (default: 63)')
    parser.add_argument('-s', '--sigma', dest='log2_sigma_value', type=float, default=0.,
                       help='Log2 sigma value: the base 2 log of the size of the kernel used in calculating wasserstein distortion (default: 0)')
    parser.add_argument('-l2', '--l2-turns', dest='l2_turns', type=int, default=90000,
                       help='How many rounds of L2 optimization to be done (default: 90000)')
    parser.add_argument('-ws', '--ws-turns', dest='ws_turns', type=int, default=10000,
                       help='How many rounds of WS optimization to be done (default: 10000)')
    parser.add_argument('-o', '--output', dest='output_path', default="",
                       help='Output path: Can be a file or directory. If input is a directory, output must be a directory (default: out/)')
    parser.add_argument('-space', '--variable-space', dest='var_space', default="rgb",
                       help='Specifies the training variable space to use during training. Can be "rgb", "xyz", or "dct" (default: rgb)')
    parser.add_argument('-i', '--intermediate', default="out/",
                       help='Folder to store intermediate files in (default: out/)')
    parser.add_argument('-c', '--use-settings', dest='take_settings', action='store_true',
                       help='Use settings from settings.conf to apply different compression settings to each layer')
    
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
                split_f = f.replace('/', '_').split('.')
                ff = split_f[-2].strip('_') if len(split_f) > 1 and split_f[-2] != "" else split_f[-1].strip('_')
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