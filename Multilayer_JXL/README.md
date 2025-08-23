# How to install
The installation process is currently only supported on Linux.
Requirements: CMake, Ninja

Installation steps
- Clone or download the codebase
- Run the ```build.sh``` script
The script will compile the necessary libjxl libraries and create a virtual enviroment with the necessary pyhton modules

# How to run
The ```main.py``` file acts as an API for the two steps of compressing the image: training and JXL encoding.
To successfully run the script, use the command ```python main.py [arguments]```.
The following arguments are supported:
- `<filename>`: necessary argument that specifies the input file. if the argument is written as `<folder_path>/` the script will try to compress all the files in the folder. When an error occurs during compression the scripts writes the error and continues to the next file. Supported types are [all Pyllow supported file types](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html).
- `<layer>`: must go after (not necessarily directly after) the input parameter. is an integer 0-4 that describes how many layers will be ancoded in the JXL file.
- `-o <filename>`: specifies an output path for the JXL file. If the input is a folder, output must also be a folder (`<folder_path>/`) or it will be overridden with `out/`. If the input is a single file, output can be a filename or a folder, in the second case the final path will be `<folder_path>/<filename>.jxl`.
- `-l <lambda>` specifies the float value of the lambda parameter. Defaults to 0
- `-g <gamma>` specifies the float value of the gamma parameter. Defaults to 0
- `-s <sigma>` specifies the float value of the sigma parameter. Defaults to 0
- `-l2 <turns>` specifies how many rounds of l2 training will be applied. Defaults to 90000
- `-ws <turns>` specifies how many rounds of wasserstein distortion training will be applied. Defaults to 10000
- `-p <patience>` specifies the integer value of the patience parameter. Defaults to 100000
- `-d <delta>` specifies the float value of the delta parameter. Defaults to 1e-6
- `-prefix <prefix>` specifies the folder to use for the output files. Defaults to `out/`. It is useless to set this if you don't care about the intermediate files.
- `-space <space>` specifies the training space to use for the JXL encoding. Defaults to "rgb", can be also "xyb" or "dct"
- `-c` enables the use of the settings from the settings.conf file. Each line in the config file should be in the format `<layer>: <value>`, where layer is an integer from 0 to 4 which refers to the 2^<layer>x downsampled layer and <value> is the corresponding setting.