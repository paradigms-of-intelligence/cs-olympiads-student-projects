# How to install
The installation process is currently only supported on Linux.
Requirements: CMake, Ninja

Installation steps
- Clone or download the codebase
- Run the ```build.sh``` script
The script will compile the necessary libjxl libraries and create a virtual environment with the necessary python modules

# How to run
The ```main.py``` file acts as an API for the two steps of compressing the image: training and JXL encoding.
To successfully run the script, use the command ```python main.py [arguments]```.
The following arguments are supported:
- `<filename>`: necessary argument that specifies the input file. If the argument is written as `<folder_path>/` the script will try to compress all the files in the folder. When an error occurs during compression the scripts writes the error and continues to the next file. Supported types are [all Pillow supported file types](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html).
- `<number_of_layers>`: must go after the input parameter. Represents how many layers will be encoded in the JXL file. Valid range 1 to 4.
- `-o <filename>`: specifies an output path for the JXL file. If the input is a folder, output must also be a folder (`<folder_path>/`). If the input is a single file, output can be a filename or a folder, in the second case the final path will be `<folder_path>/<filename>.jxl`. Defaults to `out/`
- `-l <lambda>` specifies the float value of the lambda parameter, which represents the weight of compression loss compared to wasserstein/l2 loss. Defaults to 1
- `-g <gamma>` specifies the float value of the gamma parameter, which represents the weight of coefficient loss compared to context loss. Defaults to 63
- `-s <sigma>` specifies the float value of the log2_sigma parameter used in calculating wasserstein distortion. Defaults to 0
- `-l2 <turns>` specifies how many rounds of l2 training will be applied. Defaults to 90000
- `-ws <turns>` specifies how many rounds of wasserstein distortion training will be applied. Defaults to 10000
- `-i <intermediate>` specifies the folder to store intermediate files in. Defaults to `out/`. It is useless to set this if you don't care about the intermediate files.
- `-space <space>` specifies the training variable space to use for the JXL encoding. Defaults to "rgb", can be also "xyb" or "dct"
- `-c` enables the use of compression settings from the settings.conf file. Each line in the config file should be in the format `<layer>: <value>`, where layer is an integer from 0 to 3 which refers to the $2^{< layer>}$ downsampled layer and `<value>` is an float value 0-25 which specifies the compression strength. 0 is lossless compression. These settings do not affect the training process, but only the encoding process.