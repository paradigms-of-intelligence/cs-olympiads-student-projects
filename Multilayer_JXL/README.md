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

For a complete list of available arguments and their descriptions, run:
```
python main.py --help
```