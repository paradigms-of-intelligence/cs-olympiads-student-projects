#!/bin/bash

# Check if ninja is installed
if ! command -v ninja &> /dev/null; then
    echo "Error: ninja is not installed. Please install ninja to continue."
    exit 1
fi

if [ -d "build" ]; then
    cd build
    if [ -f "build.ninja" ]; then
        echo "Build directory already exists and is configured. Initialise again? [y/N]"
        read -r build_again
        if [[ $build_again == [yY] ]]; then
            cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -G Ninja
            cd ..
            ninja -C build/
        else
            cd ..
            ninja -C build/
        fi
    else
        cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -G Ninja
        cd ..
        ninja -C build/
    fi
else
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -G Ninja
    cd ..
    ninja -C build/
fi

echo "libjxl cloned and patched successfully."

if [ ! -d "venv" ]; then
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "Python virtual environment 'venv' created and dependencies installed."
else
    echo "Python virtual environment 'venv' found. Reinitialise? [y/N]"
    read -r reinit
    if [[ $reinit == [yY] ]]; then
        source venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
        echo "Python virtual environment 'venv' reinitialized and dependencies installed."
    else
        echo "Using existing Python virtual environment 'venv'."
        source venv/bin/activate
    fi
fi
mkdir -p out
echo "Virtual environment 'venv' is configured."
echo "Build finished successfully."