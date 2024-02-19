#!/bin/bash
if command -v conda &> /dev/null; then
    echo "Conda is already installed."
    source ~/miniconda3/etc/profile.d/conda.sh 
else
    # Install Miniconda (adjust the URL and file name based on your needs)
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p ~/miniconda3
    rm miniconda.sh
    source ~/miniconda3/etc/profile.d/conda.sh
    echo "Conda has been installed."
fi
# Check if the environment.yml file exists
if [ -f environment.yml ]; then
    # Check if the environment name is provided as an argument
    if [ -z "$1" ]; then
        # Set default environment name to "humanoid"
        ENV_NAME="humanoids"
    else
        ENV_NAME="$1"
    fi

    # Create a Conda environment using the environment.yml file and the provided or default name
    conda env create -f environment.yml --name "$ENV_NAME"

    # Activate the newly created environment
    source ~/miniconda3/bin/activate "$ENV_NAME"

    # Display a message indicating successful installation
    echo "Conda environment '$ENV_NAME' has been created and activated."

    pip3 install -r requirements.txt
    source ~/miniconda3/bin/activate "$ENV_NAME"
    pip3 install -e .

    # Installing jax GPU support
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU is available."
        pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    fi

else
    # Display an error message if environment.yml file is not found
    echo "Error: environment.yml file not found."
    exit 1
fi
