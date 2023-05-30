#!/bin/bash

# This script is used to add a new model to the project.
MNIST_ML_CPP_ROOT="$(dirname "$(pwd)")"

# If MNIST_ML_CPP_ROOT does not end in cpp, add it
if [[ ! $MNIST_ML_CPP_ROOT =~ .*/cpp$ ]]; then
    echo "Not in the correct directory: $MNIST_ML_CPP_ROOT"
    MNIST_ML_CPP_ROOT="$MNIST_ML_CPP_ROOT"/cpp
fi

echo "Setting MNIST_ML_CPP_ROOT to $MNIST_ML_CPP_ROOT"

# Check if the model name is provided.
if [ $# -eq 0 ]; then
    echo "Please provide the model name."
    exit 1
fi

# Check if the model name is valid.
if [[ ! $1 =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "Invalid model name. Only letters, numbers, underscore and dashes are allowed."
    exit 1
fi

# Check if the model name is duplicated.
if [ -d "$MNIST_ML_CPP_ROOT"/"$1" ]; then
    echo "Model $1 already exists."
    exit 1
fi

mkdir -p "$MNIST_ML_CPP_ROOT"/"$1"/src "$MNIST_ML_CPP_ROOT"/"$1"/include

filename_prefix=$(echo "$1" | tr - _) # Replace dashes with underscores
touch "$MNIST_ML_CPP_ROOT"/"$1"/include/"$filename_prefix".hpp 
touch "$MNIST_ML_CPP_ROOT"/"$1"/src/"$filename_prefix".cc 
touch "$MNIST_ML_CPP_ROOT"/"$1"/Makefile