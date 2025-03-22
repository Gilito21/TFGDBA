#!/bin/bash
# Script to detect the CUDA compute capability (architecture) of the installed GPU

# Make sure nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi command not found - defaulting to architecture 86" >&2
    echo "86"
    exit 0
fi

# Try to extract the CUDA compute capability from nvidia-smi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)

if [ -z "$GPU_NAME" ]; then
    echo "No GPU detected or nvidia-smi failed - defaulting to architecture 86" >&2
    echo "86"
    exit 0
fi

echo "Detected GPU: $GPU_NAME" >&2

# Match the GPU model with its architecture
if [[ "$GPU_NAME" == *"A100"* ]]; then
    # A100 - Ampere
    echo "80"
elif [[ "$GPU_NAME" == *"A10"* ]] || [[ "$GPU_NAME" == *"A40"* ]] || [[ "$GPU_NAME" == *"A6000"* ]]; then
    # A10, A40, A6000 - Ampere
    echo "86"
elif [[ "$GPU_NAME" == *"H100"* ]]; then
    # H100 - Hopper
    echo "90"
elif [[ "$GPU_NAME" == *"3090"* ]] || [[ "$GPU_NAME" == *"3080"* ]] || [[ "$GPU_NAME" == *"3070"* ]] || 
     [[ "$GPU_NAME" == *"3060"* ]] || [[ "$GPU_NAME" == *"3050"* ]]; then
    # RTX 30 series - Ampere
    echo "86"
elif [[ "$GPU_NAME" == *"4090"* ]] || [[ "$GPU_NAME" == *"4080"* ]] || [[ "$GPU_NAME" == *"4070"* ]] || 
     [[ "$GPU_NAME" == *"4060"* ]]; then
    # RTX 40 series - Ada Lovelace
    echo "89"
elif [[ "$GPU_NAME" == *"2080"* ]] || [[ "$GPU_NAME" == *"2070"* ]] || 
     [[ "$GPU_NAME" == *"2060"* ]] || [[ "$GPU_NAME" == *"1660"* ]]; then
    # RTX 20 series, GTX 16 series - Turing
    echo "75"
elif [[ "$GPU_NAME" == *"1080"* ]] || [[ "$GPU_NAME" == *"1070"* ]] || 
     [[ "$GPU_NAME" == *"1060"* ]] || [[ "$GPU_NAME" == *"1050"* ]]; then
    # GTX 10 series - Pascal
    echo "61"
elif [[ "$GPU_NAME" == *"TITAN V"* ]] || [[ "$GPU_NAME" == *"V100"* ]]; then
    # Titan V, V100 - Volta
    echo "70"
else
    # Default to a widely supported architecture if we can't identify the GPU
    echo "Warning: Could not identify GPU architecture for $GPU_NAME - using default (86)" >&2
    echo "86"
fi
