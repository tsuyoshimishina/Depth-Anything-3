#!/bin/bash
# Depth-Anything-3 Setup Script
# This script sets up the development environment for Depth-Anything-3.
#
# Prerequisites:
#   - cuda-toolkit-12-4 installed at system level
#   - mamba or conda installed for Python virtual environment management
#
# Usage:
#   ./setup.sh --new-env --basic       # Create new environment and install basic packages
#   ./setup.sh --all                   # Install all optional packages (app + gsplat)
#   ./setup.sh --new-env --all         # Full installation from scratch

set -eo pipefail

# ==============================================================================
# Argument Parsing
# ==============================================================================
TEMP=$(getopt -o h --long help,new-env,basic,app,gsplat,all -n 'setup.sh' -- "$@")

eval set -- "$TEMP"

HELP=false
NEW_ENV=false
BASIC=false
APP=false
GSPLAT=false
ALL=false
ERROR=false

if [ "$#" -eq 1 ]; then
    HELP=true
fi

while true; do
    case "$1" in
        -h|--help) HELP=true; shift ;;
        --new-env) NEW_ENV=true; shift ;;
        --basic) BASIC=true; shift ;;
        --app) APP=true; shift ;;
        --gsplat) GSPLAT=true; shift ;;
        --all) ALL=true; shift ;;
        --) shift; break ;;
        *) ERROR=true; break ;;
    esac
done

if [ "$ERROR" = true ]; then
    echo "Error: Invalid argument"
    HELP=true
fi

if [ "$HELP" = true ]; then
    echo "Depth-Anything-3 Setup Script"
    echo ""
    echo "Usage: ./setup.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help      Display this help message"
    echo "  --new-env       Create a new conda/mamba environment 'depth-anything-3'"
    echo "  --basic         Install basic dependencies (torch, torchvision, xformers, etc.)"
    echo "  --app           Install Gradio UI dependencies"
    echo "  --gsplat        Install gsplat for Gaussian Splatting support"
    echo "  --all           Install all optional dependencies (app + gsplat)"
    echo ""
    echo "Examples:"
    echo "  ./setup.sh --new-env --basic    # Initial setup with basic packages"
    echo "  ./setup.sh --all                # Add all optional features"
    echo "  ./setup.sh --new-env --all      # Full installation from scratch"
    echo ""
    echo "Environment switching:"
    echo "  conda activate depth-anything-3  # Activate Depth-Anything-3"
    echo "  conda activate sam3d-objects     # Switch to SAM 3D"
    echo "  conda activate trellis2          # Switch to TRELLIS.2"
    exit 0
fi

# ==============================================================================
# Configuration
# ==============================================================================
ENV_NAME="depth-anything-3"
PYTHON_VERSION="3.11"
PYTORCH_VERSION="2.6.0"
TORCHVISION_VERSION="0.21.0"
CUDA_VERSION="cu124"
WORKDIR=$(pwd)

# Verify that conda (or mamba, which provides conda) is available
if ! command -v conda > /dev/null 2>&1; then
    echo "Error: conda not found. Please install conda or mamba."
    exit 1
fi

# ==============================================================================
# Helper Functions
# ==============================================================================
print_header() {
    echo ""
    echo "============================================================"
    echo " $1"
    echo "============================================================"
}

check_gpu() {
    if command -v nvidia-smi > /dev/null; then
        echo "CUDA GPU detected"
        return 0
    else
        echo "Error: No CUDA GPU found. This package requires CUDA."
        exit 1
    fi
}

# ==============================================================================
# Environment Setup
# ==============================================================================
if [ "$NEW_ENV" = true ]; then
    print_header "Creating new environment: $ENV_NAME"

    check_gpu

    # Check if environment already exists
    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "Environment '$ENV_NAME' already exists."
        read -p "Do you want to remove and recreate it? [y/N]: " response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            echo "Removing existing environment..."
            conda env remove -n "$ENV_NAME" -y
        else
            echo "Aborting. Please use '--basic' or other options without '--new-env'."
            exit 1
        fi
    fi

    # Create new environment
    echo "Creating conda environment with Python $PYTHON_VERSION..."
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"

    # Initialize conda/mamba for the current shell
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if [ -f "$(conda info --base)/etc/profile.d/mamba.sh" ]; then
        source "$(conda info --base)/etc/profile.d/mamba.sh"
    fi

    # Activate environment
    echo "Activating environment..."
    conda activate "$ENV_NAME"

    # Install PyTorch with CUDA 12.4 support
    print_header "Installing PyTorch $PYTORCH_VERSION with CUDA 12.4"
    pip install torch==$PYTORCH_VERSION torchvision==$TORCHVISION_VERSION --index-url https://download.pytorch.org/whl/$CUDA_VERSION

    echo "Environment '$ENV_NAME' created successfully."
fi

# ==============================================================================
# Basic Dependencies
# ==============================================================================
if [ "$BASIC" = true ]; then
    print_header "Installing basic dependencies"

    # Ensure we're in the right environment
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"

    # Install xformers (required for efficient attention)
    echo "Installing xformers..."
    pip install xformers

    # Install the package in editable mode
    echo "Installing depth-anything-3 package..."
    cd "$WORKDIR"
    pip install -e .

    # Install system dependencies for OpenCV if needed
    if ! dpkg -s libgl1-mesa-glx > /dev/null 2>&1; then
        echo "Installing system dependency: libgl1-mesa-glx..."
        sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx
    fi

    echo "Basic installation complete."
fi

# ==============================================================================
# Optional: Gradio App
# ==============================================================================
if [ "$APP" = true ] || [ "$ALL" = true ]; then
    print_header "Installing Gradio UI dependencies"

    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"

    cd "$WORKDIR"
    pip install -e ".[app]"

    echo "Gradio UI dependencies installed."
fi

# ==============================================================================
# Optional: gsplat (Gaussian Splatting)
# ==============================================================================
if [ "$GSPLAT" = true ] || [ "$ALL" = true ]; then
    print_header "Installing gsplat for Gaussian Splatting"

    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"

    # gsplat requires --no-build-isolation due to CUDA compilation
    echo "Installing gsplat (this may take several minutes)..."
    pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70

    echo "gsplat installed successfully."
fi

# ==============================================================================
# Completion Message
# ==============================================================================
print_header "Setup Complete"

echo ""
echo "Depth-Anything-3 environment is ready!"
echo ""
echo "To activate the environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To run the CLI:"
echo "  da3 --help"
echo ""
echo "To run the Gradio UI (if installed):"
echo "  da3 app"
echo ""
echo "To switch between environments:"
echo "  conda activate depth-anything-3  # Depth-Anything-3"
echo "  conda activate sam3d-objects     # SAM 3D"
echo "  conda activate trellis2          # TRELLIS.2"
echo ""
