#!/bin/bash
# Script to run model comparison

# Ensure we're in the correct directory
cd "$(dirname "$0")"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Setting up virtual environment..."
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Navigate to training directory
cd training

# Default models to compare
MODELS=("vit_base" "vit_large" "biomedclip" "swin_base" "convnext")

# Parse arguments
if [ $# -gt 0 ]; then
    MODELS=("$@")
fi

echo "Starting model comparison for: ${MODELS[*]}"

# Run the comparison
python train_all_models.py --configs_dir configs --models "${MODELS[@]}"

echo "Model comparison complete! Results are in the comparison_results directory."
