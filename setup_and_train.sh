#!/bin/bash

# Setup and training script for skin lesion diagnosis

echo "=== Skin Lesion Diagnosis Training Setup ==="
echo "Starting at $(date)"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Navigate to training directory
cd training

# Start training
echo "Starting training with BLIP-2 VQA..."
cd training
python train_vqa.py --config config.yaml

echo "Training complete at $(date)"
