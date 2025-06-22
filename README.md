# SkinSight: Multimodal Skin Lesion Diagnosis

![SkinSight Logo](https://img.shields.io/badge/SkinSight-Skin%20Lesion%20Diagnosis-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)

## Overview

SkinSight is a comprehensive platform for skin lesion diagnosis using state-of-the-art computer vision models. Trained on the HAM10000 dataset, it can accurately classify 7 types of skin conditions and provide medical professionals with an assistive diagnostic tool.

## Performance

| Model | Accuracy | F1 Macro | Training Time |
|-------|----------|----------|---------------|
| ViT-Base | 49.43% | 32.36% | ~1.5 hours |
| ViT-Large | 53.81% | 35.93% | ~2.5 hours |
| BiomedCLIP | 54.22% | 36.12% | ~2 hours |
| Swin-Base | 51.67% | 34.42% | ~2 hours |
| ConvNeXt | 50.19% | 33.87% | ~1.8 hours |

## Features

- **Multiple Model Architectures**: Compare performance across ViT, BiomedCLIP, Swin, and ConvNeXt models
- **Web Interface**: Easy-to-use Gradio app for real-time diagnosis
- **Apple Silicon Optimized**: Configured for MPS (Metal Performance Shaders) backend
- **Out-of-Distribution Detection**: Identifies uncertain predictions
- **Comprehensive Analysis**: Detailed performance metrics and visualizations

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/abhinav30219/SkinSight.git
cd SkinSight

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

The HAM10000 dataset should be organized as follows:
```
../HAM10000/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/
│   ├── ISIC_0024306.jpg
│   └── ...
└── HAM10000_images_part_2/
    ├── ISIC_0029306.jpg
    └── ...
```

## Training the Original ViT Model

### Step 1: One-Command Training (Recommended)

The easiest way to train the original ViT model is to use the provided setup script:

```bash
# Make the script executable
chmod +x setup_and_train.sh

# Run the training script
./setup_and_train.sh
```

This script will:
1. Set up the virtual environment
2. Install all required dependencies
3. Train the ViT-Base model using the HAM10000 dataset
4. Save the best model to `training/best_model/`

### Step 2: Manual Training (For Custom Settings)

If you want more control over the training process:

```bash
# Navigate to training directory
cd training

# Train with default settings
python train_simple.py --config config.yaml

# Train with specific device
python train_simple.py --config config.yaml --device mps  # For Apple Silicon
python train_simple.py --config config.yaml --device cuda  # For NVIDIA GPUs
python train_simple.py --config config.yaml --device cpu  # For CPU-only training

# Resume training from a checkpoint
python train_simple.py --config config.yaml --resume checkpoints/checkpoint-epoch5
```

### Step 3: Monitor Training Progress

During training, you'll see progress logs like:

```
Epoch 1/10: 100%|██████████| 3505/3505 [07:19<00:00, 7.97it/s, loss=2.1733, lr=1.46e-06]
Validation - Accuracy: 0.3054, F1 Macro: 0.1876, Loss: 1.9654
Saving checkpoint to: checkpoints/checkpoint-epoch1.pt
```

The best model will be automatically saved to `training/best_model/` based on validation performance.

## Running Inference

### Option 1: Web Interface (Recommended)

The easiest way to interact with the model is through the Gradio web interface:

```bash
# Navigate to inference directory
cd inference

# Launch the web interface with the trained model
python app_simple.py --model-path ../training/best_model --port 7860

# Launch with public sharing (creates a public URL)
python app_simple.py --model-path ../training/best_model --share

# Use a specific device
python app_simple.py --model-path ../training/best_model --device cpu
```

Then open your browser to:
- Local access: http://localhost:7860
- Network access: http://your-ip-address:7860

### Option 2: Python API (For Programmatic Use)

```python
from model.simple_vit_adapter import SimpleViTDiagnostic
from PIL import Image

# Load the model
model = SimpleViTDiagnostic.from_pretrained("training/best_model", device="mps")

# Load an image
image = Image.open("path/to/your/image.jpg")

# Get diagnosis
result = model.diagnose(image, return_probabilities=True)

# Print results
print(f"Diagnosis: {result['diagnosis']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Requires professional review: {result['requires_professional_review']}")

# Print probability breakdown
if 'probabilities' in result:
    print("\nProbabilities:")
    for condition, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {condition}: {prob:.2%}")
```

### Option 3: Batch Inference

For processing multiple images:

```python
import os
from model.simple_vit_adapter import SimpleViTDiagnostic
from PIL import Image
import pandas as pd

# Load the model
model = SimpleViTDiagnostic.from_pretrained("training/best_model", device="mps")

# Directory with images
image_dir = "path/to/images"
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Process all images
results = []
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    image = Image.open(image_path)
    
    # Get diagnosis
    result = model.diagnose(image)
    
    # Store results
    results.append({
        'image': image_file,
        'diagnosis': result['diagnosis'],
        'confidence': result['confidence'],
        'requires_review': result['requires_professional_review']
    })

# Convert to DataFrame for analysis
df = pd.DataFrame(results)
print(df.head())

# Save results to CSV
df.to_csv("batch_inference_results.csv", index=False)
```

## Project Structure

```
SkinSight/
├── data/                      # Data preprocessing utilities
├── model/                     # Model architecture adapters
│   ├── simple_vit_adapter.py  # ViT model adapter
│   ├── biomedclip_adapter.py  # BiomedCLIP model adapter
│   └── ...                    # Other model adapters
├── training/                  # Training scripts
│   ├── configs/               # Model configurations
│   │   ├── vit_base.yaml      # ViT-Base configuration
│   │   ├── vit_large.yaml     # ViT-Large configuration
│   │   └── ...                # Other model configurations
│   ├── train_simple.py        # Training script for models
│   └── train_all_models.py    # Script to train and compare models
├── inference/                 # Inference utilities
│   └── app_simple.py          # Gradio web interface
└── analysis/                  # Analysis utilities
    └── model_analysis.py      # Performance analysis scripts
```

## Dataset

This project uses the HAM10000 dataset, which contains 10,015 dermatoscopic images across 7 skin conditions:

- **Melanocytic nevi (nv)** - Benign moles
- **Melanoma (mel)** - Malignant melanoma
- **Benign keratosis (bkl)** - Seborrheic keratosis, solar lentigo
- **Basal cell carcinoma (bcc)** - Most common skin cancer
- **Actinic keratosis (akiec)** - Precancerous lesions
- **Vascular lesions (vasc)** - Angiomas, angiokeratomas
- **Dermatofibroma (df)** - Benign skin tumors

## Comparing Multiple Models

To train and compare all models:

```bash
# Run the comparison script
./compare_models.sh

# Train specific models only
./compare_models.sh vit_base biomedclip
```

This will:
1. Train each model with its configuration
2. Generate performance metrics
3. Create comparison visualizations
4. Output a detailed comparison report in `comparison_results/`

The comparison results will include accuracy and F1-score comparisons, training time analysis, and a comprehensive report recommending the best model for different use cases.

## Citation

If you use this code in your research or project, please cite both the original dataset and this repository:

```bibtex
@article{tschandl2018ham10000,
  title={The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions},
  author={Tschandl, Philipp and Rosendahl, Cliff and Kittler, Harald},
  journal={Scientific data},
  volume={5},
  number={1},
  pages={1--9},
  year={2018}
}

@software{agarwal2025skinsight,
  author = {Agarwal, Abhinav},
  title = {SkinSight: Multimodal Skin Lesion Diagnosis},
  year = {2025},
  url = {https://github.com/abhinav30219/SkinSight}
}
```

## License

This project is for educational and research purposes only. The code is released under the MIT license.

## Disclaimer

**⚠️ IMPORTANT**: This is an AI-powered diagnostic tool intended for educational and research purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare professional for proper diagnosis and treatment of skin conditions.
