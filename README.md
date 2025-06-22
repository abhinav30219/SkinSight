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

### Running Inference

```bash
# Launch the web interface
cd inference
python app_simple.py --model-path ../training/best_model
```

This will start a web server at http://localhost:7860 where you can upload dermatoscopic images for diagnosis.

### Training a Model

```bash
# Train a specific model
cd training
python train_simple.py --config configs/vit_base.yaml

# Compare multiple models
python train_all_models.py --configs_dir configs
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

## Model Comparison

![Accuracy Comparison](https://raw.githubusercontent.com/abhinav30219/SkinSight/main/comparison_results/accuracy_comparison.png)

## Citation

If you use this code, please cite:

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
```

## License

This project is for educational and research purposes only. The code is released under the MIT license.

## Disclaimer

**⚠️ IMPORTANT**: This is an AI-powered diagnostic tool intended for educational and research purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare professional for proper diagnosis and treatment of skin conditions.
