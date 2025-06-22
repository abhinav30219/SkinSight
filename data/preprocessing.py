"""
Data preprocessing module for HAM10000 dataset.
Handles image loading, augmentation, and dataset preparation for Qwen-VL.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import json


class HAM10000Dataset(Dataset):
    """Custom dataset for HAM10000 skin lesion images."""
    
    # Diagnosis mapping
    DX_MAPPING = {
        'nv': 'melanocytic nevi',
        'mel': 'melanoma',
        'bkl': 'benign keratosis',
        'bcc': 'basal cell carcinoma',
        'akiec': 'actinic keratosis',
        'vasc': 'vascular lesions',
        'df': 'dermatofibroma'
    }
    
    # Class to index mapping
    CLASS_TO_IDX = {dx: idx for idx, dx in enumerate(DX_MAPPING.keys())}
    IDX_TO_CLASS = {idx: dx for dx, idx in CLASS_TO_IDX.items()}
    
    def __init__(
        self,
        metadata_path: str,
        image_dirs: List[str],
        transform: Optional[transforms.Compose] = None,
        mode: str = 'train',
        indices: Optional[List[int]] = None
    ):
        """
        Initialize HAM10000 dataset.
        
        Args:
            metadata_path: Path to HAM10000_metadata.csv
            image_dirs: List of directories containing images
            transform: Torchvision transforms to apply
            mode: 'train', 'val', or 'test'
            indices: Specific indices to use (for train/val/test split)
        """
        self.metadata = pd.read_csv(metadata_path)
        self.image_dirs = image_dirs
        self.transform = transform
        self.mode = mode
        
        # Filter by indices if provided
        if indices is not None:
            self.metadata = self.metadata.iloc[indices].reset_index(drop=True)
        
        # Create image path mapping
        self.image_paths = self._create_image_paths()
        
        # Create prompt template
        self.prompt_template = (
            "What type of skin lesion is shown in this dermatoscopic image? "
            "Choose from: melanocytic nevi, melanoma, benign keratosis, "
            "basal cell carcinoma, actinic keratosis, vascular lesions, or dermatofibroma."
        )
    
    def _create_image_paths(self) -> Dict[str, str]:
        """Create mapping of image_id to full path."""
        image_paths = {}
        for image_dir in self.image_dirs:
            for img_file in os.listdir(image_dir):
                if img_file.endswith('.jpg'):
                    img_id = img_file.replace('.jpg', '')
                    image_paths[img_id] = os.path.join(image_dir, img_file)
        return image_paths
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item by index."""
        row = self.metadata.iloc[idx]
        image_id = row['image_id']
        dx = row['dx']
        
        # Load image
        image_path = self.image_paths[image_id]
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.CLASS_TO_IDX[dx]
        
        # Create item
        item = {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'dx': dx,
            'dx_name': self.DX_MAPPING[dx],
            'image_id': image_id,
            'prompt': self.prompt_template
        }
        
        return item


def get_transforms(mode: str = 'train', input_size: int = 448) -> transforms.Compose:
    """
    Get transforms for different modes.
    Qwen-VL uses 448x448 input size.
    """
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP normalization
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
    
    return transform


def get_weighted_sampler(dataset: HAM10000Dataset) -> WeightedRandomSampler:
    """Create weighted sampler to handle class imbalance."""
    # Count samples per class
    class_counts = np.zeros(len(HAM10000Dataset.CLASS_TO_IDX))
    for i in range(len(dataset)):
        label = dataset.metadata.iloc[i]['dx']
        class_idx = HAM10000Dataset.CLASS_TO_IDX[label]
        class_counts[class_idx] += 1
    
    # Calculate weights
    total_samples = len(dataset)
    class_weights = total_samples / (len(class_counts) * class_counts)
    
    # Create sample weights
    sample_weights = []
    for i in range(len(dataset)):
        label = dataset.metadata.iloc[i]['dx']
        class_idx = HAM10000Dataset.CLASS_TO_IDX[label]
        sample_weights.append(class_weights[class_idx])
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )
    
    return sampler


def create_data_splits(
    metadata_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create train/val/test splits ensuring each split has all classes.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    metadata = pd.read_csv(metadata_path)
    
    # Get unique image IDs (to handle duplicate images)
    unique_images = metadata.drop_duplicates(subset=['image_id'])
    
    # Stratified split by diagnosis
    X = unique_images.index
    y = unique_images['dx']
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_ratio,
        stratify=y,
        random_state=random_state
    )
    
    # Second split: train vs val
    val_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        stratify=y_temp,
        random_state=random_state
    )
    
    return list(X_train), list(X_val), list(X_test)


def save_split_info(
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    save_path: str
):
    """Save split information for reproducibility."""
    split_info = {
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
        'train_size': len(train_indices),
        'val_size': len(val_indices),
        'test_size': len(test_indices)
    }
    
    with open(save_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"Split info saved to {save_path}")
    print(f"Train: {len(train_indices)} samples")
    print(f"Val: {len(val_indices)} samples")
    print(f"Test: {len(test_indices)} samples")


def load_split_info(split_path: str) -> Tuple[List[int], List[int], List[int]]:
    """Load previously saved split information."""
    with open(split_path, 'r') as f:
        split_info = json.load(f)
    
    return (
        split_info['train_indices'],
        split_info['val_indices'],
        split_info['test_indices']
    )


def create_dataloaders(
    metadata_path: str,
    image_dirs: List[str],
    batch_size: int = 8,
    num_workers: int = 4,
    use_weighted_sampling: bool = True,
    split_path: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    """
    # Get or create splits
    if split_path and os.path.exists(split_path):
        train_indices, val_indices, test_indices = load_split_info(split_path)
    else:
        train_indices, val_indices, test_indices = create_data_splits(metadata_path)
        if split_path:
            save_split_info(train_indices, val_indices, test_indices, split_path)
    
    # Create datasets
    train_dataset = HAM10000Dataset(
        metadata_path,
        image_dirs,
        transform=get_transforms('train'),
        mode='train',
        indices=train_indices
    )
    
    val_dataset = HAM10000Dataset(
        metadata_path,
        image_dirs,
        transform=get_transforms('val'),
        mode='val',
        indices=val_indices
    )
    
    test_dataset = HAM10000Dataset(
        metadata_path,
        image_dirs,
        transform=get_transforms('test'),
        mode='test',
        indices=test_indices
    )
    
    # Create dataloaders
    if use_weighted_sampling:
        train_sampler = get_weighted_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=False  # Set to False for MPS
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_loader, val_loader, test_loader
