"""
Simple Vision Transformer adapter for skin lesion classification.
Uses a smaller, more efficient approach suitable for Apple Silicon.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTForImageClassification, ViTImageProcessor
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image


class SimpleViTDiagnostic(nn.Module):
    """
    Simple ViT-based model for skin lesion classification.
    More efficient for training on Apple Silicon.
    """
    
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        num_classes: int = 7,
        device: str = "mps"
    ):
        super().__init__()
        
        self.model_name = model_name
        self.device = device
        self.num_classes = num_classes
        
        # Load ViT model and processor
        print(f"Loading {model_name}...")
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        
        # Load model with custom number of classes
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # Move to device
        self.model = self.model.to(device)
        
        # Define the skin conditions
        self.skin_conditions = [
            'melanocytic nevi',
            'melanoma',
            'benign keratosis',
            'basal cell carcinoma',
            'actinic keratosis',
            'vascular lesions',
            'dermatofibroma'
        ]
        
        # Add a simple out-of-distribution detector
        self.ood_threshold = 0.5  # Can be tuned based on validation data
        
    def prepare_inputs(
        self,
        images: Union[torch.Tensor, List[Image.Image]]
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for the model.
        """
        # Convert tensors to PIL if needed
        if torch.is_tensor(images):
            pil_images = []
            for img_tensor in images:
                # Denormalize if needed
                if img_tensor.max() <= 1.0:
                    img_tensor = img_tensor * 255
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                pil_images.append(Image.fromarray(img_np))
        else:
            pil_images = images
        
        # Process with ViT processor
        inputs = self.processor(
            images=pil_images,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def forward(
        self,
        images: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training or inference.
        """
        # Prepare inputs
        inputs = self.prepare_inputs(images)
        
        # Forward pass
        outputs = self.model(**inputs, labels=labels)
        
        # Get logits and probabilities
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        
        # Calculate confidence (max probability)
        confidence, predictions = torch.max(probs, dim=-1)
        
        result = {
            "logits": logits,
            "predictions": predictions,
            "confidence": confidence,
            "probabilities": probs
        }
        
        if labels is not None:
            result["loss"] = outputs.loss
            
        return result
    
    def diagnose(
        self,
        image: Union[torch.Tensor, Image.Image],
        return_probabilities: bool = False
    ) -> Dict[str, Union[str, float, List[float]]]:
        """
        Diagnose a single image.
        """
        # Ensure image is in correct format
        if torch.is_tensor(image) and image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(image)
        
        # Get prediction and confidence
        prediction = outputs["predictions"].item()
        confidence = outputs["confidence"].item()
        
        # Check if it's out-of-distribution
        if confidence < self.ood_threshold:
            diagnosis = "Uncertain - image may not be a dermatoscopic image or contains an unknown condition"
            requires_review = True
        else:
            diagnosis = self.skin_conditions[prediction]
            requires_review = confidence < 0.8
        
        result = {
            "diagnosis": diagnosis,
            "confidence": float(confidence),
            "requires_professional_review": requires_review
        }
        
        if return_probabilities:
            probs = outputs["probabilities"].squeeze().cpu().numpy()
            result["probabilities"] = {
                condition: float(prob) 
                for condition, prob in zip(self.skin_conditions, probs)
            }
        
        return result
    
    def save_pretrained(self, save_path: str):
        """Save the fine-tuned model."""
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
        
        # Save configuration
        import json
        config = {
            'model_name': self.model_name,
            'skin_conditions': self.skin_conditions,
            'num_classes': self.num_classes,
            'ood_threshold': self.ood_threshold,
            'device': self.device
        }
        with open(f"{save_path}/diagnostic_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Model saved to {save_path}")
    
    @classmethod
    def from_pretrained(cls, load_path: str, device: str = "mps"):
        """Load a fine-tuned model."""
        # Load configuration
        import json
        with open(f"{load_path}/diagnostic_config.json", 'r') as f:
            config = json.load(f)
        
        # Initialize model
        model = cls(
            model_name=config['model_name'],
            num_classes=config['num_classes'],
            device=device
        )
        
        # Load weights
        model.model = ViTForImageClassification.from_pretrained(
            load_path,
            num_labels=config['num_classes']
        )
        model.model = model.model.to(device)
        
        # Set OOD threshold
        model.ood_threshold = config['ood_threshold']
        
        return model
