"""
Adapter for BiomedCLIP model for skin lesion diagnosis.
Provides functionality to fine-tune and use the BiomedCLIP model.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModel, AutoProcessor
from PIL import Image
import numpy as np
import json
from typing import Dict, Any, List, Tuple, Optional, Union

class BiomedCLIPDiagnostic:
    """BiomedCLIP model for skin lesion diagnosis."""
    
    def __init__(
        self,
        model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        num_classes: int = 7,
        device: str = "mps"
    ):
        """Initialize BiomedCLIP model."""
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device
        self.skin_conditions = [
            "melanocytic nevi",
            "melanoma",
            "benign keratosis",
            "basal cell carcinoma",
            "actinic keratosis",
            "vascular lesions",
            "dermatofibroma"
        ]
        self.ood_threshold = 0.5  # Threshold for out-of-distribution detection
        
        # Load model
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Initialize model
        self.model = AutoModel.from_pretrained(model_name)
        
        # Add classification head
        self.classifier = nn.Linear(self.model.config.projection_dim, num_classes)
        
        # Move to device
        self.model.to(device)
        self.classifier.to(device)
        
        # Set to evaluation mode
        self.model.eval()
        self.classifier.eval()
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for a batch of images."""
        # Process images
        with torch.no_grad():
            # Get image features
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            outputs = self.model.get_image_features(**inputs)
            
            # Get logits
            logits = self.classifier(outputs)
            
            # Get probabilities
            probs = F.softmax(logits, dim=1)
            
            # Get predictions
            preds = torch.argmax(probs, dim=1)
            
            # Get confidence (max probability)
            confidence, _ = torch.max(probs, dim=1)
            
            return {
                "logits": logits,
                "probabilities": probs,
                "predictions": preds,
                "confidence": confidence,
                "features": outputs
            }
    
    @classmethod
    def from_pretrained(cls, model_path: str, device: str = "mps") -> "BiomedCLIPDiagnostic":
        """Load model from a saved checkpoint."""
        # Get model configuration
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
            
            model_name = config.get("model_name", "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
            num_classes = config.get("num_classes", 7)
        else:
            model_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            num_classes = 7
        
        # Create model
        model = cls(model_name=model_name, num_classes=num_classes, device=device)
        
        # Load weights
        model_file = Path(model_path) / "pytorch_model.bin"
        if model_file.exists():
            state_dict = torch.load(model_file, map_location=device)
            
            # Separate model and classifier weights
            model_weights = {k: v for k, v in state_dict.items() if not k.startswith("classifier.")}
            classifier_weights = {k.replace("classifier.", ""): v for k, v in state_dict.items() if k.startswith("classifier.")}
            
            # Load weights
            if model_weights:
                model.model.load_state_dict(model_weights)
            if classifier_weights:
                model.classifier.load_state_dict(classifier_weights)
        
        # Set to evaluation mode
        model.model.eval()
        model.classifier.eval()
        
        return model
    
    def save_pretrained(self, output_dir: str) -> None:
        """Save model to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model configuration
        config = {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "skin_conditions": self.skin_conditions,
            "ood_threshold": self.ood_threshold
        }
        
        with open(output_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Combine model and classifier weights
        state_dict = {}
        for k, v in self.model.state_dict().items():
            state_dict[k] = v
        for k, v in self.classifier.state_dict().items():
            state_dict[f"classifier.{k}"] = v
        
        # Save weights
        torch.save(state_dict, output_path / "pytorch_model.bin")
    
    def __call__(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for a batch of images."""
        return self.forward(images)
    
    def preprocess_image(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """Preprocess an image for the model."""
        # Convert to PIL Image if necessary
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        return inputs["pixel_values"]
    
    def diagnose(
        self, 
        image: Union[Image.Image, np.ndarray],
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """Diagnose a skin lesion image."""
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.forward(image_tensor)
            
            prediction = outputs["predictions"][0].item()
            confidence = outputs["confidence"][0].item()
            probabilities = outputs["probabilities"][0].cpu().numpy()
            
            # Convert probabilities to dictionary
            prob_dict = {
                self.skin_conditions[i]: float(probabilities[i])
                for i in range(len(self.skin_conditions))
            }
            
            # Sort probabilities by value
            sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            
            # Check if confidence is below threshold (potential OOD)
            if confidence < self.ood_threshold:
                diagnosis = "Uncertain (possible out-of-distribution image)"
                requires_review = True
            else:
                diagnosis = self.skin_conditions[prediction]
                requires_review = confidence < 0.7  # Additional threshold for review
            
            # Prepare result
            result = {
                "diagnosis": diagnosis,
                "confidence": confidence,
                "requires_professional_review": requires_review
            }
            
            if return_probabilities:
                result["probabilities"] = {k: float(v) for k, v in prob_dict.items()}
            
            return result

def prepare_model(
    model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    num_classes: int = 7,
    device: str = "mps"
) -> BiomedCLIPDiagnostic:
    """Prepare BiomedCLIP model for fine-tuning."""
    return BiomedCLIPDiagnostic(
        model_name=model_name,
        num_classes=num_classes,
        device=device
    )
