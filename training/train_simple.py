"""
Simple training script for ViT-based skin lesion classification.
Optimized for Apple Silicon and smaller memory footprint.
"""

import os
import sys
import argparse
import yaml
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.simple_vit_adapter import SimpleViTDiagnostic
from data.preprocessing import create_dataloaders


class SimpleTrainer:
    """
    Trainer for simple ViT model.
    """
    
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.device = self._get_device()
        self.setup_model()
        self.setup_data()
        self.setup_optimization()
        
        self.global_step = 0
        self.best_val_accuracy = 0.0
        self.early_stopping_counter = 0
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _get_device(self) -> str:
        """Get device based on configuration and availability."""
        device_type = self.config['device']['type']
        
        if device_type == "cuda" and torch.cuda.is_available():
            device = "cuda"
        elif device_type == "mps" and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        
        print(f"Using device: {device}")
        return device
    
    def setup_model(self):
        """Initialize the ViT model."""
        print("Setting up ViT model...")
        
        self.model = SimpleViTDiagnostic(
            model_name="google/vit-base-patch16-224",
            num_classes=7,
            device=self.device
        )
        
        # Enable gradient checkpointing if specified
        if self.config['training'].get('gradient_checkpointing', False):
            self.model.model.gradient_checkpointing_enable()
    
    def setup_data(self):
        """Set up data loaders."""
        print("Setting up data loaders...")
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            metadata_path=self.config['data']['metadata_path'],
            image_dirs=self.config['data']['image_dirs'],
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            use_weighted_sampling=self.config['data']['use_weighted_sampling'],
            split_path=self.config['data']['split_path']
        )
        
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
    
    def setup_optimization(self):
        """Set up optimizer and scheduler."""
        # Calculate total training steps
        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * self.config['training']['num_epochs']
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            betas=(self.config['optimizer']['adam_beta1'], self.config['optimizer']['adam_beta2']),
            eps=self.config['optimizer']['adam_epsilon'],
            weight_decay=self.config['optimizer']['weight_decay']
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=1e-6
        )
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}"
        )
        
        for batch in progress_bar:
            # Get batch data
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images, labels=labels)
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['max_grad_norm']
            )
            
            # Update weights
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            epoch_loss += loss.item()
            all_predictions.extend(outputs['predictions'].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            self.global_step += 1
        
        # Calculate epoch metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        print(f"\nEpoch {epoch+1} - Loss: {epoch_loss/len(self.train_loader):.4f}, "
              f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return epoch_loss / len(self.train_loader)
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_confidences = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images, labels=labels)
                
                total_loss += outputs['loss'].item()
                all_predictions.extend(outputs['predictions'].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(outputs['confidence'].cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1_macro = f1_score(all_labels, all_predictions, average='macro')
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
        avg_confidence = np.mean(all_confidences)
        
        metrics = {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'avg_confidence': avg_confidence
        }
        
        return metrics
    
    def save_model(self, save_path: str):
        """Save the model."""
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
    
    def save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        checkpoint_path = os.path.join(
            self.config['output']['model_dir'],
            f"checkpoint-epoch{epoch+1}"
        )
        
        self.save_model(checkpoint_path)
        
        # Save training state
        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'best_val_accuracy': self.best_val_accuracy,
            'early_stopping_counter': self.early_stopping_counter,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict()
        }
        
        torch.save(state, os.path.join(checkpoint_path, 'training_state.pt'))
    
    def train(self):
        """Main training loop."""
        print("Starting training...")
        
        for epoch in range(self.config['training']['num_epochs']):
            print(f"\n--- Epoch {epoch+1}/{self.config['training']['num_epochs']} ---")
            
            # Train epoch
            train_loss = self.train_epoch(epoch)
            
            # Evaluate on validation set
            val_metrics = self.evaluate(self.val_loader)
            print(f"Validation - Loss: {val_metrics['loss']:.4f}, "
                  f"Accuracy: {val_metrics['accuracy']:.4f}, "
                  f"F1 Macro: {val_metrics['f1_macro']:.4f}, "
                  f"Avg Confidence: {val_metrics['avg_confidence']:.4f}")
            
            # Check for improvement
            if val_metrics['accuracy'] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics['accuracy']
                self.save_model(self.config['output']['best_model_dir'])
                self.early_stopping_counter = 0
                print(f"New best validation accuracy: {self.best_val_accuracy:.4f}")
            else:
                self.early_stopping_counter += 1
            
            # Early stopping
            if self.early_stopping_counter >= self.config['training']['early_stopping_patience']:
                print("Early stopping triggered!")
                break
            
            # Save checkpoint
            if (epoch + 1) % self.config['training']['save_epochs'] == 0:
                self.save_checkpoint(epoch)
        
        # Final evaluation on test set
        print("\n--- Final Evaluation on Test Set ---")
        test_metrics = self.evaluate(self.test_loader)
        print(f"Test - Loss: {test_metrics['loss']:.4f}, "
              f"Accuracy: {test_metrics['accuracy']:.4f}, "
              f"F1 Macro: {test_metrics['f1_macro']:.4f}, "
              f"F1 Weighted: {test_metrics['f1_weighted']:.4f}, "
              f"Avg Confidence: {test_metrics['avg_confidence']:.4f}")
        
        # Save final model
        self.save_model(os.path.join(self.config['output']['model_dir'], 'final'))
        
        # Save test results
        with open(os.path.join(self.config['output']['model_dir'], 'test_results.json'), 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        print("\nTraining completed!")


def main():
    parser = argparse.ArgumentParser(description="Train ViT model for skin lesion classification")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create trainer and start training
    trainer = SimpleTrainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()
