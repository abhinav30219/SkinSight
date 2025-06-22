"""
Training script for Qwen-VL skin lesion classifier.
Optimized for Apple Silicon (MPS) with gradient accumulation and mixed precision.
"""

import os
import sys
import yaml
import json
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime
from pathlib import Path
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import set_seed

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import create_dataloaders, HAM10000Dataset
from model.qwen_vl_adapter import QwenVLClassifier
from peft import LoraConfig


class Trainer:
    """Trainer class for Qwen-VL skin lesion classifier."""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set seed for reproducibility
        set_seed(self.config['seed'])
        
        # Initialize accelerator for mixed precision training
        # Note: MPS doesn't support fp16 mixed precision in Accelerate
        device_type = self.config['device']['type']
        if device_type == 'mps' and self.config['training']['fp16']:
            print("Note: FP16 mixed precision is not supported on MPS. Using default precision.")
            mixed_precision = 'no'
        else:
            mixed_precision = 'fp16' if self.config['training']['fp16'] else 'no'
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            mixed_precision=mixed_precision,
            device_placement=True
        )
        
        # Setup device
        self.device = torch.device(self.config['device']['type'])
        print(f"Using device: {self.device}")
        
        # Create output directories
        self.setup_directories()
        
        # Initialize metrics storage
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.global_step = 0
        
    def setup_directories(self):
        """Create necessary directories for outputs."""
        self.checkpoint_dir = Path(self.config['output']['model_dir'])
        self.log_dir = Path(self.config['output']['log_dir'])
        self.best_model_dir = Path(self.config['output']['best_model_dir'])
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(self.log_dir)
    
    def setup_model(self):
        """Initialize model with LoRA configuration."""
        print("Setting up Qwen-VL model...")
        
        # Create LoRA config
        lora_config = LoraConfig(
            r=self.config['model']['lora_config']['r'],
            lora_alpha=self.config['model']['lora_config']['lora_alpha'],
            lora_dropout=self.config['model']['lora_config']['lora_dropout'],
            target_modules=self.config['model']['lora_config']['target_modules'],
            task_type="CAUSAL_LM"
        )
        
        # Initialize model
        self.model = QwenVLClassifier(
            model_name=self.config['model']['name'],
            num_classes=self.config['model']['num_classes'],
            use_lora=self.config['model']['use_lora'],
            lora_config=lora_config,
            device=self.device
        )
        
        # Enable gradient checkpointing if specified
        if self.config['training']['gradient_checkpointing']:
            self.model.model.gradient_checkpointing_enable()
        
        return self.model
    
    def setup_data(self):
        """Setup data loaders."""
        print("Setting up data loaders...")
        
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
        
        return self.train_loader, self.val_loader, self.test_loader
    
    def setup_optimization(self):
        """Setup optimizer and scheduler."""
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config['training']['learning_rate'],
            betas=(self.config['optimizer']['adam_beta1'], self.config['optimizer']['adam_beta2']),
            eps=self.config['optimizer']['adam_epsilon'],
            weight_decay=self.config['optimizer']['weight_decay']
        )
        
        # Calculate total training steps
        num_training_steps = (
            len(self.train_loader) // self.config['training']['gradient_accumulation_steps']
        ) * self.config['training']['num_epochs']
        
        # Setup scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config['scheduler']['num_warmup_steps'],
            num_training_steps=num_training_steps
        )
        
        # Prepare with accelerator
        self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader
        )
        
        return self.optimizer, self.scheduler
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Forward pass
            outputs = self.model(
                images=batch['image'],
                prompts=batch['prompt'],
                labels=batch['label']
            )
            
            loss = outputs['loss']
            loss = loss / self.config['training']['gradient_accumulation_steps']
            
            # Backward pass
            self.accelerator.backward(loss)
            
            # Update weights
            if (batch_idx + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                # Gradient clipping
                if self.config['training']['max_grad_norm'] > 0:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['max_grad_norm']
                    )
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config['training']['logging_steps'] == 0:
                    self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                    self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)
                
                # Evaluation
                if self.global_step % self.config['training']['eval_steps'] == 0:
                    val_metrics = self.evaluate(self.val_loader)
                    self.log_metrics(val_metrics, 'val')
                    
                    # Early stopping check
                    if self.check_early_stopping(val_metrics):
                        return True
                
                # Save checkpoint
                if self.global_step % self.config['training']['save_steps'] == 0:
                    self.save_checkpoint(epoch)
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        return False  # Continue training
    
    def evaluate(self, dataloader):
        """Evaluate model on given dataloader."""
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                outputs = self.model(
                    images=batch['image'],
                    prompts=batch['prompt'],
                    labels=batch['label']
                )
                
                loss = outputs['loss']
                logits = outputs['logits']
                
                preds = torch.argmax(logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
                total_loss += loss.item()
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')
        precision_macro, recall_macro, _, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro'
        )
        
        metrics = {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'confusion_matrix': confusion_matrix(all_labels, all_preds)
        }
        
        self.model.train()
        return metrics
    
    def log_metrics(self, metrics: dict, split: str):
        """Log metrics to tensorboard."""
        for key, value in metrics.items():
            if key != 'confusion_matrix':
                self.writer.add_scalar(f'{split}/{key}', value, self.global_step)
        
        # Log confusion matrix
        if 'confusion_matrix' in metrics:
            self.plot_confusion_matrix(metrics['confusion_matrix'], split)
    
    def plot_confusion_matrix(self, cm: np.ndarray, split: str):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(10, 8))
        class_names = list(HAM10000Dataset.DX_MAPPING.keys())
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title(f'Confusion Matrix - {split}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save figure
        save_path = self.log_dir / f'confusion_matrix_{split}_{self.global_step}.png'
        plt.savefig(save_path)
        plt.close()
        
        # Log to tensorboard
        self.writer.add_figure(f'{split}/confusion_matrix', plt.gcf(), self.global_step)
    
    def check_early_stopping(self, val_metrics: dict) -> bool:
        """Check if should stop training early."""
        current_f1 = val_metrics['f1_macro']
        
        if current_f1 > self.best_val_f1:
            self.best_val_f1 = current_f1
            self.patience_counter = 0
            self.save_best_model()
            print(f"New best model! F1-macro: {current_f1:.4f}")
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.config['training']['early_stopping_patience']:
            print(f"Early stopping triggered. Best F1-macro: {self.best_val_f1:.4f}")
            return True
        
        return False
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}_step_{self.global_step}'
        self.accelerator.unwrap_model(self.model).save_adapter(str(checkpoint_path))
        
        # Save training state
        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'best_val_f1': self.best_val_f1,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict()
        }
        torch.save(state, checkpoint_path / 'training_state.pt')
    
    def save_best_model(self):
        """Save best model."""
        self.accelerator.unwrap_model(self.model).save_adapter(str(self.best_model_dir))
        
        # Save metrics
        metrics = {
            'best_val_f1': self.best_val_f1,
            'global_step': self.global_step
        }
        with open(self.best_model_dir / 'best_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def train(self):
        """Main training loop."""
        print("Starting training...")
        
        # Setup components
        self.setup_model()
        self.setup_data()
        self.setup_optimization()
        
        # Training loop
        for epoch in range(self.config['training']['num_epochs']):
            print(f"\n--- Epoch {epoch+1}/{self.config['training']['num_epochs']} ---")
            
            # Train for one epoch
            should_stop = self.train_epoch(epoch)
            
            if should_stop:
                break
            
            # Evaluate on validation set
            val_metrics = self.evaluate(self.val_loader)
            print(f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val Accuracy: {val_metrics['accuracy']:.4f}, "
                  f"Val F1-macro: {val_metrics['f1_macro']:.4f}")
        
        # Final evaluation on test set
        print("\n--- Final Evaluation on Test Set ---")
        test_metrics = self.evaluate(self.test_loader)
        print(f"Test Loss: {test_metrics['loss']:.4f}, "
              f"Test Accuracy: {test_metrics['accuracy']:.4f}, "
              f"Test F1-macro: {test_metrics['f1_macro']:.4f}")
        
        # Save test metrics
        with open(self.best_model_dir / 'test_metrics.json', 'w') as f:
            test_metrics['confusion_matrix'] = test_metrics['confusion_matrix'].tolist()
            json.dump(test_metrics, f, indent=2)
        
        # Close tensorboard writer
        self.writer.close()
        
        print("\nTraining complete!")
        print(f"Best model saved to: {self.best_model_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train Qwen-VL skin lesion classifier')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = Trainer(args.config)
    trainer.train()


if __name__ == '__main__':
    main()
