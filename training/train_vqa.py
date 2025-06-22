"""
Training script for BLIP-2 VQA model on HAM10000 dataset.
Uses Visual Question Answering approach for skin lesion diagnosis.
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
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
# import wandb  # Commented out for debugging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.blip2_vqa_adapter import BLIP2VQADiagnostic
from data.preprocessing import HAM10000Dataset, create_dataloaders


class VQADataset(Dataset):
    """
    Dataset wrapper that converts classification labels to VQA format.
    """
    
    def __init__(self, base_dataset, model, augment_questions=True):
        self.base_dataset = base_dataset
        self.model = model
        self.augment_questions = augment_questions
        self.label_to_condition = {
            0: 'melanocytic nevi',
            1: 'melanoma',
            2: 'benign keratosis',
            3: 'basal cell carcinoma',
            4: 'actinic keratosis',
            5: 'vascular lesions',
            6: 'dermatofibroma'
        }
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get base item
        item = self.base_dataset[idx]
        image = item['image']
        label = item['label'].item()
        
        # Convert label to condition name
        condition = self.label_to_condition[label]
        
        # Create QA pairs
        qa_pairs = self.model.create_training_qa_pairs(
            condition,
            augment=self.augment_questions
        )
        
        # Randomly select one QA pair for this training step
        question, answer = random.choice(qa_pairs)
        
        return image, question, answer, label


class BLIP2VQATrainer:
    """
    Trainer for BLIP-2 VQA model.
    """
    
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.device = self._get_device()
        self.setup_model()
        self.setup_data()
        self.setup_optimization()
        self.setup_logging()
        
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
            # Check for FP16 support
            if self.config['device']['mixed_precision']:
                print("Note: FP16 mixed precision is not supported on MPS. Using default precision.")
                self.config['device']['mixed_precision'] = False
        else:
            device = "cpu"
            self.config['device']['mixed_precision'] = False
        
        print(f"Using device: {device}")
        return device
    
    def setup_model(self):
        """Initialize the BLIP-2 VQA model."""
        print("Setting up BLIP-2 VQA model...")
        
        # Create LoRA config if specified
        lora_config = None
        if self.config['model']['use_lora']:
            from peft import LoraConfig, TaskType
            lora_config = LoraConfig(
                r=self.config['model']['lora_config']['r'],
                lora_alpha=self.config['model']['lora_config']['lora_alpha'],
                lora_dropout=self.config['model']['lora_config']['lora_dropout'],
                target_modules=self.config['model']['lora_config']['target_modules'],
                bias="none",
                task_type=TaskType.QUESTION_ANS,
            )
        
        self.model = BLIP2VQADiagnostic(
            model_name=self.config['model']['name'],
            use_lora=self.config['model']['use_lora'],
            lora_config=lora_config,
            device=self.device
        )
        
        # Enable gradient checkpointing if specified
        if self.config['training']['gradient_checkpointing']:
            self.model.model.gradient_checkpointing_enable()
    
    def setup_data(self):
        """Set up data loaders."""
        print("Setting up data loaders...")
        
        # Create base dataloaders
        base_train_loader, base_val_loader, base_test_loader = create_dataloaders(
            metadata_path=self.config['data']['metadata_path'],
            image_dirs=self.config['data']['image_dirs'],
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            use_weighted_sampling=self.config['data']['use_weighted_sampling'],
            split_path=self.config['data']['split_path']
        )
        
        # Wrap datasets for VQA
        train_dataset = VQADataset(base_train_loader.dataset, self.model, augment_questions=True)
        val_dataset = VQADataset(base_val_loader.dataset, self.model, augment_questions=False)
        test_dataset = VQADataset(base_test_loader.dataset, self.model, augment_questions=False)
        
        # Create new dataloaders with custom collate function
        def vqa_collate_fn(batch):
            images, questions, answers, labels = zip(*batch)
            images = torch.stack(images)
            return images, list(questions), list(answers), torch.tensor(labels)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            collate_fn=vqa_collate_fn,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            collate_fn=vqa_collate_fn,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            collate_fn=vqa_collate_fn,
            pin_memory=True
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
    
    def setup_optimization(self):
        """Set up optimizer and scheduler."""
        # Calculate total training steps
        steps_per_epoch = len(self.train_loader) // self.config['training']['gradient_accumulation_steps']
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
        
        # Gradient scaler for mixed precision
        self.scaler = GradScaler() if self.config['device']['mixed_precision'] else None
    
    def setup_logging(self):
        """Set up logging directories and wandb."""
        # Create directories
        os.makedirs(self.config['output']['model_dir'], exist_ok=True)
        os.makedirs(self.config['output']['log_dir'], exist_ok=True)
        os.makedirs(self.config['output']['best_model_dir'], exist_ok=True)
        
        # Initialize wandb if available
        # try:
        #     wandb.init(
        #         project="skin-lesion-vqa",
        #         config=self.config,
        #         name=f"blip2-vqa-{time.strftime('%Y%m%d-%H%M%S')}"
        #     )
        # except:
        #     print("Wandb not available, continuing without it")
        print("Wandb disabled for debugging")
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch+1}"
        )
        
        for batch_idx, (images, questions, answers, labels) in progress_bar:
            # Move images to device
            images = images.to(self.device)
            
            # Accumulate gradients
            if batch_idx % self.config['training']['gradient_accumulation_steps'] == 0:
                self.optimizer.zero_grad()
            
            # Forward pass
            if self.scaler:
                with autocast():
                    outputs = self.model(images, questions, answers)
                    loss = outputs['loss'] / self.config['training']['gradient_accumulation_steps']
                
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(images, questions, answers)
                loss = outputs['loss'] / self.config['training']['gradient_accumulation_steps']
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['max_grad_norm']
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['max_grad_norm']
                    )
                    self.optimizer.step()
                
                self.scheduler.step()
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config['training']['logging_steps'] == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'lr': f"{lr:.2e}"
                    })
                    
                    # try:
                    #     wandb.log({
                    #         'train/loss': loss.item(),
                    #         'train/lr': lr,
                    #         'train/epoch': epoch + 1,
                    #         'train/step': self.global_step
                    #     })
                    # except:
                    #     pass
                
                # Evaluation
                if self.global_step % self.config['training']['eval_steps'] == 0:
                    val_metrics = self.evaluate(self.val_loader)
                    print(f"\nValidation at step {self.global_step}: {val_metrics}")
                    
                    # Check for improvement
                    if val_metrics['accuracy'] > self.best_val_accuracy:
                        self.best_val_accuracy = val_metrics['accuracy']
                        self.save_model(self.config['output']['best_model_dir'])
                        self.early_stopping_counter = 0
                    else:
                        self.early_stopping_counter += 1
                    
                    # Early stopping
                    if self.early_stopping_counter >= self.config['training']['early_stopping_patience']:
                        print("Early stopping triggered!")
                        return True
                
                # Save checkpoint
                if self.global_step % self.config['training']['save_steps'] == 0:
                    self.save_checkpoint(epoch, batch_idx)
            
            epoch_loss += loss.item()
        
        return False  # No early stopping
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate the model using VQA."""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, questions, answers, labels in tqdm(dataloader, desc="Evaluating"):
                images = images.to(self.device)
                
                # Get predictions using the diagnose method
                batch_predictions = []
                for i in range(len(images)):
                    result = self.model.diagnose(images[i], return_all_stages=True)
                    
                    # Map diagnosis back to label
                    diagnosis = result.get('diagnosis', '').lower()
                    pred_label = 7  # Default to unknown
                    
                    for label_idx, condition in enumerate(self.model.skin_conditions):
                        if condition.lower() in diagnosis:
                            pred_label = label_idx
                            break
                    
                    batch_predictions.append(pred_label)
                
                all_predictions.extend(batch_predictions)
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Filter out unknown predictions for metrics
        valid_mask = all_predictions < 7
        valid_predictions = all_predictions[valid_mask]
        valid_labels = all_labels[valid_mask]
        
        metrics = {
            'accuracy': accuracy_score(valid_labels, valid_predictions),
            'f1_macro': f1_score(valid_labels, valid_predictions, average='macro'),
            'f1_weighted': f1_score(valid_labels, valid_predictions, average='weighted'),
            'coverage': len(valid_predictions) / len(all_predictions)  # How many were classified
        }
        
        self.model.train()
        return metrics
    
    def save_model(self, save_path: str):
        """Save the model."""
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
    
    def save_checkpoint(self, epoch: int, batch_idx: int):
        """Save training checkpoint."""
        checkpoint_path = os.path.join(
            self.config['output']['model_dir'],
            f"checkpoint-epoch{epoch+1}-step{self.global_step}"
        )
        
        self.save_model(checkpoint_path)
        
        # Save training state
        state = {
            'epoch': epoch,
            'batch_idx': batch_idx,
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
            should_stop = self.train_epoch(epoch)
            
            if should_stop:
                break
            
            # Evaluate at end of epoch
            val_metrics = self.evaluate(self.val_loader)
            print(f"\nEpoch {epoch+1} validation metrics: {val_metrics}")
            
            # try:
            #     wandb.log({
            #         'val/accuracy': val_metrics['accuracy'],
            #         'val/f1_macro': val_metrics['f1_macro'],
            #         'val/f1_weighted': val_metrics['f1_weighted'],
            #         'val/coverage': val_metrics['coverage'],
            #         'val/epoch': epoch + 1
            #     })
            # except:
            #     pass
        
        # Final evaluation on test set
        print("\n--- Final Evaluation on Test Set ---")
        test_metrics = self.evaluate(self.test_loader)
        print(f"Test metrics: {test_metrics}")
        
        # Save final model
        self.save_model(os.path.join(self.config['output']['model_dir'], 'final'))
        
        print("\nTraining completed!")


def main():
    parser = argparse.ArgumentParser(description="Train BLIP-2 VQA model for skin lesion diagnosis")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create trainer and start training
    trainer = BLIP2VQATrainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()
