"""
Analysis script to evaluate the trained model's performance in detail.
Generates confusion matrix, per-class metrics, and identifies strengths/weaknesses.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.simple_vit_adapter import SimpleViTDiagnostic
from data.preprocessing import create_dataloaders


class ModelAnalyzer:
    """Analyze the trained model's performance in detail."""
    
    def __init__(self, model_path: str, config_path: str, device: str = "mps"):
        self.model_path = model_path
        self.config_path = config_path
        self.device = device
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = SimpleViTDiagnostic.from_pretrained(model_path, device)
        self.model.eval()
        
        # Class names
        self.class_names = self.model.skin_conditions
        self.class_abbr = {
            'melanocytic nevi': 'nv',
            'melanoma': 'mel',
            'benign keratosis': 'bkl',
            'basal cell carcinoma': 'bcc',
            'actinic keratosis': 'akiec',
            'vascular lesions': 'vasc',
            'dermatofibroma': 'df'
        }
        
        # Create output directory
        self.output_dir = Path("analysis_results")
        self.output_dir.mkdir(exist_ok=True)
    
    def evaluate_on_test_set(self):
        """Evaluate model on the entire test set."""
        print("Loading test data...")
        
        # Create dataloaders
        _, _, test_loader = create_dataloaders(
            metadata_path=self.config['data']['metadata_path'],
            image_dirs=self.config['data']['image_dirs'],
            batch_size=16,  # Larger batch for evaluation
            num_workers=0,
            use_weighted_sampling=False,
            split_path=self.config['data']['split_path']
        )
        
        print(f"Evaluating on {len(test_loader.dataset)} test samples...")
        
        all_predictions = []
        all_labels = []
        all_probs = []
        all_confidences = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                images = batch['image'].to(self.device)
                labels = batch['label']
                
                # Get model outputs
                outputs = self.model(images)
                
                # Store results
                all_predictions.extend(outputs['predictions'].cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.append(outputs['probabilities'].cpu().numpy())
                all_confidences.extend(outputs['confidence'].cpu().numpy())
        
        # Concatenate probabilities
        all_probs = np.vstack(all_probs)
        
        return {
            'predictions': np.array(all_predictions),
            'labels': np.array(all_labels),
            'probabilities': all_probs,
            'confidences': np.array(all_confidences)
        }
    
    def create_confusion_matrix(self, labels, predictions):
        """Create and save confusion matrix visualization."""
        cm = confusion_matrix(labels, predictions)
        
        # Calculate normalized version
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=[self.class_abbr[c] for c in self.class_names],
                   yticklabels=[self.class_abbr[c] for c in self.class_names],
                   ax=ax1)
        ax1.set_title('Confusion Matrix (Counts)', fontsize=16)
        ax1.set_xlabel('Predicted', fontsize=12)
        ax1.set_ylabel('True', fontsize=12)
        
        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=[self.class_abbr[c] for c in self.class_names],
                   yticklabels=[self.class_abbr[c] for c in self.class_names],
                   ax=ax2)
        ax2.set_title('Confusion Matrix (Normalized)', fontsize=16)
        ax2.set_xlabel('Predicted', fontsize=12)
        ax2.set_ylabel('True', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return cm, cm_normalized
    
    def analyze_per_class_performance(self, labels, predictions, probabilities):
        """Analyze performance for each class."""
        # Get classification report
        report = classification_report(
            labels, predictions, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Create DataFrame for easier analysis
        metrics_df = pd.DataFrame(report).T
        metrics_df = metrics_df[metrics_df.index.isin(self.class_names)]
        
        # Add sample counts
        unique, counts = np.unique(labels, return_counts=True)
        class_counts = dict(zip(unique, counts))
        metrics_df['n_samples'] = [class_counts.get(i, 0) for i in range(len(self.class_names))]
        
        # Sort by F1-score
        metrics_df = metrics_df.sort_values('f1-score', ascending=False)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Precision
        ax = axes[0, 0]
        metrics_df['precision'].plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('Precision by Class', fontsize=14)
        ax.set_xlabel('Class')
        ax.set_ylabel('Precision')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        # Recall
        ax = axes[0, 1]
        metrics_df['recall'].plot(kind='bar', ax=ax, color='lightcoral')
        ax.set_title('Recall by Class', fontsize=14)
        ax.set_xlabel('Class')
        ax.set_ylabel('Recall')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        # F1-score
        ax = axes[1, 0]
        metrics_df['f1-score'].plot(kind='bar', ax=ax, color='lightgreen')
        ax.set_title('F1-Score by Class', fontsize=14)
        ax.set_xlabel('Class')
        ax.set_ylabel('F1-Score')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        # Sample distribution
        ax = axes[1, 1]
        metrics_df['n_samples'].plot(kind='bar', ax=ax, color='gold')
        ax.set_title('Number of Test Samples by Class', fontsize=14)
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'per_class_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed metrics
        metrics_df.to_csv(self.output_dir / 'per_class_metrics.csv')
        
        return metrics_df
    
    def analyze_confidence_distribution(self, confidences, predictions, labels):
        """Analyze model confidence distribution."""
        # Separate correct and incorrect predictions
        correct_mask = predictions == labels
        correct_confidences = confidences[correct_mask]
        incorrect_confidences = confidences[~correct_mask]
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Overall confidence distribution
        ax = axes[0, 0]
        ax.hist(confidences, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(x=0.5, color='red', linestyle='--', label='OOD threshold')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Count')
        ax.set_title('Overall Confidence Distribution')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Confidence by correctness
        ax = axes[0, 1]
        ax.hist(correct_confidences, bins=30, alpha=0.5, color='green', 
                label=f'Correct (n={len(correct_confidences)})', density=True)
        ax.hist(incorrect_confidences, bins=30, alpha=0.5, color='red', 
                label=f'Incorrect (n={len(incorrect_confidences)})', density=True)
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Density')
        ax.set_title('Confidence Distribution by Correctness')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Accuracy vs confidence threshold
        ax = axes[1, 0]
        thresholds = np.linspace(0, 1, 50)
        accuracies = []
        coverages = []
        
        for thresh in thresholds:
            mask = confidences >= thresh
            if mask.sum() > 0:
                acc = (predictions[mask] == labels[mask]).mean()
                cov = mask.mean()
            else:
                acc = 0
                cov = 0
            accuracies.append(acc)
            coverages.append(cov)
        
        ax.plot(thresholds, accuracies, 'b-', label='Accuracy')
        ax.plot(thresholds, coverages, 'g-', label='Coverage')
        ax.set_xlabel('Confidence Threshold')
        ax.set_ylabel('Metric Value')
        ax.set_title('Accuracy and Coverage vs Confidence Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Per-class average confidence
        ax = axes[1, 1]
        class_confidences = {}
        for i, class_name in enumerate(self.class_names):
            mask = labels == i
            if mask.sum() > 0:
                class_confidences[class_name] = confidences[mask].mean()
        
        sorted_classes = sorted(class_confidences.items(), key=lambda x: x[1], reverse=True)
        classes, conf_values = zip(*sorted_classes)
        
        ax.bar(range(len(classes)), conf_values, color='purple', alpha=0.7)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([self.class_abbr[c] for c in classes], rotation=45)
        ax.set_xlabel('Class')
        ax.set_ylabel('Average Confidence')
        ax.set_title('Average Confidence by Class')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'mean_confidence': float(confidences.mean()),
            'std_confidence': float(confidences.std()),
            'mean_correct_confidence': float(correct_confidences.mean()),
            'mean_incorrect_confidence': float(incorrect_confidences.mean()),
            'low_confidence_ratio': float((confidences < 0.5).mean())
        }
    
    def identify_common_mistakes(self, cm, cm_normalized):
        """Identify the most common misclassifications."""
        mistakes = []
        
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j and cm[i, j] > 0:
                    mistakes.append({
                        'true_class': self.class_names[i],
                        'predicted_class': self.class_names[j],
                        'count': int(cm[i, j]),
                        'percentage': float(cm_normalized[i, j] * 100)
                    })
        
        # Sort by count
        mistakes.sort(key=lambda x: x['count'], reverse=True)
        
        # Save top mistakes
        with open(self.output_dir / 'common_mistakes.json', 'w') as f:
            json.dump(mistakes[:20], f, indent=2)
        
        return mistakes
    
    def generate_summary_report(self, results, metrics_df, confidence_stats, mistakes):
        """Generate a comprehensive summary report."""
        predictions = results['predictions']
        labels = results['labels']
        
        # Overall metrics
        overall_accuracy = (predictions == labels).mean()
        
        report = f"""
# Skin Lesion Diagnosis Model Analysis Report

## Overall Performance
- **Test Accuracy**: {overall_accuracy:.2%}
- **Number of Test Samples**: {len(labels)}
- **Average Confidence**: {confidence_stats['mean_confidence']:.2%}
- **Low Confidence Predictions (<50%)**: {confidence_stats['low_confidence_ratio']:.2%}

## Per-Class Performance Summary

### Best Performing Classes:
"""
        
        # Add top 3 classes
        for idx, row in metrics_df.head(3).iterrows():
            report += f"1. **{idx}**: F1={row['f1-score']:.2f}, Precision={row['precision']:.2f}, Recall={row['recall']:.2f}\n"
        
        report += "\n### Worst Performing Classes:\n"
        
        # Add bottom 3 classes
        for idx, row in metrics_df.tail(3).iterrows():
            report += f"1. **{idx}**: F1={row['f1-score']:.2f}, Precision={row['precision']:.2f}, Recall={row['recall']:.2f}\n"
        
        report += "\n## Common Misclassifications\n"
        
        # Add top 5 mistakes
        for i, mistake in enumerate(mistakes[:5], 1):
            report += f"{i}. **{mistake['true_class']}** → **{mistake['predicted_class']}**: "
            report += f"{mistake['count']} times ({mistake['percentage']:.1f}% of {mistake['true_class']} samples)\n"
        
        report += f"""
## Confidence Analysis
- **Mean Confidence (Correct)**: {confidence_stats['mean_correct_confidence']:.2%}
- **Mean Confidence (Incorrect)**: {confidence_stats['mean_incorrect_confidence']:.2%}
- **Confidence Gap**: {confidence_stats['mean_correct_confidence'] - confidence_stats['mean_incorrect_confidence']:.2%}

## Recommendations
1. The model performs best on: {', '.join(metrics_df.head(3).index.tolist())}
2. The model struggles with: {', '.join(metrics_df.tail(3).index.tolist())}
3. Consider additional training data for classes with low recall
4. The confidence threshold could be adjusted based on the use case
"""
        
        # Save report
        with open(self.output_dir / 'analysis_report.md', 'w') as f:
            f.write(report)
        
        print(report)
        
        return report
    
    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("\n=== Starting Model Analysis ===\n")
        
        # Evaluate on test set
        print("1. Evaluating on test set...")
        results = self.evaluate_on_test_set()
        
        # Create confusion matrix
        print("2. Creating confusion matrix...")
        cm, cm_normalized = self.create_confusion_matrix(
            results['labels'], 
            results['predictions']
        )
        
        # Analyze per-class performance
        print("3. Analyzing per-class performance...")
        metrics_df = self.analyze_per_class_performance(
            results['labels'],
            results['predictions'],
            results['probabilities']
        )
        
        # Analyze confidence distribution
        print("4. Analyzing confidence distribution...")
        confidence_stats = self.analyze_confidence_distribution(
            results['confidences'],
            results['predictions'],
            results['labels']
        )
        
        # Identify common mistakes
        print("5. Identifying common mistakes...")
        mistakes = self.identify_common_mistakes(cm, cm_normalized)
        
        # Generate summary report
        print("6. Generating summary report...")
        self.generate_summary_report(results, metrics_df, confidence_stats, mistakes)
        
        print(f"\n=== Analysis Complete! Results saved to {self.output_dir} ===")


if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Analyze trained model performance')
    parser.add_argument(
        '--model-path',
        type=str,
        default='../training/best_model',
        help='Path to the trained model'
    )
    parser.add_argument(
        '--config-path',
        type=str,
        default='../training/config.yaml',
        help='Path to training configuration'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='mps',
        choices=['mps', 'cuda', 'cpu'],
        help='Device to run analysis on'
    )
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = ModelAnalyzer(args.model_path, args.config_path, args.device)
    analyzer.run_full_analysis()
