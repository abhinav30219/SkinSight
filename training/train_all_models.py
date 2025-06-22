"""
Script to train multiple models on the HAM10000 dataset using different architectures.
Executes training for each configuration file and tracks metrics for comparison.
"""

import os
import sys
import argparse
import yaml
import json
import torch
import datetime
import subprocess
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_training(config_path, output_dir=None):
    """Run training with the specified config file."""
    cmd = [sys.executable, "train_simple.py", "--config", config_path]
    
    if output_dir:
        cmd.extend(["--output_dir", output_dir])
    
    print(f"\n{'='*80}")
    print(f"Starting training with config: {config_path}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Log the output
    log_path = Path(config_path).stem + "_log.txt"
    with open(log_path, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n\nERRORS:\n")
            f.write(result.stderr)
    
    # Check if training was successful
    if result.returncode != 0:
        print(f"Training failed with config: {config_path}")
        print(f"Check log file for details: {log_path}")
        return False, None
    
    print(f"Training completed for config: {config_path}")
    return True, log_path

def extract_metrics(log_path):
    """Extract metrics from the log file."""
    with open(log_path, "r") as f:
        log_content = f.read()
    
    metrics = {}
    
    # Extract final test metrics
    test_section = log_content.split("--- Final Evaluation on Test Set ---")
    if len(test_section) > 1:
        test_results = test_section[1].strip().split("\n")[1]
        if "Test - " in test_results:
            metrics_text = test_results.split("Test - ")[1]
            metrics_parts = metrics_text.split(", ")
            for part in metrics_parts:
                if ":" in part:
                    key, value = part.split(":")
                    try:
                        metrics[key.strip()] = float(value.strip())
                    except ValueError:
                        metrics[key.strip()] = value.strip()
    
    # Extract training time
    training_time = None
    for line in log_content.splitlines():
        if "Total training time:" in line:
            try:
                training_time = float(line.split(":")[1].strip().split()[0])
                metrics["training_time_minutes"] = training_time
            except (ValueError, IndexError):
                pass
    
    # Extract peak memory usage
    peak_memory = None
    for line in log_content.splitlines():
        if "Peak memory usage:" in line:
            try:
                peak_memory = float(line.split(":")[1].strip().split()[0])
                metrics["peak_memory_gb"] = peak_memory
            except (ValueError, IndexError):
                pass
    
    return metrics

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def compare_models(results):
    """Generate comparison report and visualizations."""
    # Create DataFrame for comparison
    df = pd.DataFrame(results).T
    
    # Sort by accuracy
    df = df.sort_values("Accuracy", ascending=False)
    
    # Save to CSV
    df.to_csv("model_comparison.csv")
    
    # Create visualizations
    create_comparison_plots(df)
    
    # Generate markdown report
    create_markdown_report(df)
    
    return df

def create_comparison_plots(df):
    """Create comparison visualizations."""
    # Create output directory
    os.makedirs("comparison_results", exist_ok=True)
    
    # Accuracy comparison
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=df.index, y="Accuracy", data=df)
    plt.title("Accuracy Comparison", fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    
    # Add values on top of bars
    for i, v in enumerate(df["Accuracy"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    
    plt.tight_layout()
    plt.savefig("comparison_results/accuracy_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # F1 Macro comparison
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=df.index, y="F1 Macro", data=df)
    plt.title("F1 Macro Score Comparison", fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("F1 Macro", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    
    # Add values on top of bars
    for i, v in enumerate(df["F1 Macro"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    
    plt.tight_layout()
    plt.savefig("comparison_results/f1_macro_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Training time comparison
    if "training_time_minutes" in df.columns:
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=df.index, y="training_time_minutes", data=df)
        plt.title("Training Time Comparison", fontsize=16)
        plt.xlabel("Model", fontsize=14)
        plt.ylabel("Training Time (minutes)", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        
        # Add values on top of bars
        for i, v in enumerate(df["training_time_minutes"]):
            ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=10)
        
        plt.tight_layout()
        plt.savefig("comparison_results/training_time_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    # Memory usage comparison
    if "peak_memory_gb" in df.columns:
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=df.index, y="peak_memory_gb", data=df)
        plt.title("Peak Memory Usage Comparison", fontsize=16)
        plt.xlabel("Model", fontsize=14)
        plt.ylabel("Peak Memory (GB)", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        
        # Add values on top of bars
        for i, v in enumerate(df["peak_memory_gb"]):
            ax.text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=10)
        
        plt.tight_layout()
        plt.savefig("comparison_results/memory_usage_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    # Combined metrics (spider plot)
    metrics = ["Accuracy", "F1 Macro", "F1 Weighted", "Precision Macro", "Recall Macro"]
    available_metrics = [m for m in metrics if m in df.columns]
    
    if len(available_metrics) >= 3:
        # Normalize all metrics to 0-1 scale for radar chart
        df_norm = df[available_metrics].copy()
        for col in df_norm.columns:
            if df_norm[col].max() > 0:
                df_norm[col] = df_norm[col] / df_norm[col].max()
        
        # Create radar chart
        fig = plt.figure(figsize=(10, 8))
        
        # Number of metrics
        N = len(available_metrics)
        
        # Angle for each metric
        angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create subplot
        ax = plt.subplot(111, polar=True)
        
        # Draw one axis per metric and add labels
        plt.xticks(angles[:-1], available_metrics, size=12)
        
        # Draw y-axis scale and labels
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
        plt.ylim(0, 1)
        
        # Plot each model
        for i, model in enumerate(df_norm.index):
            values = df_norm.loc[model].values.flatten().tolist()
            values += values[:1]  # Close the loop
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
            ax.fill(angles, values, alpha=0.1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title("Model Performance Comparison", size=15, y=1.1)
        plt.tight_layout()
        plt.savefig("comparison_results/radar_chart.png", dpi=300, bbox_inches="tight")
        plt.close()

def create_markdown_report(df):
    """Generate a markdown report with model comparison results."""
    report = "# Model Comparison Report\n\n"
    report += f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    report += "## Overall Ranking\n\n"
    report += "Models ranked by test accuracy:\n\n"
    
    # Create table header
    cols = ["Model", "Accuracy", "F1 Macro", "F1 Weighted", "Training Time (min)", "Memory (GB)"]
    available_cols = []
    for col in cols:
        if col == "Model":
            available_cols.append(col)
        elif col == "Training Time (min)" and "training_time_minutes" in df.columns:
            available_cols.append(col)
        elif col == "Memory (GB)" and "peak_memory_gb" in df.columns:
            available_cols.append(col)
        elif col in df.columns:
            available_cols.append(col)
    
    # Create table header
    report += "| " + " | ".join(available_cols) + " |\n"
    report += "| " + " | ".join(["---" for _ in available_cols]) + " |\n"
    
    # Add rows
    for model in df.index:
        row = [model]
        for col in available_cols[1:]:
            if col == "Training Time (min)":
                value = df.loc[model, "training_time_minutes"]
                row.append(f"{value:.1f}")
            elif col == "Memory (GB)":
                value = df.loc[model, "peak_memory_gb"]
                row.append(f"{value:.1f}")
            else:
                value = df.loc[model, col]
                row.append(f"{value:.4f}")
        
        report += "| " + " | ".join(row) + " |\n"
    
    report += "\n## Performance Visualization\n\n"
    report += "![Accuracy Comparison](comparison_results/accuracy_comparison.png)\n\n"
    report += "![F1 Macro Comparison](comparison_results/f1_macro_comparison.png)\n\n"
    
    if "training_time_minutes" in df.columns:
        report += "![Training Time Comparison](comparison_results/training_time_comparison.png)\n\n"
    
    if "peak_memory_gb" in df.columns:
        report += "![Memory Usage Comparison](comparison_results/memory_usage_comparison.png)\n\n"
    
    report += "![Radar Chart](comparison_results/radar_chart.png)\n\n"
    
    report += "## Conclusion\n\n"
    
    best_model = df.index[0]
    report += f"The **{best_model}** model achieved the best overall performance "
    report += f"with {df.loc[best_model, 'Accuracy']:.2%} accuracy and "
    report += f"{df.loc[best_model, 'F1 Macro']:.2%} F1 Macro score on the test set.\n\n"
    
    report += "### Recommendations\n\n"
    report += "Based on the comparison results:\n\n"
    
    # Performance-focused recommendation
    report += "1. **For best performance**: Use the " + best_model + " model.\n"
    
    # Balance recommendation
    if "training_time_minutes" in df.columns and len(df) > 1:
        df['efficiency'] = df['Accuracy'] / df['training_time_minutes']
        most_efficient = df['efficiency'].idxmax()
        report += f"2. **For balanced performance and speed**: Consider the {most_efficient} model.\n"
    
    # Memory recommendation
    if "peak_memory_gb" in df.columns and len(df) > 1:
        lowest_memory = df['peak_memory_gb'].idxmin()
        report += f"3. **For memory-constrained environments**: The {lowest_memory} model uses the least memory.\n"
    
    # Save report
    with open("comparison_results/model_comparison_report.md", "w") as f:
        f.write(report)
    
    return report

def main():
    """Main function to train and compare models."""
    parser = argparse.ArgumentParser(description="Train and compare multiple models")
    parser.add_argument(
        "--configs_dir",
        type=str,
        default="configs",
        help="Directory containing model configuration files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save model outputs"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Specific model configs to train (e.g., vit_base swin_base)"
    )
    
    args = parser.parse_args()
    
    # Get configuration files
    configs_dir = Path(args.configs_dir)
    
    if args.models:
        config_files = [configs_dir / f"{model}.yaml" for model in args.models]
    else:
        config_files = list(configs_dir.glob("*.yaml"))
    
    # Validate config files
    valid_configs = []
    for config_file in config_files:
        if config_file.exists():
            valid_configs.append(config_file)
        else:
            print(f"Warning: Configuration file not found: {config_file}")
    
    if not valid_configs:
        print("No valid configuration files found. Exiting.")
        return
    
    print(f"Found {len(valid_configs)} configuration files for training:")
    for config in valid_configs:
        print(f"  - {config}")
    
    # Train models
    results = {}
    for config_file in valid_configs:
        model_name = config_file.stem
        
        # Load config to get model details
        config = load_config(config_file)
        model_type = config.get("model", {}).get("model_type", "unknown")
        model_backbone = config.get("model", {}).get("name", "unknown")
        
        # Run training
        success, log_path = run_training(str(config_file), args.output_dir)
        
        if success and log_path:
            # Extract metrics
            metrics = extract_metrics(log_path)
            
            # Store results
            results[model_name] = metrics
            results[model_name]["model_type"] = model_type
            results[model_name]["backbone"] = model_backbone
            
            print(f"Metrics for {model_name}:")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
    
    # Compare models if we have results
    if results:
        print("\nComparing models...")
        comparison_df = compare_models(results)
        
        print("\nComparison complete. Results saved to comparison_results/")
        
        # Display top 3 models
        print("\nTop performing models:")
        for i, (model, row) in enumerate(comparison_df.head(3).iterrows(), 1):
            print(f"{i}. {model}: {row['Accuracy']:.4f} accuracy, {row.get('F1 Macro', 0):.4f} F1")
    else:
        print("No training results to compare.")

if __name__ == "__main__":
    main()
