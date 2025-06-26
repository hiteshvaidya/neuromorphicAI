#!/usr/bin/env python3
"""
Extract grid search experiment parameters and results
using model_config.pkl and metrics.csv files
"""

import os
import pickle
import pandas as pd
import csv
import re
from glob import glob

def extract_experiment_info(exp_dir):
    """Extract dataset and experiment ID from directory name"""
    match = re.match(r'grid_search_([^_]+)_(\d+)', os.path.basename(exp_dir))
    if match:
        dataset = match.group(1)
        exp_id = int(match.group(2))
        return dataset, exp_id
    return None, None

def load_model_config(exp_dir):
    """Load model configuration from model_config.pkl"""
    config_file = os.path.join(exp_dir, "model_config.pkl")
    if os.path.exists(config_file):
        try:
            with open(config_file, 'rb') as f:
                config = pickle.load(f)
                
            # Extract scalar parameters (not tensors)
            params = {}
            if 'tau_radius' in config:
                params['tau_radius'] = float(config['tau_radius'])
            if 'tau_lr' in config:
                params['tau_lr'] = float(config['tau_lr'])
            if 'unitsX' in config:
                params['units'] = int(config['unitsX'])
            if 'shapeX' in config and 'unitsX' in config:
                params['unit_size'] = int(config['shapeX'] // config['unitsX'])
                
            return params
        except Exception as e:
            print(f"Error loading config from {config_file}: {e}")
            return {}
    return {}

def load_metrics(exp_dir):
    """Load metrics from metrics.csv"""
    metrics_file = os.path.join(exp_dir, "metrics.csv")
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)  # Read header row
                values = next(reader)   # Read data row
                
            metrics = {}
            for i, header in enumerate(headers):
                if header == 'b':
                    continue  # Skip the 'b' array
                try:
                    metrics[header] = float(values[i])
                except:
                    metrics[header] = values[i]
                    
            return metrics
        except Exception as e:
            print(f"Error loading metrics from {metrics_file}: {e}")
            return {}
    return {}

def analyze_experiments():
    """Analyze all grid search experiments"""
    results = []
    
    # Find all grid search experiment directories
    pattern = os.path.join("logs", "grid_search_*")
    exp_dirs = glob(pattern)
    exp_dirs.sort()
    
    print(f"Found {len(exp_dirs)} grid search experiments")
    print("-" * 50)
    
    for exp_dir in exp_dirs:
        dataset, exp_id = extract_experiment_info(exp_dir)
        if dataset is None:
            continue
            
        print(f"Processing {os.path.basename(exp_dir)}...")
        
        # Load model configuration parameters
        config_params = load_model_config(exp_dir)
        
        # Load performance metrics
        metrics = load_metrics(exp_dir)
        
        if config_params or metrics:
            result = {
                'experiment_id': exp_id,
                'dataset': dataset,
                'experiment_dir': os.path.basename(exp_dir),
                **config_params,
                **metrics
            }
            results.append(result)
    
    return results

def create_report(results):
    """Create comprehensive report"""
    if not results:
        print("No results to analyze")
        return
    
    df = pd.DataFrame(results)
    
    # Save detailed results to CSV
    df.to_csv("grid_search_experiment_report.csv", index=False)
    print(f"Detailed results saved to: grid_search_experiment_report.csv")
    
    # Create summary report
    with open("grid_search_summary_report.txt", 'w') as f:
        f.write("GRID SEARCH EXPERIMENTS SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Dataset breakdown
        f.write("DATASET BREAKDOWN\n")
        f.write("-" * 20 + "\n")
        dataset_counts = df['dataset'].value_counts()
        for dataset, count in dataset_counts.items():
            f.write(f"{dataset.upper()}: {count} experiments\n")
        f.write(f"TOTAL: {len(df)} experiments\n\n")
        
        # Performance metrics summary
        metrics = ['bwt', 'AA', 'LA', 'FM', 'mem']
        metric_names = {
            'bwt': 'Backward Transfer',
            'AA': 'Average Accuracy', 
            'LA': 'Learning Accuracy',
            'FM': 'Forgetting Measure',
            'mem': 'Memory (bytes)'
        }
        
        f.write("PERFORMANCE METRICS SUMMARY\n")
        f.write("-" * 30 + "\n")
        for metric in metrics:
            if metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 0:
                    f.write(f"{metric_names.get(metric, metric)}:\n")
                    f.write(f"  Mean: {values.mean():.4f}\n")
                    f.write(f"  Std:  {values.std():.4f}\n")
                    f.write(f"  Min:  {values.min():.4f}\n")
                    f.write(f"  Max:  {values.max():.4f}\n\n")
        
        # Best performers by dataset
        f.write("BEST PERFORMERS BY DATASET\n")
        f.write("-" * 30 + "\n")
        for dataset in df['dataset'].unique():
            f.write(f"\n{dataset.upper()} DATASET:\n")
            dataset_df = df[df['dataset'] == dataset]
            
            if 'AA' in dataset_df.columns:
                # Best accuracy
                best_acc_idx = dataset_df['AA'].idxmax()
                best_acc = dataset_df.loc[best_acc_idx]
                f.write(f"  Best Average Accuracy: {best_acc['AA']:.4f}\n")
                f.write(f"    Experiment: {best_acc['experiment_dir']}\n")
                
                # Parameters if available
                param_info = []
                if 'units' in best_acc:
                    param_info.append(f"units={best_acc['units']}")
                if 'tau_radius' in best_acc:
                    param_info.append(f"tau_radius={best_acc['tau_radius']}")
                if 'tau_lr' in best_acc:
                    param_info.append(f"tau_lr={best_acc['tau_lr']}")
                if param_info:
                    f.write(f"    Parameters: {', '.join(param_info)}\n")
            
            if 'bwt' in dataset_df.columns:
                # Best BWT (least negative)
                best_bwt_idx = dataset_df['bwt'].idxmax()
                best_bwt = dataset_df.loc[best_bwt_idx]
                f.write(f"  Best BWT (least forgetting): {best_bwt['bwt']:.4f}\n")
                f.write(f"    Experiment: {best_bwt['experiment_dir']}\n")
        
        # Parameter summary if available
        f.write("\n\nPARAMETER SUMMARY\n")
        f.write("-" * 20 + "\n")
        param_cols = ['units', 'unit_size', 'tau_radius', 'tau_lr']
        for param in param_cols:
            if param in df.columns:
                unique_values = sorted(df[param].dropna().unique())
                f.write(f"{param.upper()}: {len(unique_values)} unique values {unique_values}\n")
    
    print(f"Summary report saved to: grid_search_summary_report.txt")
    
    # Display quick summary
    print("\nQUICK SUMMARY:")
    print(f"Total experiments: {len(df)}")
    print("Dataset breakdown:")
    for dataset, count in dataset_counts.items():
        print(f"  {dataset}: {count}")
    
    if 'AA' in df.columns:
        print(f"Average Accuracy range: {df['AA'].min():.3f} - {df['AA'].max():.3f}")
    if 'bwt' in df.columns:
        print(f"BWT range: {df['bwt'].min():.3f} - {df['bwt'].max():.3f}")

def main():
    print("Grid Search Experiment Analyzer")
    print("=" * 40)
    
    # Change to the correct directory
    os.chdir("/data/hitesh/neuromorphicAI/som")
    
    # Analyze experiments
    results = analyze_experiments()
    
    if results:
        print(f"\nSuccessfully processed {len(results)} experiments")
        create_report(results)
    else:
        print("No valid experiments found!")

if __name__ == "__main__":
    main()
