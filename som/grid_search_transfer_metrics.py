#!/usr/bin/env python3
"""
grid_search_transfer_metrics.py

Description: Grid search program for transfer_metric_som.py with multi-GPU support
Author: Auto-generated for neuromorphicAI project
Date: June 2025
"""

import subprocess
import itertools
import multiprocessing
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import argparse
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib and seaborn for better plots
plt.style.use('default')
sns.set_palette("husl")


class GridSearchManager:
    def __init__(self, conda_env="som", max_workers=None, results_dir="grid_search_results"):
        """
        Initialize GridSearchManager
        
        :param conda_env: Name of conda environment to use
        :param max_workers: Maximum number of parallel workers (GPUs)
        :param results_dir: Directory to store results
        """
        self.conda_env = conda_env
        self.max_workers = max_workers or self._get_gpu_count()
        self.results_dir = results_dir
        self.results_file = os.path.join(results_dir, "grid_search_results.csv")
        self.config_file = os.path.join(results_dir, "grid_search_config.json")
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize results DataFrame
        self.results_df = pd.DataFrame()
        
    def _get_gpu_count(self):
        """Get number of available GPUs"""
        try:
            result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_count = len(result.stdout.strip().split('\n'))
                print(f"Detected {gpu_count} GPUs")
                return gpu_count
            else:
                print("No GPUs detected, using CPU")
                return 1
        except FileNotFoundError:
            print("nvidia-smi not found, using CPU")
            return 1
    
    def generate_parameter_grid(self, grid_config):
        """
        Generate all parameter combinations from grid configuration
        
        :param grid_config: Dictionary with parameter ranges
        :return: List of parameter dictionaries
        """
        # Extract parameter names and values
        param_names = list(grid_config.keys())
        param_values = list(grid_config.values())
        
        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        
        # Convert to list of dictionaries
        param_grid = []
        for combo in combinations:
            param_dict = dict(zip(param_names, combo))
            param_grid.append(param_dict)
        
        print(f"Generated {len(param_grid)} parameter combinations")
        return param_grid
    
    def run_single_experiment(self, params, gpu_id, experiment_id):
        """
        Run a single experiment with given parameters
        
        :param params: Dictionary of parameters
        :param gpu_id: GPU ID to use
        :param experiment_id: Unique experiment identifier
        :return: Dictionary with results
        """
        try:
            # Create experiment directory
            exp_dir = f"grid_search_{params['dataset']}_{experiment_id}"
            log_dir = os.path.join(self.results_dir, "logs", exp_dir)
            os.makedirs(log_dir, exist_ok=True)
            
            # Build command
            cmd = self._build_command(params, gpu_id, exp_dir)
            
            print(f"[GPU {gpu_id}] Running experiment {experiment_id}: {params['dataset']}")
            print(f"[GPU {gpu_id}] Command: {' '.join(cmd)}")
            
            # Run experiment
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  cwd=os.getcwd())
            end_time = time.time()
            
            if result.returncode != 0:
                print(f"[GPU {gpu_id}] Experiment {experiment_id} failed:")
                print(f"[GPU {gpu_id}] Error: {result.stderr}")
                return None
            
            # Parse results
            results = self._parse_results(log_dir, params, experiment_id)
            results['execution_time'] = end_time - start_time
            results['gpu_id'] = gpu_id
            
            print(f"[GPU {gpu_id}] Experiment {experiment_id} completed successfully")
            return results
            
        except Exception as e:
            print(f"[GPU {gpu_id}] Error in experiment {experiment_id}: {str(e)}")
            return None
    
    def _build_command(self, params, gpu_id, exp_dir):
        """Build command line for transfer_metric_som.py"""
        cmd = [
            "conda", "run", "-n", self.conda_env,
            "env", f"CUDA_VISIBLE_DEVICES={gpu_id}",
            "python", "transfer_metric_som.py"
        ]
        
        # Add parameters
        cmd.extend(["-u", str(params["units"])])
        cmd.extend(["-r", str(params["radius"])])
        cmd.extend(["-lr", str(params["learning_rate"])])
        cmd.extend(["-va", str(params["variance_alpha"])])
        cmd.extend(["-v", str(params["variance"])])
        cmd.extend(["-fp", exp_dir])
        cmd.extend(["-d", params["dataset"]])
        cmd.extend(["-tr", str(params["tau_radius"])])
        cmd.extend(["-tlr", str(params["tau_lr"])])
        cmd.extend(["-us", str(params["unit_size"])])
        cmd.extend(["-nt", str(params["n_tasks"])])
        cmd.extend(["-ts", str(params["task_size"])])
        cmd.extend(["-t", params["training_type"]])
        
        if params.get("vanilla", False):
            cmd.extend(["-vanilla", "True"])
        
        return cmd
    
    def _parse_results(self, log_dir, params, experiment_id):
        """Parse results from experiment log files"""
        results = params.copy()
        results['experiment_id'] = experiment_id
        
        # Parse metrics from metrics.csv
        metrics_file = os.path.join("logs", log_dir, "metrics.csv")
        try:
            if os.path.exists(metrics_file):
                df = pd.read_csv(metrics_file)
                if len(df) > 0:
                    row = df.iloc[0]
                    results['bwt'] = row['bwt']
                    results['fwt'] = row['fwt'] 
                    results['average_accuracy'] = row['AA']
                    results['learning_accuracy'] = row['LA']
                    results['forgetting_measure'] = row['FM']
                    results['memory'] = row['mem']
            else:
                print(f"Warning: metrics.csv not found for experiment {experiment_id}")
                # Set default values
                for metric in ['bwt', 'fwt', 'average_accuracy', 'learning_accuracy', 
                             'forgetting_measure', 'memory']:
                    results[metric] = np.nan
        except Exception as e:
            print(f"Error parsing results for experiment {experiment_id}: {str(e)}")
            for metric in ['bwt', 'fwt', 'average_accuracy', 'learning_accuracy', 
                         'forgetting_measure', 'memory']:
                results[metric] = np.nan
        
        return results
    
    def run_grid_search(self, grid_config):
        """
        Run grid search with all parameter combinations
        
        :param grid_config: Dictionary with parameter ranges
        """
        # Generate parameter grid
        param_grid = self.generate_parameter_grid(grid_config)
        
        # Save configuration
        config_data = {
            'grid_config': grid_config,
            'total_experiments': len(param_grid),
            'max_workers': self.max_workers,
            'conda_env': self.conda_env,
            'start_time': datetime.now().isoformat()
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"Starting grid search with {len(param_grid)} experiments using {self.max_workers} workers")
        
        # Run experiments in parallel
        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_params = {}
            for i, params in enumerate(param_grid):
                gpu_id = i % self.max_workers
                future = executor.submit(self.run_single_experiment, params, gpu_id, i)
                future_to_params[future] = (params, i)
            
            # Collect results
            for future in as_completed(future_to_params):
                params, exp_id = future_to_params[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                        # Save intermediate results
                        self._save_intermediate_results(results)
                except Exception as e:
                    print(f"Exception in experiment {exp_id}: {str(e)}")
        
        # Convert to DataFrame and save final results
        self.results_df = pd.DataFrame(results)
        self.results_df.to_csv(self.results_file, index=False)
        
        print(f"Grid search completed! Results saved to {self.results_file}")
        print(f"Total successful experiments: {len(results)}")
        
        return self.results_df
    
    def _save_intermediate_results(self, results):
        """Save intermediate results during grid search"""
        if results:
            df = pd.DataFrame(results)
            df.to_csv(self.results_file, index=False)
    
    def load_results(self, results_file=None):
        """Load results from CSV file"""
        if results_file is None:
            results_file = self.results_file
        
        if os.path.exists(results_file):
            self.results_df = pd.read_csv(results_file)
            print(f"Loaded {len(self.results_df)} results from {results_file}")
            return self.results_df
        else:
            print(f"Results file {results_file} not found")
            return None
    
    def create_visualizations(self, output_dir=None):
        """Create comprehensive visualizations of grid search results"""
        if output_dir is None:
            output_dir = os.path.join(self.results_dir, "visualizations")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.results_df.empty:
            print("No results to visualize")
            return
        
        print("Creating visualizations...")
        
        # Performance metrics to analyze
        metrics = ['bwt', 'average_accuracy', 'learning_accuracy', 'forgetting_measure', 'memory']
        
        # Parameters to analyze
        param_cols = ['units', 'radius', 'learning_rate', 'variance_alpha', 'variance', 
                     'task_size', 'tau_radius', 'tau_lr', 'dataset']
        
        # 1. Overall performance summary
        self._plot_performance_summary(output_dir, metrics)
        
        # 2. Parameter sensitivity analysis
        self._plot_parameter_sensitivity(output_dir, param_cols, metrics)
        
        # 3. Dataset comparison
        self._plot_dataset_comparison(output_dir, metrics)
        
        # 4. Correlation analysis
        self._plot_correlation_analysis(output_dir, param_cols, metrics)
        
        # 5. Best configurations
        self._plot_best_configurations(output_dir, metrics)
        
        # 6. Parameter interaction heatmaps
        self._plot_parameter_interactions(output_dir, param_cols, metrics)
        
        print(f"Visualizations saved to {output_dir}")
    
    def _plot_performance_summary(self, output_dir, metrics):
        """Plot overall performance summary"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            data = self.results_df[metric].dropna()
            
            if len(data) > 0:
                ax.hist(data, bins=30, alpha=0.7, edgecolor='black')
                ax.set_title(f'Distribution of {metric.replace("_", " ").title()}')
                ax.set_xlabel(metric.replace("_", " ").title())
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                
                # Add statistics
                mean_val = data.mean()
                std_val = data.std()
                ax.axvline(mean_val, color='red', linestyle='--', 
                          label=f'Mean: {mean_val:.4f}')
                ax.legend()
        
        # Remove empty subplot
        if len(metrics) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_sensitivity(self, output_dir, param_cols, metrics):
        """Plot parameter sensitivity analysis"""
        numeric_params = []
        for param in param_cols:
            if param in self.results_df.columns and self.results_df[param].dtype in ['int64', 'float64']:
                numeric_params.append(param)
        
        for metric in metrics:
            if metric not in self.results_df.columns:
                continue
                
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            axes = axes.flatten()
            
            for i, param in enumerate(numeric_params[:8]):  # Limit to 8 parameters
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                
                # Create scatter plot
                x_data = self.results_df[param].dropna()
                y_data = self.results_df.loc[x_data.index, metric].dropna()
                
                if len(x_data) > 0 and len(y_data) > 0:
                    ax.scatter(x_data, y_data, alpha=0.6)
                    
                    # Add trend line
                    z = np.polyfit(x_data, y_data, 1)
                    p = np.poly1d(z)
                    ax.plot(x_data, p(x_data), "r--", alpha=0.8)
                    
                    # Calculate correlation
                    corr = np.corrcoef(x_data, y_data)[0, 1]
                    ax.set_title(f'{param} vs {metric} (r={corr:.3f})')
                    ax.set_xlabel(param.replace("_", " ").title())
                    ax.set_ylabel(metric.replace("_", " ").title())
                    ax.grid(True, alpha=0.3)
            
            # Remove empty subplots
            for j in range(len(numeric_params), len(axes)):
                fig.delaxes(axes[j])
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'parameter_sensitivity_{metric}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_dataset_comparison(self, output_dir, metrics):
        """Plot dataset comparison"""
        if 'dataset' not in self.results_df.columns:
            return
        
        datasets = self.results_df['dataset'].unique()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Box plot for each dataset
            data_by_dataset = []
            labels = []
            
            for dataset in datasets:
                data = self.results_df[self.results_df['dataset'] == dataset][metric].dropna()
                if len(data) > 0:
                    data_by_dataset.append(data)
                    labels.append(dataset)
            
            if data_by_dataset:
                ax.boxplot(data_by_dataset, labels=labels)
                ax.set_title(f'{metric.replace("_", " ").title()} by Dataset')
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.grid(True, alpha=0.3)
                
                # Rotate x-axis labels if needed
                plt.setp(ax.get_xticklabels(), rotation=45)
        
        # Remove empty subplot
        if len(metrics) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dataset_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_analysis(self, output_dir, param_cols, metrics):
        """Plot correlation matrix"""
        # Select numeric columns
        numeric_cols = []
        for col in param_cols + metrics:
            if col in self.results_df.columns and self.results_df[col].dtype in ['int64', 'float64']:
                numeric_cols.append(col)
        
        if len(numeric_cols) < 2:
            return
        
        # Calculate correlation matrix
        corr_matrix = self.results_df[numeric_cols].corr()
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f')
        plt.title('Parameter and Metric Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_best_configurations(self, output_dir, metrics):
        """Plot best configurations for each metric"""
        fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 4*len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Find best configurations (top 10)
            if metric in ['forgetting_measure', 'memory']:
                # Lower is better
                best_configs = self.results_df.nsmallest(10, metric)
            else:
                # Higher is better
                best_configs = self.results_df.nlargest(10, metric)
            
            if len(best_configs) > 0:
                # Create labels for configurations
                labels = []
                for idx, row in best_configs.iterrows():
                    label = f"{row['dataset']}\nu={row['units']}, r={row['radius']}"
                    labels.append(label)
                
                values = best_configs[metric].values
                bars = ax.bar(range(len(values)), values)
                ax.set_title(f'Top 10 Configurations by {metric.replace("_", " ").title()}')
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.grid(True, alpha=0.3)
                
                # Color bars by dataset if multiple datasets
                if 'dataset' in self.results_df.columns:
                    datasets = best_configs['dataset'].unique()
                    colors = plt.cm.Set3(np.linspace(0, 1, len(datasets)))
                    dataset_to_color = dict(zip(datasets, colors))
                    
                    for bar, dataset in zip(bars, best_configs['dataset']):
                        bar.set_color(dataset_to_color[dataset])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'best_configurations.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_interactions(self, output_dir, param_cols, metrics):
        """Plot parameter interaction heatmaps"""
        numeric_params = []
        for param in param_cols:
            if param in self.results_df.columns and self.results_df[param].dtype in ['int64', 'float64']:
                numeric_params.append(param)
        
        if len(numeric_params) < 2:
            return
        
        for metric in metrics[:3]:  # Limit to first 3 metrics
            if metric not in self.results_df.columns:
                continue
            
            # Create interaction matrix
            n_params = len(numeric_params)
            interaction_matrix = np.zeros((n_params, n_params))
            
            for i, param1 in enumerate(numeric_params):
                for j, param2 in enumerate(numeric_params):
                    if i != j:
                        # Calculate correlation between params and their effect on metric
                        df_subset = self.results_df[[param1, param2, metric]].dropna()
                        if len(df_subset) > 2:
                            corr = np.corrcoef(df_subset[param1] * df_subset[param2], 
                                             df_subset[metric])[0, 1]
                            interaction_matrix[i, j] = corr
            
            # Plot heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(interaction_matrix, 
                       xticklabels=[p.replace('_', ' ').title() for p in numeric_params],
                       yticklabels=[p.replace('_', ' ').title() for p in numeric_params],
                       annot=True, cmap='coolwarm', center=0, fmt='.3f')
            plt.title(f'Parameter Interactions Effect on {metric.replace("_", " ").title()}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'parameter_interactions_{metric}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_report(self, output_file=None):
        """Generate a comprehensive report of grid search results"""
        if output_file is None:
            output_file = os.path.join(self.results_dir, "grid_search_report.txt")
        
        if self.results_df.empty:
            print("No results to report")
            return
        
        with open(output_file, 'w') as f:
            f.write("GRID SEARCH RESULTS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic statistics
            f.write(f"Total experiments: {len(self.results_df)}\n")
            f.write(f"Datasets tested: {', '.join(self.results_df['dataset'].unique())}\n\n")
            
            # Performance metrics summary
            metrics = ['bwt', 'average_accuracy', 'learning_accuracy', 'forgetting_measure', 'memory']
            f.write("PERFORMANCE METRICS SUMMARY\n")
            f.write("-" * 30 + "\n")
            
            for metric in metrics:
                if metric in self.results_df.columns:
                    data = self.results_df[metric].dropna()
                    if len(data) > 0:
                        f.write(f"{metric.replace('_', ' ').title()}:\n")
                        f.write(f"  Mean: {data.mean():.4f}\n")
                        f.write(f"  Std:  {data.std():.4f}\n")
                        f.write(f"  Min:  {data.min():.4f}\n")
                        f.write(f"  Max:  {data.max():.4f}\n\n")
            
            # Best configurations
            f.write("BEST CONFIGURATIONS\n")
            f.write("-" * 20 + "\n")
            
            for metric in metrics[:3]:  # Top 3 metrics
                if metric in self.results_df.columns:
                    if metric in ['forgetting_measure', 'memory']:
                        best = self.results_df.loc[self.results_df[metric].idxmin()]
                        f.write(f"Best {metric.replace('_', ' ').title()} (lowest): {best[metric]:.4f}\n")
                    else:
                        best = self.results_df.loc[self.results_df[metric].idxmax()]
                        f.write(f"Best {metric.replace('_', ' ').title()} (highest): {best[metric]:.4f}\n")
                    
                    f.write(f"  Dataset: {best['dataset']}\n")
                    f.write(f"  Units: {best['units']}\n")
                    f.write(f"  Radius: {best['radius']}\n")
                    f.write(f"  Learning Rate: {best['learning_rate']}\n\n")
        
        print(f"Report generated: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Grid Search for Transfer Metric SOM")
    parser.add_argument('--conda-env', default='som', help='Conda environment name')
    parser.add_argument('--max-workers', type=int, help='Maximum number of parallel workers')
    parser.add_argument('--results-dir', default='grid_search_results', help='Results directory')
    parser.add_argument('--config-file', help='JSON file with grid search configuration')
    parser.add_argument('--visualize-only', action='store_true', help='Only create visualizations from existing results')
    
    args = parser.parse_args()
    
    # Initialize grid search manager
    manager = GridSearchManager(
        conda_env=args.conda_env,
        max_workers=args.max_workers,
        results_dir=args.results_dir
    )
    
    if args.visualize_only:
        # Only create visualizations
        manager.load_results()
        manager.create_visualizations()
        manager.generate_report()
        return
    
    # Define grid search configuration
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            grid_config = json.load(f)
    else:
        # Default grid configuration
        grid_config = {
            'dataset': ['mnist', 'fashion', 'kmnist'],
            'units': [10, 15, 20],
            'radius': [0.5, 1.0, 1.5],
            'learning_rate': [0.05, 0.07, 0.1],
            'variance_alpha': [0.8, 0.9, 0.95],
            'variance': [0.5, 1.0, 1.5],
            'task_size': [1],
            'tau_radius': [8, 10, 12],
            'tau_lr': [40, 45, 50],
            'unit_size': [28],
            'n_tasks': [10],
            'training_type': ['class'],
            'vanilla': [False, True]
        }
    
    print("Starting grid search...")
    print(f"Configuration: {grid_config}")
    
    # Run grid search
    results_df = manager.run_grid_search(grid_config)
    
    # Create visualizations and report
    manager.create_visualizations()
    manager.generate_report()
    
    print("Grid search completed successfully!")


if __name__ == "__main__":
    main()
