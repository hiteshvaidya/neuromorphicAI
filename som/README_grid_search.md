# Grid Search for Transfer Metric SOM

This directory contains a comprehensive grid search system for the Transfer Metric SOM (`transfer_metric_som.py`) with multi-GPU support and visualization capabilities.

## Features

- **Multi-GPU Support**: Automatically detects and utilizes all available GPUs for parallel execution
- **Comprehensive Parameter Grid**: Tests multiple combinations of SOM parameters
- **Multiple Datasets**: Supports MNIST, Fashion-MNIST, and Kuzushiji-MNIST (referred to as kmnist)
- **Performance Tracking**: Tracks key metrics including BWT, learning accuracy, average accuracy, forgetting measure, and memory requirements
- **Rich Visualizations**: Generates plots for parameter sensitivity, dataset comparison, correlation analysis, and best configurations
- **Automated Reports**: Creates summary reports with best configurations and statistics

## Files

- `grid_search_transfer_metrics.py`: Main grid search program
- `grid_search_config.json`: Full parameter grid configuration
- `grid_search_config_small.json`: Smaller configuration for testing
- `run_grid_search.sh`: Shell script to easily run grid search
- `test_grid_search.py`: Setup verification script
- `grid_search_requirements.txt`: Additional Python dependencies

## Setup

### 1. Install Dependencies

Make sure you have the required packages in your conda environment:

```bash
conda activate tf
conda install pandas matplotlib seaborn numpy tensorflow
```

Or install from requirements file:

```bash
conda activate tf
pip install -r grid_search_requirements.txt
```

### 2. Verify Setup

Run the test script to verify everything is set up correctly:

```bash
conda activate tf
python test_grid_search.py
```

This will check:
- Required Python packages
- GPU availability
- Conda environment
- Data directory structure

## Usage

### Quick Start

Run the grid search with default configuration:

```bash
./run_grid_search.sh
```

### Custom Configuration

1. **Edit Configuration**: Modify `grid_search_config.json` to customize the parameter grid
2. **Run Grid Search**:
   ```bash
   conda activate tf
   python grid_search_transfer_metrics.py --config-file grid_search_config.json
   ```

### Command Line Options

```bash
python grid_search_transfer_metrics.py [OPTIONS]

Options:
  --conda-env ENV         Conda environment name (default: tf)
  --max-workers N         Maximum parallel workers/GPUs (default: auto-detect)
  --results-dir DIR       Results directory (default: grid_search_results)
  --config-file FILE      JSON configuration file
  --visualize-only        Only create visualizations from existing results
```

## Configuration

The grid search configuration is defined in JSON format. Example:

```json
{
  "dataset": ["mnist", "fashion", "kmnist"],
  "units": [10, 15, 20, 25],
  "radius": [0.5, 0.8, 1.0, 1.2, 1.5],
  "learning_rate": [0.05, 0.07, 0.1, 0.12],
  "variance_alpha": [0.8, 0.9, 0.95],
  "variance": [0.5, 1.0, 1.5, 2.0],
  "task_size": [1, 2],
  "tau_radius": [6, 8, 10, 12],
  "tau_lr": [35, 40, 45, 50],
  "vanilla": [false, true]
}
```

### Parameters Explained

- **dataset**: Datasets to test (mnist, fashion, kmnist)
- **units**: Number of units in SOM grid (e.g., 15 = 15×15 grid)
- **radius**: Initial neighborhood radius
- **learning_rate**: Initial learning rate
- **variance_alpha**: Alpha for running variance (0-1)
- **variance**: Initial variance value
- **task_size**: Classes per task in incremental learning
- **tau_radius**: Radius decay time constant
- **tau_lr**: Learning rate decay time constant
- **vanilla**: Whether to use vanilla SOM (true) or contSOM (false)

## Results

The grid search generates several outputs:

### 1. Raw Results
- `grid_search_results.csv`: Detailed results for all experiments
- `grid_search_config.json`: Configuration used
- `logs/`: Individual experiment logs

### 2. Visualizations
- `performance_summary.png`: Distribution of performance metrics
- `parameter_sensitivity_[metric].png`: Parameter effects on each metric
- `dataset_comparison.png`: Performance comparison across datasets
- `correlation_matrix.png`: Parameter and metric correlations
- `best_configurations.png`: Top configurations for each metric
- `parameter_interactions_[metric].png`: Parameter interaction effects

### 3. Reports
- `grid_search_report.txt`: Summary statistics and best configurations

## Performance Metrics

The system tracks these key metrics:

1. **BWT (Backward Transfer)**: How much learning new tasks affects old tasks
2. **Average Accuracy**: Mean accuracy across all tasks after training
3. **Learning Accuracy**: Accuracy when learning each new task
4. **Forgetting Measure**: Amount of knowledge forgotten over time
5. **Memory**: Model memory requirements in bytes

## Multi-GPU Execution

The system automatically:
- Detects available GPUs
- Distributes experiments across GPUs
- Manages CUDA device assignment
- Handles parallel execution safely

## Example Workflows

### 1. Quick Test (Small Grid)
```bash
# Test with minimal configuration
python grid_search_transfer_metrics.py --config-file grid_search_config_small.json --max-workers 1
```

### 2. Full Grid Search
```bash
# Run complete grid search with all GPUs
./run_grid_search.sh
```

### 3. Analysis Only
```bash
# Generate visualizations from existing results
python grid_search_transfer_metrics.py --visualize-only --results-dir existing_results
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Install missing packages with conda/pip
2. **GPU Not Detected**: Check NVIDIA drivers and CUDA installation
3. **Out of Memory**: Reduce batch size or number of parallel workers
4. **Long Execution Time**: Use smaller configuration for testing

### Performance Tips

1. **Use Multiple GPUs**: Significantly reduces execution time
2. **Start Small**: Test with `grid_search_config_small.json` first
3. **Monitor Resources**: Check GPU memory usage during execution
4. **Save Intermediate Results**: Results are saved continuously

## Expected Runtime

Runtime depends on:
- Number of parameter combinations
- Available GPUs
- Dataset size
- Hardware specifications

Examples:
- Small config (64 experiments): 30-60 minutes on single GPU
- Full config (1000+ experiments): Several hours with multiple GPUs

## Integration with Existing Code

The grid search system:
- Uses existing `transfer_metric_som.py` without modifications
- Respects all command-line arguments
- Maintains compatibility with current workflow
- Generates results in same format as manual runs
