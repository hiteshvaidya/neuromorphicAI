#!/bin/bash

# Grid Search Runner for Transfer Metric SOM
# This script runs a comprehensive grid search with multi-GPU support

echo "Starting Grid Search for Transfer Metric SOM"
echo "============================================"

# Check if conda environment exists
if ! conda env list | grep -q "tf"; then
    echo "Error: Conda environment 'tf' not found. Please create it first."
    exit 1
fi

# Check for GPUs
echo "Checking available GPUs..."
nvidia-smi --list-gpus 2>/dev/null
if [ $? -eq 0 ]; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "Found $GPU_COUNT GPU(s)"
else
    echo "No GPUs found, will use CPU"
    GPU_COUNT=1
fi

# Create results directory
RESULTS_DIR="grid_search_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR
echo "Results will be saved to: $RESULTS_DIR"

# Run grid search
echo "Starting grid search with $GPU_COUNT parallel workers..."
conda run -n tf python grid_search_transfer_metrics.py \
    --conda-env tf \
    --max-workers $GPU_COUNT \
    --results-dir $RESULTS_DIR \
    --config-file grid_search_config.json

# Check if grid search completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "Grid search completed successfully!"
    echo "Results saved to: $RESULTS_DIR"
    echo ""
    echo "Generated files:"
    echo "- grid_search_results.csv: Raw experimental results"
    echo "- visualizations/: Performance plots and analysis"
    echo "- grid_search_report.txt: Summary report"
    echo ""
    echo "To view visualizations, check the files in $RESULTS_DIR/visualizations/"
else
    echo ""
    echo "Grid search failed. Check the logs for details."
    exit 1
fi
