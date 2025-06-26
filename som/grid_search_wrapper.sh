#!/bin/bash
# Simple wrapper to run grid search in som environment
conda run -n som python grid_search_transfer_metrics.py "$@"
