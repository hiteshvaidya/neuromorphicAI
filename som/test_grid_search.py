#!/usr/bin/env python3
"""
test_grid_search.py

Simple test script to verify grid search setup
"""

import subprocess
import sys
import os

def test_dependencies():
    """Test if all required dependencies are available"""
    print("Testing dependencies...")
    
    try:
        import pandas as pd
        print("✓ pandas")
    except ImportError:
        print("✗ pandas - install with: conda install pandas")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib")
    except ImportError:
        print("✗ matplotlib - install with: conda install matplotlib")
        return False
    
    try:
        import seaborn as sns
        print("✓ seaborn")
    except ImportError:
        print("✗ seaborn - install with: conda install seaborn")
        return False
    
    try:
        import numpy as np
        print("✓ numpy")
    except ImportError:
        print("✗ numpy - install with: conda install numpy")
        return False
    
    try:
        import tensorflow as tf
        print("✓ tensorflow")
    except ImportError:
        print("✗ tensorflow - install with: conda install tensorflow")
        return False
    
    return True

def test_gpu_setup():
    """Test GPU setup"""
    print("\nTesting GPU setup...")
    
    try:
        result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_count = len(result.stdout.strip().split('\n'))
            print(f"✓ Found {gpu_count} GPU(s)")
            print(result.stdout.strip())
            return gpu_count
        else:
            print("✗ No GPUs detected")
            return 0
    except FileNotFoundError:
        print("✗ nvidia-smi not found")
        return 0

def test_conda_env():
    """Test conda environment"""
    print("\nTesting conda environment...")
    
    try:
        result = subprocess.run(['conda', 'env', 'list'], 
                              capture_output=True, text=True)
        if 'som' in result.stdout:
            print("✓ Conda environment 'som' found")
            return True
        else:
            print("✗ Conda environment 'som' not found")
            return False
    except FileNotFoundError:
        print("✗ conda not found")
        return False

def test_data_directory():
    """Test data directory structure"""
    print("\nTesting data directory...")
    
    data_dir = "../data"
    datasets = ["mnist", "fashion", "kmnist"]
    
    if not os.path.exists(data_dir):
        print(f"✗ Data directory {data_dir} not found")
        return False
    
    for dataset in datasets:
        dataset_dir = os.path.join(data_dir, dataset)
        if os.path.exists(dataset_dir):
            print(f"✓ {dataset} dataset found")
            
            # Check for train/test/valid directories
            for split in ["train", "test", "valid"]:
                split_dir = os.path.join(dataset_dir, split)
                if os.path.exists(split_dir):
                    print(f"  ✓ {split} directory found")
                else:
                    print(f"  ✗ {split} directory not found")
        else:
            print(f"✗ {dataset} dataset not found")
    
    return True

def run_small_test():
    """Run a small grid search test"""
    print("\nRunning small grid search test...")
    
    # Get GPU count for parallel execution
    gpu_count = test_gpu_setup()
    max_workers = max(1, gpu_count)  # Use at least 1 worker, more if GPUs available
    
    cmd = [
        "python", "grid_search_transfer_metrics.py",
        "--conda-env", "som",
        "--max-workers", str(max_workers),
        "--results-dir", "test_results",
        "--config-file", "grid_search_config_small.json"
    ]
    
    print("Command:", " ".join(cmd))
    print(f"This will run a small grid search using {max_workers} workers (GPUs)")
    print("This will run a small grid search with 2 datasets × 2 units × 2 radius × 2 learning_rate × 2 variance × 2 vanilla = 64 experiments")
    if max_workers > 1:
        estimated_time = f"Estimated time: {5 + (10 // max_workers)}-{15 + (30 // max_workers)} minutes with {max_workers} GPUs"
    else:
        estimated_time = "Estimated time: 10-30 minutes with CPU"
    print(estimated_time)
    
    response = input("Do you want to run the test? (y/N): ")
    if response.lower() == 'y':
        try:
            result = subprocess.run(cmd, cwd=os.getcwd())
            return result.returncode == 0
        except Exception as e:
            print(f"Error running test: {e}")
            return False
    else:
        print("Test skipped")
        return True

def main():
    print("Grid Search Setup Test")
    print("=" * 30)
    
    # Test dependencies
    deps_ok = test_dependencies()
    
    # Test GPU setup
    gpu_count = test_gpu_setup()
    
    # Test conda environment
    conda_ok = test_conda_env()
    
    # Test data directory
    data_ok = test_data_directory()
    
    print("\n" + "=" * 30)
    print("SUMMARY")
    print("=" * 30)
    
    if deps_ok:
        print("✓ All dependencies available")
    else:
        print("✗ Missing dependencies - install with:")
        print("  conda install pandas matplotlib seaborn numpy tensorflow")
    
    if gpu_count > 0:
        print(f"✓ {gpu_count} GPU(s) available for parallel execution")
    else:
        print("✗ No GPUs available - will use CPU (slower)")
    
    if conda_ok:
        print("✓ Conda environment 'som' ready")
    else:
        print("✗ Conda environment 'som' not found")
    
    if data_ok:
        print("✓ Data directory structure looks good")
    else:
        print("✗ Data directory issues detected")
    
    if deps_ok and conda_ok and data_ok:
        print("\n✓ Setup appears to be ready for grid search!")
        
        # Offer to run small test
        run_small_test()
    else:
        print("\n✗ Setup issues detected. Please fix before running grid search.")
        sys.exit(1)

if __name__ == "__main__":
    main()
