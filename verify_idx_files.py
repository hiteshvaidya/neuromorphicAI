#!/usr/bin/env python3
"""
verify_idx_files.py

Description: Verify that the downloaded IDX files can be loaded correctly
"""

import os
import sys
sys.path.append('./som')

def read_idx_images(filename):
    """Read IDX format image files"""
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        
        print(f"  Magic: {magic}")
        print(f"  Images: {num_images}")
        print(f"  Dimensions: {rows}x{cols}")
        
        return True

def read_idx_labels(filename):
    """Read IDX format label files"""
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        
        print(f"  Magic: {magic}")
        print(f"  Labels: {num_labels}")
        
        return True

def verify_dataset_idx_files(dataset_name):
    """Verify IDX files for a specific dataset"""
    print(f"\n=== {dataset_name.upper()} ===")
    
    base_path = f"./data/{dataset_name}/raw"
    
    # Check if directory exists
    if not os.path.exists(base_path):
        print(f"❌ Directory not found: {base_path}")
        return False
    
    files_to_check = [
        ("train-images-idx3-ubyte", read_idx_images),
        ("train-labels-idx1-ubyte", read_idx_labels),
        ("t10k-images-idx3-ubyte", read_idx_images),
        ("t10k-labels-idx1-ubyte", read_idx_labels)
    ]
    
    success = True
    for filename, reader_func in files_to_check:
        filepath = os.path.join(base_path, filename)
        
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filename}")
            success = False
            continue
            
        try:
            print(f"\n📁 {filename}:")
            reader_func(filepath)
            print(f"✅ Successfully read {filename}")
            
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")
            success = False
    
    return success

def main():
    """Main verification function"""
    print("Verifying downloaded IDX files...\n")
    
    datasets = ['mnist', 'fashion', 'kuzushiji_mnist']
    
    success_count = 0
    total_datasets = len(datasets)
    
    for dataset in datasets:
        if verify_dataset_idx_files(dataset):
            success_count += 1
    
    print(f"\n=== SUMMARY ===")
    print(f"Successfully verified: {success_count}/{total_datasets} datasets")
    
    if success_count == total_datasets:
        print("✅ All IDX files verified successfully!")
        print("\nFile Structure:")
        print("data/")
        for dataset in datasets:
            print(f"├── {dataset}/")
            print(f"│   └── raw/")
            print(f"│       ├── train-images-idx3-ubyte")
            print(f"│       ├── train-labels-idx1-ubyte")
            print(f"│       ├── t10k-images-idx3-ubyte")
            print(f"│       └── t10k-labels-idx1-ubyte")
    else:
        print(f"⚠️  {total_datasets - success_count} datasets had issues")

if __name__ == "__main__":
    main()
