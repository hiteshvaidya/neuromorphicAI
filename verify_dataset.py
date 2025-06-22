#!/usr/bin/env python3
"""
verify_dataset.py

Description: Verify that the generated dataset files can be loaded correctly
"""

import pickle as pkl
import numpy as np
import sys
import os

sys.path.append('./som')
from sample import Sample

def verify_dataset(dataset_name, split, class_idx):
    """Verify a specific dataset file"""
    filepath = f'./data/{dataset_name}/{split}/{class_idx}.pkl'
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return False
    
    try:
        # Load the samples
        samples = pkl.load(open(filepath, 'rb'))
        
        if len(samples) == 0:
            print(f"⚠️  Empty file: {filepath}")
            return True
            
        # Test the first sample
        first_sample = samples[0]
        
        # Check if it's a Sample object
        if not isinstance(first_sample, Sample):
            print(f"❌ Invalid sample type in {filepath}: {type(first_sample)}")
            return False
            
        # Get sample properties
        label = first_sample.getLabel()
        image = first_sample.getImage()
        shape = first_sample.getShape()
        
        print(f"✅ {filepath}: {len(samples)} samples, label={label}, shape={shape}, image_type={type(image)}")
        
        # Verify image data
        if hasattr(image, 'numpy'):
            img_array = image.numpy()
        else:
            img_array = np.array(image)
            
        print(f"   Image stats: min={img_array.min():.3f}, max={img_array.max():.3f}, mean={img_array.mean():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return False

def main():
    """Main verification function"""
    print("Verifying generated dataset files...\n")
    
    datasets = ['mnist', 'fashion', 'kuzushiji_mnist']
    splits = ['train', 'test', 'valid']
    
    success_count = 0
    total_count = 0
    
    for dataset in datasets:
        print(f"\n=== {dataset.upper()} ===")
        
        for split in splits:
            print(f"\n{split.capitalize()}:")
            
            for class_idx in range(10):
                total_count += 1
                if verify_dataset(dataset, split, class_idx):
                    success_count += 1
    
    print(f"\n=== SUMMARY ===")
    print(f"Successfully verified: {success_count}/{total_count} files")
    
    if success_count == total_count:
        print("✅ All dataset files verified successfully!")
    else:
        print(f"⚠️  {total_count - success_count} files had issues")

if __name__ == "__main__":
    main()
