#!/usr/bin/env python3
"""
verify_label_types.py

Description: Verify that labels in Sample objects are stored as Python int, not numpy int
"""

import pickle as pkl
import sys
import numpy as np
sys.path.append('./som')
from sample import Sample

def verify_label_types():
    """Verify that labels are Python int type"""
    print("Verifying label types in Sample objects...\n")
    
    datasets = ['mnist', 'fashion', 'kuzushiji_mnist']
    splits = ['train', 'test', 'valid']
    
    all_good = True
    
    for dataset in datasets:
        print(f"=== {dataset.upper()} ===")
        
        for split in splits:
            print(f"  {split.capitalize()}:")
            
            for class_idx in range(3):  # Check first 3 classes as sample
                pickle_file = f"./data/{dataset}/{split}/{class_idx}.pkl"
                
                try:
                    # Load samples
                    samples = pkl.load(open(pickle_file, 'rb'))
                    
                    if len(samples) > 0:
                        # Check first sample's label type
                        first_sample = samples[0]
                        label = first_sample.getLabel()
                        label_type = type(label)
                        
                        if label_type == int and not isinstance(label, np.integer):
                            print(f"    ✅ Class {class_idx}: {label_type}")
                        else:
                            print(f"    ❌ Class {class_idx}: {label_type} (should be <class 'int'>)")
                            all_good = False
                    
                except Exception as e:
                    print(f"    ❌ Error loading {pickle_file}: {e}")
                    all_good = False
    
    print(f"\n{'='*50}")
    if all_good:
        print("✅ All labels are correctly stored as Python int!")
    else:
        print("❌ Some labels have incorrect types!")
    
    return all_good

if __name__ == "__main__":
    verify_label_types()
