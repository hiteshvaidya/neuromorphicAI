#!/usr/bin/env python3
"""
regenerate_pickle_files.py

Description: Regenerate all pickle files from raw IDX data using Sample class
Version: 1.0
Author: Auto-generated for neuromorphicAI project
"""

import numpy as np
import pickle as pkl
import os
import sys
from sklearn.model_selection import train_test_split

# Add som directory to path to import Sample class
sys.path.append('./som')
from sample import Sample

def read_idx_images(filename):
    """Read IDX format image files"""
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        data = f.read()
        images = np.frombuffer(data, dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
        return images.astype('float32') / 255.0  # Normalize to [0, 1]

def read_idx_labels(filename):
    """Read IDX format label files"""
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        data = f.read()
        labels = np.frombuffer(data, dtype=np.uint8)
        return labels

def generate_sample_objects(images, labels):
    """
    Generate Sample objects from images and labels
    
    :param images: normalized images (0.0-1.0)
    :type images: numpy 3D array
    :param labels: class labels
    :type labels: numpy 1D array
    :return: array of Sample objects
    :rtype: numpy array
    """
    samples = []
    for image, label in zip(images, labels):
        # Create Sample object with label, width, height, and image data
        # Ensure label is converted to native Python int, not numpy int
        python_label = int(label.item()) if hasattr(label, 'item') else int(label)
        sample = Sample(python_label, image.shape[1], image.shape[0], image)
        samples.append(sample)
    return np.asarray(samples, dtype=object)

def save_class_wise_samples(samples, labels, output_dir, split_name):
    """
    Save samples class-wise as pickle files
    
    :param samples: array of Sample objects
    :type samples: numpy array
    :param labels: class labels
    :type labels: numpy array
    :param output_dir: output directory path
    :type output_dir: str
    :param split_name: name of the split (train/test/valid)
    :type split_name: str
    """
    print(f"Saving {split_name} samples to {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Remove old pickle files
    for class_idx in range(10):
        old_file = os.path.join(output_dir, f"{class_idx}.pkl")
        if os.path.exists(old_file):
            os.remove(old_file)
            print(f"Removed old file: {old_file}")
    
    # Save class-wise samples
    for class_idx in range(10):
        # Find indices for current class
        class_indices = np.where(labels == class_idx)[0]
        
        if len(class_indices) == 0:
            print(f"Warning: No samples found for class {class_idx} in {split_name}")
            # Create empty file for consistency
            empty_samples = np.array([], dtype=object)
            pkl.dump(empty_samples, open(os.path.join(output_dir, f"{class_idx}.pkl"), 'wb'))
            continue
        
        # Get samples for current class
        class_samples = samples[class_indices]
        
        # Save to pickle file
        filename = os.path.join(output_dir, f"{class_idx}.pkl")
        pkl.dump(class_samples, open(filename, 'wb'))
        print(f"Saved {len(class_samples)} samples for class {class_idx} to {filename}")

def process_dataset(dataset_name, dataset_dir):
    """
    Process a single dataset and regenerate all pickle files
    
    :param dataset_name: name of the dataset
    :type dataset_name: str
    :param dataset_dir: path to dataset directory
    :type dataset_dir: str
    """
    print(f"\n=== Processing {dataset_name.upper()} ===")
    
    raw_dir = os.path.join(dataset_dir, 'raw')
    
    # Check if raw directory exists
    if not os.path.exists(raw_dir):
        print(f"❌ Raw directory not found: {raw_dir}")
        return False
    
    # Load training data
    train_images_file = os.path.join(raw_dir, 'train-images-idx3-ubyte')
    train_labels_file = os.path.join(raw_dir, 'train-labels-idx1-ubyte')
    
    if not os.path.exists(train_images_file) or not os.path.exists(train_labels_file):
        print(f"❌ Training files not found in {raw_dir}")
        return False
    
    print("Loading training data...")
    train_images = read_idx_images(train_images_file)
    train_labels = read_idx_labels(train_labels_file)
    
    print(f"Loaded {len(train_images)} training images and {len(train_labels)} training labels")
    
    # Load test data
    test_images_file = os.path.join(raw_dir, 't10k-images-idx3-ubyte')
    test_labels_file = os.path.join(raw_dir, 't10k-labels-idx1-ubyte')
    
    if not os.path.exists(test_images_file) or not os.path.exists(test_labels_file):
        print(f"❌ Test files not found in {raw_dir}")
        return False
    
    print("Loading test data...")
    test_images = read_idx_images(test_images_file)
    test_labels = read_idx_labels(test_labels_file)
    
    print(f"Loaded {len(test_images)} test images and {len(test_labels)} test labels")
    
    # Split training data into train and validation (80% train, 20% valid)
    print("Creating train/validation split...")
    train_imgs, valid_imgs, train_lbls, valid_lbls = train_test_split(
        train_images, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    
    print(f"Split: Train={len(train_imgs)}, Valid={len(valid_imgs)}, Test={len(test_images)}")
    
    # Generate Sample objects for each split
    print("Generating Sample objects...")
    train_samples = generate_sample_objects(train_imgs, train_lbls)
    valid_samples = generate_sample_objects(valid_imgs, valid_lbls)
    test_samples = generate_sample_objects(test_images, test_labels)
    
    # Save class-wise samples
    save_class_wise_samples(train_samples, train_lbls, 
                           os.path.join(dataset_dir, 'train'), 'train')
    save_class_wise_samples(valid_samples, valid_lbls, 
                           os.path.join(dataset_dir, 'valid'), 'valid')
    save_class_wise_samples(test_samples, test_labels, 
                           os.path.join(dataset_dir, 'test'), 'test')
    
    print(f"✅ {dataset_name.upper()} processing completed!")
    return True

def verify_sample_objects(dataset_name, dataset_dir):
    """
    Verify that the generated pickle files contain valid Sample objects
    
    :param dataset_name: name of the dataset
    :type dataset_name: str
    :param dataset_dir: path to dataset directory
    :type dataset_dir: str
    """
    print(f"\nVerifying {dataset_name} Sample objects...")
    
    splits = ['train', 'test', 'valid']
    
    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        total_samples = 0
        
        for class_idx in range(10):
            pickle_file = os.path.join(split_dir, f"{class_idx}.pkl")
            
            if not os.path.exists(pickle_file):
                print(f"❌ Missing file: {pickle_file}")
                continue
            
            try:
                # Load samples
                samples = pkl.load(open(pickle_file, 'rb'))
                
                if len(samples) > 0:
                    # Check first sample
                    first_sample = samples[0]
                    
                    if not isinstance(first_sample, Sample):
                        print(f"❌ Invalid object type in {pickle_file}: {type(first_sample)}")
                        continue
                    
                    # Verify sample properties
                    label = first_sample.getLabel()
                    image = first_sample.getImage()
                    shape = first_sample.getShape()
                    
                    # Check that label is a native Python int, not numpy int
                    if not isinstance(label, int) or isinstance(label, np.integer):
                        print(f"❌ Label type incorrect in {pickle_file}: expected <class 'int'>, got {type(label)}")
                        continue
                    
                    if label != class_idx:
                        print(f"❌ Label mismatch in {pickle_file}: expected {class_idx}, got {label}")
                        continue
                    
                    # Check image data
                    if hasattr(image, 'numpy'):
                        img_array = image.numpy()
                    else:
                        img_array = np.array(image)
                    
                    if img_array.min() < 0.0 or img_array.max() > 1.0:
                        print(f"❌ Image values out of range in {pickle_file}: min={img_array.min()}, max={img_array.max()}")
                        continue
                
                total_samples += len(samples)
                print(f"✅ {split}/{class_idx}.pkl: {len(samples)} valid Sample objects")
                
            except Exception as e:
                print(f"❌ Error loading {pickle_file}: {e}")
        
        print(f"  {split.capitalize()} total: {total_samples} samples")

def main():
    """Main function to regenerate all pickle files"""
    print("Regenerating pickle files from raw IDX data using Sample class...\n")
    
    # Dataset configurations
    datasets = {
        'mnist': './data/mnist',
        'fashion': './data/fashion', 
        'kuzushiji_mnist': './data/kuzushiji_mnist'
    }
    
    success_count = 0
    total_datasets = len(datasets)
    
    for dataset_name, dataset_dir in datasets.items():
        if process_dataset(dataset_name, dataset_dir):
            success_count += 1
            verify_sample_objects(dataset_name, dataset_dir)
    
    print(f"\n=== SUMMARY ===")
    print(f"Successfully processed: {success_count}/{total_datasets} datasets")
    
    if success_count == total_datasets:
        print("✅ All datasets regenerated successfully!")
        print("\nAll pickle files now contain Sample objects created from raw IDX data.")
        print("Each Sample object includes:")
        print("  - Normalized image data (0.0-1.0)")
        print("  - Correct class labels")
        print("  - Proper image dimensions (28x28)")
    else:
        print(f"⚠️  {total_datasets - success_count} datasets had issues")

if __name__ == "__main__":
    main()
