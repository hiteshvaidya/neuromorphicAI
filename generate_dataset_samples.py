#!/usr/bin/env python3
"""
generate_dataset_samples.py

Description: Generate class-wise normalized data samples for MNIST, Fashion-MNIST, and Kuzushiji-MNIST
            Organizes data into train/test/valid splits and saves as pkl files
Version: 1.0
Author: Auto-generated for neuromorphicAI project
"""

import tensorflow as tf
import numpy as np
import pickle as pkl
import os
from sklearn.model_selection import train_test_split
import sys
sys.path.append('./som')
from sample import Sample

def create_directories():
    """Create directory structure for datasets"""
    datasets = ['mnist', 'fashion', 'kuzushiji_mnist']
    splits = ['train', 'test', 'valid']
    
    for dataset in datasets:
        for split in splits:
            dir_path = f'./data/{dataset}/{split}'
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created directory: {dir_path}")

def generate_samples(images, labels, shape_x, shape_y):
    """
    Generate a numpy array of Sample() objects from given images and labels
    
    :param images: images
    :type images: numpy 3D array
    :param labels: labels
    :type labels: numpy 1D array
    :param shape_x: width of image
    :type shape_x: int
    :param shape_y: height of image
    :type shape_y: int
    :return: samples
    :rtype: numpy 1D array
    """
    samples = []
    for image, label in zip(images, labels):
        samples.append(Sample(int(label), shape_x, shape_y, image))
    return np.asarray(samples, dtype=object)

def dump_split_data_classwise(images, labels, n_classes, path, split_name):
    """
    Class-wise split the data and dump pickle files
    
    :param images: normalized images
    :type images: numpy 3D array
    :param labels: image labels
    :type labels: numpy 1D array
    :param n_classes: number of classes
    :type n_classes: int
    :param path: directory path
    :type path: str
    :param split_name: name of the split (train/test/valid)
    :type split_name: str
    """
    print(f"Processing {split_name} split with {len(images)} samples...")
    
    for c in range(n_classes):
        indexes = np.where(labels == c)[0]
        if len(indexes) == 0:
            print(f"Warning: No samples found for class {c} in {split_name}")
            continue
            
        class_images = images[indexes]
        class_labels = labels[indexes]
        
        samples = generate_samples(class_images, class_labels, 
                                 images.shape[1], images.shape[2])
        
        filename = os.path.join(path, f"{c}.pkl")
        pkl.dump(samples, open(filename, 'wb'))
        print(f"Saved {len(samples)} samples for class {c} in {filename}")

def process_mnist():
    """Process MNIST dataset"""
    print("Processing MNIST dataset...")
    
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Create validation split from training data (80% train, 20% valid)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"MNIST - Train: {x_train.shape}, Valid: {x_valid.shape}, Test: {x_test.shape}")
    
    # Save class-wise data
    dump_split_data_classwise(x_train, y_train, 10, './data/mnist/train', 'train')
    dump_split_data_classwise(x_valid, y_valid, 10, './data/mnist/valid', 'valid')
    dump_split_data_classwise(x_test, y_test, 10, './data/mnist/test', 'test')
    
    print("MNIST processing completed!\n")

def process_fashion():
    """Process Fashion-MNIST dataset"""
    print("Processing Fashion-MNIST dataset...")
    
    # Load Fashion-MNIST data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion.load_data()
    
    # Normalize to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Create validation split from training data (80% train, 20% valid)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Fashion-MNIST - Train: {x_train.shape}, Valid: {x_valid.shape}, Test: {x_test.shape}")
    
    # Save class-wise data
    dump_split_data_classwise(x_train, y_train, 10, './data/fashion/train', 'train')
    dump_split_data_classwise(x_valid, y_valid, 10, './data/fashion/valid', 'valid')
    dump_split_data_classwise(x_test, y_test, 10, './data/fashion/test', 'test')
    
    print("Fashion-MNIST processing completed!\n")

def download_kuzushiji_mnist():
    """Download and process Kuzushiji-MNIST dataset"""
    print("Processing Kuzushiji-MNIST dataset...")
    
    try:
        # Try to import the dataset (requires internet connection)
        import urllib.request
        import gzip
        import os
        
        # URLs for Kuzushiji-MNIST
        urls = {
            'train_images': 'http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz',
            'train_labels': 'http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz',
            'test_images': 'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz',
            'test_labels': 'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz'
        }
        
        # Create temporary directory for downloaded files
        temp_dir = './temp_kmnist'
        os.makedirs(temp_dir, exist_ok=True)
        
        # Download files
        for name, url in urls.items():
            print(f"Downloading {name}...")
            filename = os.path.join(temp_dir, f"{name}.gz")
            urllib.request.urlretrieve(url, filename)
            
            # Extract the gz file
            with gzip.open(filename, 'rb') as f_in:
                with open(filename[:-3], 'wb') as f_out:
                    f_out.write(f_in.read())
        
        # Load data using idx2numpy (if available) or manual parsing
        try:
            import idx2numpy
            
            train_images = idx2numpy.convert_from_file(os.path.join(temp_dir, 'train_images'))
            train_labels = idx2numpy.convert_from_file(os.path.join(temp_dir, 'train_labels'))
            test_images = idx2numpy.convert_from_file(os.path.join(temp_dir, 'test_images'))
            test_labels = idx2numpy.convert_from_file(os.path.join(temp_dir, 'test_labels'))
            
        except ImportError:
            print("idx2numpy not available. Using manual parsing...")
            # Manual IDX file parsing
            def read_idx_images(filename):
                with open(filename, 'rb') as f:
                    magic = int.from_bytes(f.read(4), 'big')
                    num_images = int.from_bytes(f.read(4), 'big')
                    rows = int.from_bytes(f.read(4), 'big')
                    cols = int.from_bytes(f.read(4), 'big')
                    data = f.read()
                    images = np.frombuffer(data, dtype=np.uint8)
                    images = images.reshape(num_images, rows, cols)
                    return images
            
            def read_idx_labels(filename):
                with open(filename, 'rb') as f:
                    magic = int.from_bytes(f.read(4), 'big')
                    num_labels = int.from_bytes(f.read(4), 'big')
                    data = f.read()
                    labels = np.frombuffer(data, dtype=np.uint8)
                    return labels
            
            train_images = read_idx_images(os.path.join(temp_dir, 'train_images'))
            train_labels = read_idx_labels(os.path.join(temp_dir, 'train_labels'))
            test_images = read_idx_images(os.path.join(temp_dir, 'test_images'))
            test_labels = read_idx_labels(os.path.join(temp_dir, 'test_labels'))
        
        # Normalize to [0, 1]
        x_train = train_images.astype('float32') / 255.0
        x_test = test_images.astype('float32') / 255.0
        y_train = train_labels
        y_test = test_labels
        
        # Create validation split from training data (80% train, 20% valid)
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print(f"Kuzushiji-MNIST - Train: {x_train.shape}, Valid: {x_valid.shape}, Test: {x_test.shape}")
        
        # Save class-wise data
        dump_split_data_classwise(x_train, y_train, 10, './data/kuzushiji_mnist/train', 'train')
        dump_split_data_classwise(x_valid, y_valid, 10, './data/kuzushiji_mnist/valid', 'valid')
        dump_split_data_classwise(x_test, y_test, 10, './data/kuzushiji_mnist/test', 'test')
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir)
        
        print("Kuzushiji-MNIST processing completed!\n")
        
    except Exception as e:
        print(f"Error processing Kuzushiji-MNIST: {e}")
        print("Creating placeholder files for Kuzushiji-MNIST...")
        
        # Create placeholder files if download fails
        for class_idx in range(10):
            for split in ['train', 'test', 'valid']:
                # Create empty sample arrays as placeholders
                empty_samples = np.array([], dtype=object)
                filename = f'./data/kuzushiji_mnist/{split}/{class_idx}.pkl'
                pkl.dump(empty_samples, open(filename, 'wb'))
        
        print("Placeholder files created for Kuzushiji-MNIST")

def process_kuzushiji_mnist_alternative():
    """Alternative method using TensorFlow Datasets if available"""
    try:
        import tensorflow_datasets as tfds
        
        print("Using TensorFlow Datasets for Kuzushiji-MNIST...")
        
        # Load the dataset
        ds_train, ds_test = tfds.load('kmnist', split=['train', 'test'], as_supervised=True)
        
        # Convert to numpy arrays
        train_images = []
        train_labels = []
        for image, label in ds_train:
            train_images.append(image.numpy())
            train_labels.append(label.numpy())
        
        test_images = []
        test_labels = []
        for image, label in ds_test:
            test_images.append(image.numpy())
            test_labels.append(label.numpy())
        
        x_train = np.array(train_images).squeeze()
        y_train = np.array(train_labels)
        x_test = np.array(test_images).squeeze()
        y_test = np.array(test_labels)
        
        # Normalize to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Create validation split
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print(f"Kuzushiji-MNIST (tfds) - Train: {x_train.shape}, Valid: {x_valid.shape}, Test: {x_test.shape}")
        
        # Save class-wise data
        dump_split_data_classwise(x_train, y_train, 10, './data/kuzushiji_mnist/train', 'train')
        dump_split_data_classwise(x_valid, y_valid, 10, './data/kuzushiji_mnist/valid', 'valid')
        dump_split_data_classwise(x_test, y_test, 10, './data/kuzushiji_mnist/test', 'test')
        
        print("Kuzushiji-MNIST (tfds) processing completed!\n")
        return True
        
    except ImportError:
        print("TensorFlow Datasets not available for Kuzushiji-MNIST")
        return False
    except Exception as e:
        print(f"Error with TensorFlow Datasets: {e}")
        return False

def verify_data_structure():
    """Verify that all data files were created successfully"""
    print("Verifying data structure...")
    
    datasets = ['mnist', 'fashion', 'kuzushiji_mnist']
    splits = ['train', 'test', 'valid']
    
    for dataset in datasets:
        print(f"\n{dataset.upper()}:")
        for split in splits:
            print(f"  {split}:")
            total_samples = 0
            for class_idx in range(10):
                filename = f'./data/{dataset}/{split}/{class_idx}.pkl'
                if os.path.exists(filename):
                    try:
                        samples = pkl.load(open(filename, 'rb'))
                        total_samples += len(samples)
                        print(f"    Class {class_idx}: {len(samples)} samples")
                    except Exception as e:
                        print(f"    Class {class_idx}: Error loading - {e}")
                else:
                    print(f"    Class {class_idx}: File not found")
            print(f"  Total {split} samples: {total_samples}")

def main():
    """Main function to process all datasets"""
    print("Starting dataset processing...\n")
    
    # Create directory structure
    create_directories()
    
    # Process each dataset
    process_mnist()
    process_fashion()
    
    # Try alternative methods for Kuzushiji-MNIST
    if not process_kuzushiji_mnist_alternative():
        download_kuzushiji_mnist()
    
    # Verify the data structure
    verify_data_structure()
    
    print("\nDataset processing completed!")
    print("All datasets have been organized into:")
    print("  - data/mnist/{train,test,valid}/")
    print("  - data/fashion/{train,test,valid}/")
    print("  - data/kuzushiji_mnist/{train,test,valid}/")
    print("Each folder contains class-wise pkl files (0.pkl to 9.pkl)")

if __name__ == "__main__":
    main()
