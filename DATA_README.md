# Dataset Organization

This document describes the organization of the MNIST, Fashion-MNIST, and Kuzushiji-MNIST datasets that have been processed and stored in this project.

## Directory Structure

```
data/
├── mnist/
│   ├── raw/
│   │   ├── train-images-idx3-ubyte  (60,000 training images)
│   │   ├── train-labels-idx1-ubyte  (60,000 training labels)
│   │   ├── t10k-images-idx3-ubyte   (10,000 test images)
│   │   └── t10k-labels-idx1-ubyte   (10,000 test labels)
│   ├── train/
│   │   ├── 0.pkl  (4,738 samples)
│   │   ├── 1.pkl  (5,394 samples)
│   │   ├── 2.pkl  (4,766 samples)
│   │   ├── 3.pkl  (4,905 samples)
│   │   ├── 4.pkl  (4,674 samples)
│   │   ├── 5.pkl  (4,337 samples)
│   │   ├── 6.pkl  (4,734 samples)
│   │   ├── 7.pkl  (5,012 samples)
│   │   ├── 8.pkl  (4,681 samples)
│   │   └── 9.pkl  (4,759 samples)
│   ├── test/
│   │   ├── 0.pkl  (980 samples)
│   │   ├── 1.pkl  (1,135 samples)
│   │   ├── 2.pkl  (1,032 samples)
│   │   ├── 3.pkl  (1,010 samples)
│   │   ├── 4.pkl  (982 samples)
│   │   ├── 5.pkl  (892 samples)
│   │   ├── 6.pkl  (958 samples)
│   │   ├── 7.pkl  (1,028 samples)
│   │   ├── 8.pkl  (974 samples)
│   │   └── 9.pkl  (1,009 samples)
│   └── valid/
│       ├── 0.pkl  (1,185 samples)
│       ├── 1.pkl  (1,348 samples)
│       ├── 2.pkl  (1,192 samples)
│       ├── 3.pkl  (1,226 samples)
│       ├── 4.pkl  (1,168 samples)
│       ├── 5.pkl  (1,084 samples)
│       ├── 6.pkl  (1,184 samples)
│       ├── 7.pkl  (1,253 samples)
│       ├── 8.pkl  (1,170 samples)
│       └── 9.pkl  (1,190 samples)
├── fashion_mnist/
│   ├── raw/
│   │   ├── train-images-idx3-ubyte  (60,000 training images)
│   │   ├── train-labels-idx1-ubyte  (60,000 training labels)
│   │   ├── t10k-images-idx3-ubyte   (10,000 test images)
│   │   └── t10k-labels-idx1-ubyte   (10,000 test labels)
│   ├── train/     (4,800 samples per class)
│   ├── test/      (1,000 samples per class)
│   └── valid/     (1,200 samples per class)
└── kuzushiji_mnist/
    ├── raw/
    │   ├── train-images-idx3-ubyte  (60,000 training images)
    │   ├── train-labels-idx1-ubyte  (60,000 training labels)
    │   ├── t10k-images-idx3-ubyte   (10,000 test images)
    │   └── t10k-labels-idx1-ubyte   (10,000 test labels)
    ├── train/     (4,800 samples per class)
    ├── test/      (1,000 samples per class)
    └── valid/     (1,200 samples per class)
```

## Data Format

### Sample Objects
Each `.pkl` file contains an array of `Sample` objects. Each `Sample` object has the following properties:
- **Label**: Class identifier (0-9)
- **Shape**: Image dimensions (28x28 for all datasets)
- **Image**: Normalized pixel values (0.0 to 1.0)

### Data Normalization
- All images are normalized to the range [0.0, 1.0] by dividing pixel values by 255.0
- Original pixel values were in the range [0, 255]

### Class Distribution
- **MNIST**: Natural distribution from original dataset (varies by class)
- **Fashion-MNIST**: Balanced distribution (equal samples per class)
- **Kuzushiji-MNIST**: Balanced distribution (equal samples per class)

## Dataset Splits

### Training/Validation Split
- The original training data was split into 80% training and 20% validation
- Random stratified sampling was used to maintain class balance
- Random seed: 42 (for reproducibility)

### Split Sizes
| Dataset | Train | Validation | Test | Total |
|---------|--------|------------|------|-------|
| MNIST | 48,000 | 12,000 | 10,000 | 70,000 |
| Fashion-MNIST | 48,000 | 12,000 | 10,000 | 70,000 |
| Kuzushiji-MNIST | 48,000 | 12,000 | 10,000 | 70,000 |

## Usage

### Loading Data
```python
import pickle as pkl
import sys
sys.path.append('./som')
from sample import Sample

# Load class-specific data
samples = pkl.load(open('./data/mnist/train/0.pkl', 'rb'))

# Access sample properties
for sample in samples:
    label = sample.getLabel()
    image = sample.getImage()  # Returns TensorFlow tensor
    shape = sample.getShape()  # Returns (28, 28)
```

### Loading All Classes
```python
def load_dataset(dataset_name, split):
    """Load all classes for a dataset split"""
    all_samples = []
    for class_idx in range(10):
        filepath = f'./data/{dataset_name}/{split}/{class_idx}.pkl'
        class_samples = pkl.load(open(filepath, 'rb'))
        all_samples.extend(class_samples)
    return all_samples

# Example usage
train_samples = load_dataset('mnist', 'train')
test_samples = load_dataset('fashion_mnist', 'test')
```

## Dataset Sources

- **MNIST**: Original handwritten digits dataset
  - Raw IDX files from TensorFlow/Keras storage
  - URLs: https://storage.googleapis.com/tensorflow/tf-keras-datasets/
- **Fashion-MNIST**: Fashion items dataset
  - Raw IDX files from Fashion-MNIST S3 bucket
  - URLs: http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/
- **Kuzushiji-MNIST**: Japanese Kuzushiji characters
  - Raw IDX files from CODH (Center for Open Data in the Humanities)
  - URLs: http://codh.rois.ac.jp/kmnist/dataset/kmnist/

## Raw IDX Files

Each dataset includes the original IDX format files:
- `train-images-idx3-ubyte`: Training images (60,000 samples, 28×28 pixels)
- `train-labels-idx1-ubyte`: Training labels (60,000 labels, 0-9)
- `t10k-images-idx3-ubyte`: Test images (10,000 samples, 28×28 pixels)  
- `t10k-labels-idx1-ubyte`: Test labels (10,000 labels, 0-9)

### IDX File Format
- **Magic Numbers**: 2051 (images), 2049 (labels)
- **Image Format**: Unsigned byte (0-255)
- **Label Format**: Unsigned byte (0-9)
- **Byte Order**: Big-endian

## Files Generated

- `generate_dataset_samples.py`: Main script for processing and organizing datasets
- `verify_dataset.py`: Verification script to check data integrity of pkl files
- `verify_idx_files.py`: Verification script to check raw IDX files
- Individual `.pkl` files: Class-wise data samples
- Raw IDX files: Original binary format datasets

## Verification

All datasets have been verified to ensure:
- ✅ Correct file structure
- ✅ Proper Sample object format
- ✅ Normalized pixel values (0.0-1.0)
- ✅ Correct class labels
- ✅ Expected sample counts per class
- ✅ Valid image dimensions (28x28)

Run `python verify_dataset.py` to re-verify the data integrity.
Run `python verify_idx_files.py` to verify the raw IDX files.
