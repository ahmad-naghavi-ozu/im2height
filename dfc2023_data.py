"""
Contains dataset handling for DFC2023Amini dataset
"""

import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageFile
from albumentations import HorizontalFlip, VerticalFlip, Rotate, RandomRotate90
from augmenter import Augmenter

# Ensure PIL can handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class DFC2023Dataset(torch.utils.data.Dataset):
    '''
    A dataset class to handle the DFC2023Amini dataset structure
    
    This dataset class assumes the following structure:
    DFC2023Amini/
    ├── train/
    │   ├── dsm/ (elevation data - used as target)
    │   ├── rgb/ (RGB optical imagery - used as input)
    │   ├── sar/ (Synthetic Aperture Radar imagery - not used here)
    │   └── sem/ (Semantic segmentation masks - not used here)
    ├── valid/ (same structure)
    └── test/ (same structure)
    '''
    def __init__(self, dataset_root, split='train', input_type='rgb', target_type='dsm'):
        """
        Instantiate dataset for DFC2023Amini dataset.

        :param dataset_root: (str) root directory of DFC2023Amini dataset
        :param split: (str) one of 'train', 'valid', or 'test'
        :param input_type: (str) input modality, one of 'rgb' or 'sar'
        :param target_type: (str) target modality, typically 'dsm'
        """
        
        self.dataset_root = dataset_root
        self.split = split
        self.input_type = input_type
        self.target_type = target_type
        
        # Define input and target directories
        self.input_dir = os.path.join(dataset_root, split, input_type)
        self.target_dir = os.path.join(dataset_root, split, target_type)
        
        # Get lists of file names (assuming matching names between input and target)
        self.input_files = sorted([f for f in os.listdir(self.input_dir) if os.path.isfile(os.path.join(self.input_dir, f))])
        self.target_files = sorted([f for f in os.listdir(self.target_dir) if os.path.isfile(os.path.join(self.target_dir, f))])
        
        # Make sure we have matching files
        if len(self.input_files) != len(self.target_files):
            print(f"Warning: Number of input files ({len(self.input_files)}) doesn't match target files ({len(self.target_files)})")
        
        # Set up augmentations similar to NpyDataset
        transforms = [
            VerticalFlip(p=.2),
            HorizontalFlip(p=.2),
            RandomRotate90(p=.3)
        ]
        self.augmenter = Augmenter(list_of_transforms=transforms, p=.9 if split == 'train' else 0)
        
        # Create directories for NPY data (according to README.md structure)
        self.data_dir = os.path.join(os.path.dirname(dataset_root), "data")
        self.input_npy_dir = os.path.join(self.data_dir, split, "x")
        self.target_npy_dir = os.path.join(self.data_dir, split, "y")
        os.makedirs(self.input_npy_dir, exist_ok=True)
        os.makedirs(self.target_npy_dir, exist_ok=True)
        
        # Flag to avoid unnecessary conversions if files already exist
        self.convert_files = True

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx: int) -> tuple:
        # Load input image
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        file_basename = os.path.splitext(os.path.basename(input_path))[0]
        
        # Path for saving the converted npy file
        input_npy_path = os.path.join(self.input_npy_dir, f"{file_basename}.npy")
        
        # Check if converted file already exists to avoid redundant conversion
        if os.path.exists(input_npy_path):
            img = np.load(input_npy_path)
            if len(img.shape) == 2:  # If it's a 2D array, add channel dimension
                img = np.expand_dims(img, axis=2)
        # Handle different input types - images or numpy files
        elif input_path.endswith('.jpg') or input_path.endswith('.png') or input_path.endswith('.tif') or input_path.endswith('.tiff'):
            # Load image using PIL (handles jpg, png, tif)
            img = np.array(Image.open(input_path))
            
            # Ensure proper channel dimension
            if len(img.shape) == 2:  # If grayscale
                img = np.expand_dims(img, axis=2)
                
            # Save the numpy array with all channels preserved
            np.save(input_npy_path, img)
        else:  # Assume it's a numpy file
            img = np.load(input_path)
            if len(img.shape) == 2:  # If it's a 2D array, add channel dimension
                img = np.expand_dims(img, axis=2)
                
        # Load target/DSM data
        target_path = os.path.join(self.target_dir, self.target_files[idx])
        target_npy_path = os.path.join(self.target_npy_dir, f"{file_basename}.npy")
        
        # Check if converted target file already exists
        if os.path.exists(target_npy_path):
            label = np.load(target_npy_path)
            if len(label.shape) == 2:  # If it's a 2D array, add channel dimension
                label = np.expand_dims(label, axis=2)
        elif target_path.endswith('.jpg') or target_path.endswith('.png') or target_path.endswith('.tif') or target_path.endswith('.tiff'):
            label = np.array(Image.open(target_path))
            if len(label.shape) == 2:  # If it's a 2D array, add channel dimension
                label = np.expand_dims(label, axis=2)
            # Save the target as npy
            np.save(target_npy_path, label)
        else:  # Assume it's a numpy file
            label = np.load(target_path)
            if len(label.shape) == 2:  # If it's a 2D array, add channel dimension
                label = np.expand_dims(label, axis=2)
        
        # Normalize label similar to NpyDataset
        label = label - label.min()
        
        # Add padding if needed
        padding = 0
        img = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), "reflect")
        label = np.pad(label, ((padding, padding), (padding, padding), (0, 0)), "reflect")
        
        # Apply augmentations
        img, label = self.augmenter(img, label)
        
        # Convert to PyTorch tensors and ensure channel-first format
        img_tensor = torch.Tensor(img).permute((2, 0, 1))
        label_tensor = torch.Tensor(label).permute((2, 0, 1))
        
        return img_tensor, label_tensor


class DFC2023PredictionDataset(torch.utils.data.Dataset):
    '''
    A dataset class to handle prediction on DFC2023Amini data
    '''
    def __init__(self, dataset_root, split='test', input_type='rgb'):
        """
        Instantiate prediction dataset.

        :param dataset_root: (str) root directory of DFC2023Amini dataset
        :param split: (str) one of 'train', 'valid', or 'test' (typically 'test')
        :param input_type: (str) input modality, one of 'rgb' or 'sar'
        """
        
        self.dataset_root = dataset_root
        self.split = split
        self.input_type = input_type
        
        # Define input directory
        self.input_dir = os.path.join(dataset_root, split, input_type)
        
        # Get list of file paths
        self.input_files = sorted([os.path.join(self.input_dir, f) 
                                 for f in os.listdir(self.input_dir) 
                                 if os.path.isfile(os.path.join(self.input_dir, f))])
        
        # Set up the directory for NPY files matching the expected structure in README.md
        self.data_dir = os.path.join(os.path.dirname(dataset_root), "data")
        self.input_npy_dir = os.path.join(self.data_dir, split, "x")
        os.makedirs(self.input_npy_dir, exist_ok=True)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx: int) -> tuple:
        file_path = self.input_files[idx]
        file_basename = os.path.splitext(os.path.basename(file_path))[0]
        
        # Path for saving the converted npy file
        npy_path = os.path.join(self.input_npy_dir, f"{file_basename}.npy")
        
        # Check if converted file already exists
        if os.path.exists(npy_path):
            img = np.load(npy_path)
            if len(img.shape) == 2:  # If it's a 2D array, add channel dimension
                img = np.expand_dims(img, axis=2)
        # Load the input image if conversion needed
        elif file_path.endswith('.jpg') or file_path.endswith('.png') or file_path.endswith('.tif') or file_path.endswith('.tiff'):
            # Load image using PIL (handles jpg, png, tif)
            img = np.array(Image.open(file_path))
            
            # Ensure proper channel dimension
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2)
                
            # Save the numpy array with all channels preserved
            np.save(npy_path, img)
        else:  # Assume it's a numpy file
            img = np.load(file_path)
            if len(img.shape) == 2:  # If it's a 2D array, add channel dimension
                img = np.expand_dims(img, axis=2)
                
        # Add padding similar to NpyPredictionDataset
        padding = 3
        img = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), "reflect")
        
        # Convert to PyTorch tensor and ensure channel-first format
        img_tensor = torch.Tensor(img).permute((2, 0, 1))
        
        return file_path, img_tensor
