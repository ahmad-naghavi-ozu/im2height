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
    A dataset class to handle the DFC2023 dataset structure (DFC2023S, DFC2023A, DFC2023Asmall, DFC2023Amini)
    
    This dataset class assumes the following structure:
    DFC2023<VariantName>/
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
        Instantiate dataset for DFC2023 dataset variants (DFC2023S, DFC2023A, DFC2023Asmall, DFC2023Amini).

        :param dataset_root: (str) root directory of DFC2023 dataset variant
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
        
        # Extract dataset name from the path (last folder name)
        dataset_name = os.path.basename(os.path.normpath(dataset_root))
        
        # Create directories for NPY data (according to README.md structure)
        # Use the IM2HEIGHT root directory for the NPY files, not relative to dataset_root
        # Store data in dataset-specific subfolder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(script_dir, "data", dataset_name)
        self.input_npy_dir = os.path.join(self.data_dir, split, "x")
        self.target_npy_dir = os.path.join(self.data_dir, split, "y")
        os.makedirs(self.input_npy_dir, exist_ok=True)
        os.makedirs(self.target_npy_dir, exist_ok=True)
        
        # Flag to avoid unnecessary conversions if files already exist
        self.convert_files = True
        # Log where files will be saved
        # print(f"NPY files will be saved to: {self.data_dir}")

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
        
        # Ensure channels are in the last dimension for augmentation (channels_last)
        if img.shape[0] == 1 or img.shape[0] == 3:  # If channels are first (1 or 3, C×H×W)
            img = np.transpose(img, (1, 2, 0))  # Convert to H×W×C
        if label.shape[0] == 1:  # If channels are first
            label = np.transpose(label, (1, 2, 0))  # Convert to H×W×C
            
        # Apply augmentations (albumentations expects channels_last)
        img, label = self.augmenter(img, label)
        
        # Make copies to ensure contiguous memory (fixes negative stride issues)
        img = np.ascontiguousarray(img)
        label = np.ascontiguousarray(label)
        
        # Convert back to PyTorch tensors with channel-first format
        img_tensor = torch.Tensor(img).permute((2, 0, 1))  # H×W×C -> C×H×W
        label_tensor = torch.Tensor(label).permute((2, 0, 1))  # H×W×C -> C×H×W
        
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
        
        # Extract dataset name from the path (last folder name)
        dataset_name = os.path.basename(os.path.normpath(dataset_root))
        
        # Set up the directory for NPY files matching the expected structure in README.md
        # Use the IM2HEIGHT root directory for the NPY files, in dataset-specific subfolder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(script_dir, "data", dataset_name)
        self.input_npy_dir = os.path.join(self.data_dir, split, "x")
        os.makedirs(self.input_npy_dir, exist_ok=True)
        # Comment out the print statement to keep the terminal clean
        # print(f"NPY files will be saved to: {self.data_dir}")

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx: int) -> tuple:
        file_path = self.input_files[idx]
        file_basename = os.path.splitext(os.path.basename(file_path))[0]
        
        # Path for saving the converted npy file
        npy_path = os.path.join(self.input_npy_dir, f"{file_basename}.npy")
        
        # If the NPY file exists, but is corrupted or has wrong dimensions, 
        # let's remove it and regenerate it
        if os.path.exists(npy_path):
            try:
                img = np.load(npy_path)
                
                # Check if the file has reasonable dimensions for an image
                if len(img.shape) >= 2:
                    # If any dimension is suspiciously small (like 3 for channels mistakenly used as height)
                    # or suspiciously large (>1000 for height/width, which could be wrong dimension order)
                    if (len(img.shape) == 3 and 
                        (img.shape[0] > 10 or img.shape[1] < 10 or img.shape[2] < 10)):
                        print(f"Suspicious NPY file dimensions: {img.shape}, regenerating...")
                        os.remove(npy_path)
                        # Set img to None so we regenerate below
                        img = None
                    # For 2D arrays, check reasonable image dimensions
                    elif len(img.shape) == 2 and (img.shape[0] < 10 or img.shape[1] < 10):
                        print(f"Suspicious NPY file dimensions: {img.shape}, regenerating...")
                        os.remove(npy_path)
                        img = None
            except Exception as e:
                print(f"Error loading NPY file {npy_path}: {e}")
                os.remove(npy_path)
                img = None
        else:
            img = None
            
        # If we need to generate the image data
        if img is None:
            # Load the input image if not already cached
            if file_path.endswith('.jpg') or file_path.endswith('.png') or file_path.endswith('.tif') or file_path.endswith('.tiff'):
                # Load image using PIL (handles jpg, png, tif)
                img = np.array(Image.open(file_path))
                
                # Ensure proper channel dimension (in channel-first format)
                if len(img.shape) == 2:
                    img = np.expand_dims(img, axis=0)  # Add as first dimension (channel-first)
                elif len(img.shape) == 3:
                    img = np.transpose(img, (2, 0, 1))  # H×W×C -> C×H×W
                    
                # Save the numpy array with all channels preserved
                np.save(npy_path, img)
            else:  # Assume it's a numpy file
                img = np.load(file_path)
                if len(img.shape) == 2:  # If it's a 2D array, add channel dimension
                    img = np.expand_dims(img, axis=0)  # Add as first dimension (channel-first)
                elif len(img.shape) == 3 and img.shape[2] <= 4:
                    # Convert from channels-last to channels-first
                    img = np.transpose(img, (2, 0, 1))  # H×W×C -> C×H×W
                
        # For RGB images, ensure we use only the standard 3 channels
        if self.input_type == 'rgb' and img.shape[0] > 3:
            print(f"Limiting RGB input from {img.shape[0]} to 3 channels")
            img = img[:3]  # Keep only first 3 channels for RGB
        
        # For grayscale (SAR), ensure we use only 1 channel
        if self.input_type == 'sar' and img.shape[0] > 1:
            print(f"Limiting SAR input from {img.shape[0]} to 1 channel")
            img = img[:1]  # Keep only first channel
        
        # Add padding - matching the training dataset's padding value
        padding = 0  # Changed from 3 to 0 to match DFC2023Dataset
        img = np.pad(img, ((0, 0), (padding, padding), (padding, padding)), "reflect")
        
        # Ensure we're working with contiguous array (fix any negative stride issues)
        img = np.ascontiguousarray(img)
        
        # Convert to PyTorch tensor - already in channel-first format
        img_tensor = torch.Tensor(img)
        
        # Final sanity check on channel dimension
        expected_channels = 3 if self.input_type == 'rgb' else 1
        if img_tensor.shape[0] != expected_channels:
            print(f"Warning: Tensor has {img_tensor.shape[0]} channels but expected {expected_channels}")
            print(f"Full tensor shape: {img_tensor.shape}")
            # Force the correct number of channels based on input_type
            if self.input_type == 'rgb':
                # Take first 3 channels or expand if needed
                if img_tensor.shape[0] > 3:
                    img_tensor = img_tensor[:3]
                else:
                    # If we have 1 channel but need 3, duplicate it
                    img_tensor = img_tensor.expand(3, -1, -1)  
            else:  # SAR (grayscale)
                if img_tensor.shape[0] > 1:
                    img_tensor = img_tensor[:1]
                    
        # Verify final tensor shape is appropriate for the model
        print(f"Final tensor shape: {img_tensor.shape}")
        
        return file_path, img_tensor
