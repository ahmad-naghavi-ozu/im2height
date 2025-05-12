import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

def preprocess_dataset(dataset_path, output_path, input_type="rgb", target_type="dsm"):
    """
    Preprocess the DFC2023Amini dataset by converting images to .npy format
    
    Args:
        dataset_path: Path to the DFC2023Amini dataset
        output_path: Path to save the processed .npy files
        input_type: Input data type ('rgb' or 'sar')
        target_type: Target data type (usually 'dsm')
    """
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        print(f"Processing {split} split...")
        
        # Create output directories
        input_output_dir = os.path.join(output_path, split, 'x')
        target_output_dir = os.path.join(output_path, split, 'y')
        
        os.makedirs(input_output_dir, exist_ok=True)
        os.makedirs(target_output_dir, exist_ok=True)
        
        # Source directories
        input_dir = os.path.join(dataset_path, split, input_type)
        target_dir = os.path.join(dataset_path, split, target_type)
        
        # Get input and target files
        input_files = sorted([f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))])
        target_files = sorted([f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))])
        
        # Process input files
        for file_name in tqdm(input_files, desc=f"Converting {input_type} to npy"):
            input_path = os.path.join(input_dir, file_name)
            output_npy_path = os.path.join(input_output_dir, f"{os.path.splitext(file_name)[0]}.npy")
            
            # Skip if already processed
            if os.path.exists(output_npy_path):
                continue
            
            if input_path.endswith(('.jpg', '.png', '.tif', '.tiff')):
                img = np.array(Image.open(input_path))
                # Take first channel if RGB, preserving dimensions
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = img[:, :, 0:1]
                elif len(img.shape) == 2:  # Add channel dimension if grayscale
                    img = np.expand_dims(img, axis=2)
            else:  # Already numpy file
                img = np.load(input_path)
                if len(img.shape) == 2:
                    img = np.expand_dims(img, axis=2)
            
            # Move channel dimension to first position as expected by PyTorch
            img = np.transpose(img, (2, 0, 1))
            np.save(output_npy_path, img)
        
        # Process target files
        for file_name in tqdm(target_files, desc=f"Converting {target_type} to npy"):
            target_path = os.path.join(target_dir, file_name)
            output_npy_path = os.path.join(target_output_dir, f"{os.path.splitext(file_name)[0]}.npy")
            
            # Skip if already processed
            if os.path.exists(output_npy_path):
                continue
                
            if target_path.endswith(('.jpg', '.png', '.tif', '.tiff')):
                img = np.array(Image.open(target_path))
                if len(img.shape) == 2:
                    img = np.expand_dims(img, axis=2)
            else:  # Already numpy file
                img = np.load(target_path)
                if len(img.shape) == 2:
                    img = np.expand_dims(img, axis=2)
            
            # Move channel dimension to first position as expected by PyTorch
            img = np.transpose(img, (2, 0, 1))
            np.save(output_npy_path, img)
        
        print(f"Finished processing {split} split")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess DFC2023Amini dataset for Im2Height model")
    parser.add_argument("-d", "--dataset_path", type=str, required=True,
                        help="Path to the DFC2023Amini dataset")
    parser.add_argument("-o", "--output_path", type=str, required=True,
                        help="Path to save the processed .npy files")
    parser.add_argument("-i", "--input_type", type=str, default="rgb", choices=["rgb", "sar"],
                        help="Input data type")
    parser.add_argument("-t", "--target_type", type=str, default="dsm",
                        help="Target data type")
    
    args = parser.parse_args()
    preprocess_dataset(**vars(args))
