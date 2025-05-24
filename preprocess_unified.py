#!/usr/bin/env python
"""
Unified preprocessing script for Im2Height model that handles multiple dataset formats.

This script can:
1. Convert image datasets to NPY format for faster loading
2. Handle already processed NPY datasets (no-op)
3. Support various input types (rgb, sar, etc.) and target types (dsm, etc.)
4. Auto-detect dataset format and structure
5. Maintain backward compatibility with legacy NPY structure
"""

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


def detect_dataset_format(dataset_path, split='train', input_type='rgb', target_type='dsm'):
    """
    Detect if dataset is in NPY format or image format.
    
    Returns:
        'npy': Dataset already in NPY format (data/<dataset>/train/x, data/<dataset>/train/y)
        'image': Dataset in image format (dataset/train/rgb, dataset/train/dsm)
    """
    # Check for NPY format first (processed format)
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    npy_data_dir = os.path.join(script_dir, "data", dataset_name, split)
    
    if (os.path.exists(os.path.join(npy_data_dir, 'x')) and 
        os.path.exists(os.path.join(npy_data_dir, 'y'))):
        x_files = os.listdir(os.path.join(npy_data_dir, 'x'))
        y_files = os.listdir(os.path.join(npy_data_dir, 'y'))
        if len(x_files) > 0 and len(y_files) > 0:
            return 'npy'
    
    # Check for image format (raw format)
    image_input_dir = os.path.join(dataset_path, split, input_type)
    image_target_dir = os.path.join(dataset_path, split, target_type)
    
    if (os.path.exists(image_input_dir) and os.path.exists(image_target_dir)):
        input_files = [f for f in os.listdir(image_input_dir) 
                      if os.path.isfile(os.path.join(image_input_dir, f))]
        target_files = [f for f in os.listdir(image_target_dir) 
                       if os.path.isfile(os.path.join(image_target_dir, f))]
        if len(input_files) > 0 and len(target_files) > 0:
            return 'image'
    
    return 'unknown'


def preprocess_dataset(dataset_path, output_path=None, input_type="rgb", target_type="dsm", 
                      force_reprocess=False, verbose=True):
    """
    Unified preprocessing function that handles both image and NPY datasets.
    
    Args:
        dataset_path: Path to the dataset
        output_path: Path to save the processed .npy files. If None, uses default location
        input_type: Input data type ('rgb', 'sar', etc.)
        target_type: Target data type ('dsm', etc.)
        force_reprocess: If True, reprocess even if NPY files already exist
        verbose: If True, print detailed progress information
    """
    # Extract dataset name from the path
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    
    # Set default output path
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", dataset_name)
        if verbose:
            print(f"No output path specified. Using default path: {output_path}")
    
    # Check available splits
    splits = []
    for split in ['train', 'valid', 'test']:
        split_format = detect_dataset_format(dataset_path, split, input_type, target_type)
        if split_format != 'unknown':
            splits.append((split, split_format))
    
    if not splits:
        print(f"Error: No valid dataset splits found in {dataset_path}")
        print(f"Expected structure: {dataset_path}/[train|valid|test]/{input_type}/")
        print(f"                    {dataset_path}/[train|valid|test]/{target_type}/")
        return False
    
    if verbose:
        print(f"Dataset: {dataset_name}")
        print(f"Input type: {input_type}, Target type: {target_type}")
        print(f"Found splits: {[f'{split} ({fmt})' for split, fmt in splits]}")
    
    for split, split_format in splits:
        if verbose:
            print(f"\nProcessing {split} split (format: {split_format})...")
        
        if split_format == 'npy':
            if not force_reprocess:
                if verbose:
                    print(f"Split {split} already in NPY format, skipping...")
                continue
            else:
                if verbose:
                    print(f"Force reprocessing NPY format for split {split}...")
        
        # Create output directories
        input_output_dir = os.path.join(output_path, split, 'x')
        target_output_dir = os.path.join(output_path, split, 'y')
        
        os.makedirs(input_output_dir, exist_ok=True)
        os.makedirs(target_output_dir, exist_ok=True)
        
        # Source directories
        input_dir = os.path.join(dataset_path, split, input_type)
        target_dir = os.path.join(dataset_path, split, target_type)
        
        # Get input and target files
        input_files = sorted([f for f in os.listdir(input_dir) 
                             if os.path.isfile(os.path.join(input_dir, f))])
        target_files = sorted([f for f in os.listdir(target_dir) 
                              if os.path.isfile(os.path.join(target_dir, f))])
        
        if verbose:
            print(f"Found {len(input_files)} input files and {len(target_files)} target files")
        
        # Process input files
        processed_count = 0
        for file_name in tqdm(input_files, desc=f"Converting {input_type} to npy", disable=not verbose):
            input_path = os.path.join(input_dir, file_name)
            output_npy_path = os.path.join(input_output_dir, f"{os.path.splitext(file_name)[0]}.npy")
            
            # Skip if already processed and not forcing reprocess
            if os.path.exists(output_npy_path) and not force_reprocess:
                continue
            
            processed_count += 1
            img = load_and_convert_image(input_path)
            
            # Move channel dimension to first position as expected by PyTorch
            img = np.transpose(img, (2, 0, 1))
            np.save(output_npy_path, img)
        
        # Process target files
        for file_name in tqdm(target_files, desc=f"Converting {target_type} to npy", disable=not verbose):
            target_path = os.path.join(target_dir, file_name)
            output_npy_path = os.path.join(target_output_dir, f"{os.path.splitext(file_name)[0]}.npy")
            
            # Skip if already processed and not forcing reprocess
            if os.path.exists(output_npy_path) and not force_reprocess:
                continue
            
            img = load_and_convert_image(target_path)
            
            # Move channel dimension to first position as expected by PyTorch
            img = np.transpose(img, (2, 0, 1))
            np.save(output_npy_path, img)
        
        if verbose:
            if processed_count == 0:
                print(f"All files already processed for {split} split")
            else:
                print(f"Finished processing {split} split ({processed_count} new files)")
    
    if verbose:
        print(f"\nPreprocessing complete! NPY files saved to: {output_path}")
    
    return True


def load_and_convert_image(file_path):
    """
    Load an image or numpy file and convert to consistent format.
    
    Returns:
        numpy array with shape (H, W, C) where C >= 1
    """
    if file_path.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
        # Load image using PIL
        img = np.array(Image.open(file_path))
        
        # Handle different image formats
        if len(img.shape) == 3:
            # Multi-channel image (RGB, etc.)
            pass
        elif len(img.shape) == 2:
            # Grayscale image - add channel dimension
            img = np.expand_dims(img, axis=2)
        else:
            raise ValueError(f"Unexpected image shape: {img.shape} for file {file_path}")
            
    elif file_path.endswith(('.npy', '.npz')):
        # Load numpy file
        if file_path.endswith('.npz'):
            # Handle compressed numpy files
            npz_data = np.load(file_path)
            if len(npz_data.files) == 1:
                img = npz_data[npz_data.files[0]]
            else:
                raise ValueError(f"NPZ file {file_path} contains multiple arrays: {npz_data.files}")
        else:
            img = np.load(file_path)
        
        # Ensure proper dimensionality
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        elif len(img.shape) == 3:
            # If channels are first (C, H, W), move to last (H, W, C)
            if img.shape[0] <= 10 and img.shape[0] < min(img.shape[1], img.shape[2]):
                img = np.transpose(img, (1, 2, 0))
        else:
            raise ValueError(f"Unexpected numpy array shape: {img.shape} for file {file_path}")
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    return img


def get_dataset_info(dataset_path, input_type="rgb", target_type="dsm"):
    """
    Get information about the dataset structure and format.
    """
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    info = {
        'dataset_name': dataset_name,
        'dataset_path': dataset_path,
        'input_type': input_type,
        'target_type': target_type,
        'splits': {}
    }
    
    for split in ['train', 'valid', 'test']:
        split_format = detect_dataset_format(dataset_path, split, input_type, target_type)
        if split_format != 'unknown':
            if split_format == 'npy':
                # Count NPY files
                script_dir = os.path.dirname(os.path.abspath(__file__))
                npy_data_dir = os.path.join(script_dir, "data", dataset_name, split)
                x_files = len(os.listdir(os.path.join(npy_data_dir, 'x')))
                y_files = len(os.listdir(os.path.join(npy_data_dir, 'y')))
                info['splits'][split] = {
                    'format': split_format,
                    'input_files': x_files,
                    'target_files': y_files
                }
            else:
                # Count image files
                input_dir = os.path.join(dataset_path, split, input_type)
                target_dir = os.path.join(dataset_path, split, target_type)
                input_files = len([f for f in os.listdir(input_dir) 
                                 if os.path.isfile(os.path.join(input_dir, f))])
                target_files = len([f for f in os.listdir(target_dir) 
                                  if os.path.isfile(os.path.join(target_dir, f))])
                info['splits'][split] = {
                    'format': split_format,
                    'input_files': input_files,
                    'target_files': target_files
                }
    
    return info


def main():
    parser = argparse.ArgumentParser(
        description="Unified preprocessing script for Im2Height model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess DFC2023 dataset
  python preprocess_unified.py -d /path/to/DFC2023Amini
  
  # Preprocess with SAR input instead of RGB
  python preprocess_unified.py -d /path/to/dataset -i sar
  
  # Force reprocess existing NPY files
  python preprocess_unified.py -d /path/to/dataset --force
  
  # Get dataset information without processing
  python preprocess_unified.py -d /path/to/dataset --info-only
        """)
    
    parser.add_argument("-d", "--dataset_path", type=str, required=True,
                        help="Path to the dataset")
    parser.add_argument("-o", "--output_path", type=str, default=None,
                        help="Path to save the processed .npy files (defaults to 'data/<dataset_name>' in project root)")
    parser.add_argument("-i", "--input_type", type=str, default="rgb",
                        help="Input data type (default: rgb)")
    parser.add_argument("-t", "--target_type", type=str, default="dsm",
                        help="Target data type (default: dsm)")
    parser.add_argument("--force", action="store_true",
                        help="Force reprocess even if NPY files already exist")
    parser.add_argument("--info-only", action="store_true",
                        help="Only display dataset information, don't process")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")
    
    args = parser.parse_args()
    
    # Validate dataset path
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path does not exist: {args.dataset_path}")
        return 1
    
    # Get and display dataset information
    info = get_dataset_info(args.dataset_path, args.input_type, args.target_type)
    
    if not args.quiet:
        print(f"Dataset Information:")
        print(f"  Name: {info['dataset_name']}")
        print(f"  Path: {info['dataset_path']}")
        print(f"  Input Type: {info['input_type']}")
        print(f"  Target Type: {info['target_type']}")
        print(f"  Available Splits:")
        
        for split, split_info in info['splits'].items():
            print(f"    {split}: {split_info['format']} format, "
                  f"{split_info['input_files']} input files, "
                  f"{split_info['target_files']} target files")
    
    if not info['splits']:
        print(f"Error: No valid dataset splits found!")
        print(f"Expected structure: {args.dataset_path}/[train|valid|test]/{args.input_type}/")
        print(f"                    {args.dataset_path}/[train|valid|test]/{args.target_type}/")
        return 1
    
    if args.info_only:
        return 0
    
    # Preprocess the dataset
    success = preprocess_dataset(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        input_type=args.input_type,
        target_type=args.target_type,
        force_reprocess=args.force,
        verbose=not args.quiet
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
