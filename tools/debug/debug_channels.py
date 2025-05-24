#!/usr/bin/env python3

import os
import sys
# Add the parent directory to path so we can import from the main project
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from data import PredictionDataset

def debug_channels():
    """Debug script to check channel detection in prediction dataset"""
    
    # Test with the NPY dataset - use relative path from tools/debug/ directory
    dataset_path = "../../data/DFC2023Amini"
    input_type = "rgb"
    
    print(f"Debugging channel detection for: {dataset_path}")
    print(f"Input type: {input_type}")
    
    # Convert to absolute path for checking existence
    abs_dataset_path = os.path.abspath(dataset_path)
    print(f"Absolute path: {abs_dataset_path}")
    
    # Check if dataset exists
    if not os.path.exists(abs_dataset_path):
        print(f"Error: Dataset path does not exist: {abs_dataset_path}")
        return
    
    # Check test directory
    test_dir = os.path.join(abs_dataset_path, "test")
    if not os.path.exists(test_dir):
        print(f"Error: Test directory does not exist: {test_dir}")
        return
    
    print(f"Test directory exists: {test_dir}")
    
    # List files in test directory
    test_files = os.listdir(test_dir)
    print(f"Files in test directory: {len(test_files)}")
    
    # Filter for input files
    input_files = [f for f in test_files if f.startswith(input_type) and f.endswith('.npy')]
    print(f"Input files ({input_type}): {len(input_files)}")
    if input_files:
        print(f"Sample input files: {input_files[:3]}")
    
    # Create prediction dataset
    try:
        prediction_dataset = PredictionDataset(abs_dataset_path, 'test', input_type)
        print(f"Prediction dataset created successfully")
        print(f"Dataset length: {len(prediction_dataset)}")
        
        if len(prediction_dataset) > 0:
            # Get first sample
            _, sample_tensor = prediction_dataset[0]
            print(f"Sample tensor shape: {sample_tensor.shape}")
            print(f"Detected channels: {sample_tensor.shape[0]}")
            print(f"Image size: {(sample_tensor.shape[2], sample_tensor.shape[1])}")
        else:
            print("Error: No samples in dataset")
            
    except Exception as e:
        print(f"Error creating prediction dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_channels()
