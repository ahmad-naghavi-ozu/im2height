"""
Test script to verify dataset loading and tensor shapes
"""

import os
import numpy as np
import torch
from dfc2023_data import DFC2023PredictionDataset

def main():
    # Path to the dataset
    dataset_path = "/home/asfand/Ahmad/datasets/DFC2023Amini"
    
    # Create dataset instance for testing
    dataset = DFC2023PredictionDataset(dataset_path, split="test", input_type="rgb")
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test loading a few samples
    for i in range(min(3, len(dataset))):
        file_path, tensor = dataset[i]
        print(f"\nSample {i+1}:")
        print(f"File: {os.path.basename(file_path)}")
        print(f"Tensor shape: {tensor.shape}")
        print(f"Tensor type: {tensor.dtype}")
        print(f"Tensor min/max: {tensor.min().item():.4f}/{tensor.max().item():.4f}")
        print(f"First few values: {tensor[:, 0, 0]}")
        
        # Check if tensor has expected channel count (should be 3 for RGB)
        if tensor.shape[0] != 3:
            print(f"WARNING: Expected 3 channels, got {tensor.shape[0]}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()
