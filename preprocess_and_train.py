#!/usr/bin/env python
# filepath: /home/asfand/Ahmad/IM2HEIGHT/preprocess_and_train.py
"""
Script to preprocess and train Im2Height model on DFC2023Amini dataset in one step
"""

import os
import argparse
import time
from preprocess_dfc2023 import preprocess_dataset
from train_dfc2023 import run as train_run

def main():
    parser = argparse.ArgumentParser(description="Preprocess DFC2023Amini dataset and train Im2Height model")
    parser.add_argument("-d", "--dataset_path", type=str, default="/home/asfand/Ahmad/datasets/DFC2023Amini",
                      help="Path to the DFC2023Amini dataset (default: /home/asfand/Ahmad/datasets/DFC2023Amini)")
    parser.add_argument("-i", "--input_type", type=str, default="rgb", choices=["rgb", "sar"],
                      help="Input data type (default: rgb)")
    parser.add_argument("-t", "--target_type", type=str, default="dsm",
                      help="Target data type (default: dsm)")
    parser.add_argument("-e", "--max_epochs", type=int, default=1000,
                      help="Maximum number of training epochs (default: 1000)")
    parser.add_argument("-p", "--patience", type=int, default=200,
                      help="Early stopping patience (default: 200)")
    parser.add_argument("-g", "--gpu_count", type=int, default=None,
                      help="Number of GPUs to use (default: auto-detect)")
                      
    args = parser.parse_args()
    
    # Create output directories
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "dfc2023")
    
    print(f"=== Starting Im2Height preprocessing and training ===")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Input type: {args.input_type}")
    print(f"Target type: {args.target_type}")
    print(f"NPY files will be saved to: {output_path}")
    print(f"Model weights will be saved to: {output_dir}")
    
    # Step 1: Preprocess the dataset
    print(f"\n=== Step 1: Preprocessing {args.input_type} and {args.target_type} data ===")
    start_time = time.time()
    preprocess_dataset(
        dataset_path=args.dataset_path,
        output_path=output_path,
        input_type=args.input_type,
        target_type=args.target_type
    )
    preprocess_time = time.time() - start_time
    print(f"Preprocessing completed in {preprocess_time:.2f} seconds")
    
    # Step 2: Train the model
    print(f"\n=== Step 2: Training the model ===")
    # Instead of using the dataset_path as input, we use the output_path from preprocessing
    # The original dataset_path is not used anymore
    train_run(
        dataset_path=args.dataset_path,  # Original files are still needed by DFC2023Dataset
        output_dir=output_dir,
        input_type=args.input_type,
        target_type=args.target_type,
        max_epochs=args.max_epochs,
        patience=args.patience,
        gpu_count=args.gpu_count
    )
    
    print(f"\n=== Training completed ===")
    print(f"Model weights saved to: {output_dir}")

if __name__ == "__main__":
    main()
