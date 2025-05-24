#!/usr/bin/env python3
import os
import sys
import torch

# Add the parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dfc2023_data import DFC2023Dataset
from train_dfc2023 import get_dynamic_config

def check_dataloader(dataset_path=None):
    """
    Analyze dataloader configuration and behavior
    
    Args:
        dataset_path: Path to dataset (defaults to DFC2023S)
    """
    if dataset_path is None:
        dataset_path = "/home/asfand/Ahmad/datasets/DFC2023S"
    
    print(f"=== Dataloader Analysis ===")
    print(f"Dataset path: {dataset_path}")
    
    # Initialize datasets
    try:
        train_dataset = DFC2023Dataset(dataset_path, 'train', 'x', 'y')
        valid_dataset = DFC2023Dataset(dataset_path, 'valid', 'x', 'y')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Check input sample to determine dimensions
    sample_input, sample_target = train_dataset[0]
    image_height, image_width = sample_input.shape[1], sample_input.shape[2]
    image_size = (image_width, image_height)
    
    # Get the dynamic config (same as in train_dfc2023.py)
    num_gpus = torch.cuda.device_count()
    dynamic_config = get_dynamic_config(image_size, num_gpus)
    load_config = dynamic_config["config"]
    
    print(f"- Train dataset size: {len(train_dataset)} samples")
    print(f"- Valid dataset size: {len(valid_dataset)} samples")
    print(f"- Input image size: {image_size}")
    print(f"- Input shape: {sample_input.shape}")
    print(f"- Target shape: {sample_target.shape}")
    print(f"- Number of GPUs detected: {num_gpus}")
    print(f"- Dynamic batch size: {load_config['batch_size']}")
    print(f"- Number of workers: {load_config.get('num_workers', 'default')}")
    print(f"- Pin memory: {load_config.get('pin_memory', 'default')}")
    print(f"- Gradient accumulation: {dynamic_config['gradient_accum']}")
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        shuffle=True, 
        **load_config
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        **load_config
    )
    
    # Calculate number of batches
    num_train_batches = len(train_loader)
    num_valid_batches = len(valid_loader)
    
    print(f"\n=== Batch Analysis ===")
    print(f"- Train batches: {num_train_batches}")
    print(f"- Valid batches: {num_valid_batches}")
    print(f"- Total train samples processed per epoch: {num_train_batches * load_config['batch_size']}")
    print(f"- Total valid samples processed per epoch: {num_valid_batches * load_config['batch_size']}")
    
    # Check a few batches for consistency
    print(f"\n=== Sample Batch Analysis ===")
    batch_sizes = []
    for i, (inputs, targets) in enumerate(train_loader):
        batch_sizes.append(inputs.shape[0])
        if i == 0:
            print(f"- First batch input shape: {inputs.shape}")
            print(f"- First batch target shape: {targets.shape}")
        if i >= 4:  # Just check the first few batches
            break
    
    print(f"- Sample batch sizes: {batch_sizes}")
    print(f"- Batch size consistency: {'✓ Consistent' if len(set(batch_sizes)) <= 1 else '✗ Inconsistent'}")
    
    # Memory usage estimation
    single_sample_memory = (sample_input.numel() * 4 + sample_target.numel() * 4) / (1024**2)  # MB
    batch_memory = single_sample_memory * load_config['batch_size']
    print(f"\n=== Memory Estimation ===")
    print(f"- Single sample memory: {single_sample_memory:.2f} MB")
    print(f"- Single batch memory: {batch_memory:.2f} MB")
    print(f"- Multi-GPU batch memory: {batch_memory * num_gpus:.2f} MB")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze dataloader configuration")
    parser.add_argument("-d", "--dataset", type=str, default=None,
                      help="Path to dataset (defaults to DFC2023S)")
    
    args = parser.parse_args()
    check_dataloader(args.dataset)
