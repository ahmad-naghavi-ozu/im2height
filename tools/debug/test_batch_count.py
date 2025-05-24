#!/usr/bin/env python3
import torch
import os
import sys
import argparse

# Add the parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dfc2023_data import DFC2023Dataset
from train_dfc2023 import get_dynamic_config
from pytorch_lightning import Trainer
from im2height import Im2Height

def test_batch_count(gpu_count="0,1", dataset_path=None):
    """
    Test to verify correct batch count across multiple GPUs
    
    Args:
        gpu_count: GPU specification (e.g., "0,1,2,3" or "2")
        dataset_path: Path to dataset (defaults to DFC2023S)
    """
    # Settings similar to train_dfc2023.py
    if dataset_path is None:
        dataset_path = "/home/asfand/Ahmad/datasets/DFC2023S"
    input_type = "x"
    target_type = "y"
    
    print(f"Testing batch count with dataset: {dataset_path}")
    
    # Initialize datasets
    train_dataset = DFC2023Dataset(dataset_path, 'train', input_type, target_type)
    
    # Get sample to determine dimensions
    sample_input, _ = train_dataset[0]
    image_height, image_width = sample_input.shape[1], sample_input.shape[2]
    image_size = (image_width, image_height)
    
    # Determine GPU count
    if gpu_count.strip() == "0":
        trainer_devices = [0]
        num_available_gpus = 1
    elif ',' in gpu_count:
        trainer_devices = [int(g.strip()) for g in gpu_count.split(',')]
        num_available_gpus = len(trainer_devices)
    else:
        num_available_gpus = int(gpu_count)
        trainer_devices = num_available_gpus
    
    # Get dynamic config
    dynamic_config = get_dynamic_config(image_size, num_available_gpus)
    load_config = dynamic_config["config"]
    
    # Create data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        shuffle=True, 
        **load_config
    )
    
    # Print detailed analysis
    print(f"\n=== Dataset Analysis ===")
    print(f"- Train dataset has {len(train_dataset)} samples")
    print(f"- Image size: {image_size}")
    print(f"- Using {num_available_gpus} GPUs: {trainer_devices}")
    print(f"- Batch size: {load_config['batch_size']}")
    print(f"- Number of workers: {load_config.get('num_workers', 'default')}")
    print(f"- Number of batches: {len(train_loader)}")
    print(f"- Total processed per epoch: {len(train_loader) * load_config['batch_size']}")
    print(f"- Expected batches per GPU: {len(train_loader) / num_available_gpus:.1f}")
    
    # Check for data coverage
    total_processed = len(train_loader) * load_config['batch_size']
    coverage_percentage = (total_processed / len(train_dataset)) * 100
    print(f"- Data coverage: {coverage_percentage:.1f}% ({total_processed}/{len(train_dataset)})")
    
    if coverage_percentage < 100:
        print(f"  WARNING: Not all data will be processed! Missing {len(train_dataset) - total_processed} samples")
    
    # Create a simple model
    in_channels = sample_input.shape[0]
    model = Im2Height(in_channels=in_channels, out_channels=1)
    
    # Test with standard DDP strategy
    if num_available_gpus > 1:
        print(f"\n=== Testing Multi-GPU Training ===")
        
        # Standard trainer with ddp strategy
        trainer_standard = Trainer(
            devices=trainer_devices,
            accelerator="gpu",
            max_epochs=1,
            logger=False,
            enable_progress_bar=True,
            enable_checkpointing=False,
            strategy='ddp'
        )
        
        # Enhanced trainer with custom strategy for better sample handling
        from pytorch_lightning.strategies import DDPStrategy
        custom_strategy = DDPStrategy(find_unused_parameters=False, static_graph=True)
        
        trainer_custom = Trainer(
            devices=trainer_devices,
            accelerator="gpu",
            max_epochs=1,
            logger=False,
            enable_progress_bar=True,
            enable_checkpointing=False,
            strategy=custom_strategy,
            replace_sampler_ddp=False  # Ensure we don't replace the sampler
        )
        
        print("\nTesting standard DDP training...")
        try:
            trainer_standard.fit(model, train_loader)
            print("Standard DDP training completed successfully")
        except Exception as e:
            print(f"Standard training failed: {e}")
        
        print("\nTesting enhanced DDP training with custom strategy...")
        try:
            trainer_custom.fit(model, train_loader)
            print("Enhanced DDP training completed successfully")
        except Exception as e:
            print(f"Custom training failed: {e}")
    else:
        print(f"\n=== Testing Single-GPU Training ===")
        trainer_single = Trainer(
            devices=trainer_devices,
            accelerator="gpu",
            max_epochs=1,
            logger=False,
            enable_progress_bar=True,
            enable_checkpointing=False
        )
        
        print("Testing single-GPU training...")
        try:
            trainer_single.fit(model, train_loader)
            print("Single-GPU training completed successfully")
        except Exception as e:
            print(f"Single-GPU training failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test batch count with different GPU configurations")
    parser.add_argument("-g", "--gpus", type=str, default="0,1",
                      help="GPUs to use (comma-separated indices or integer count)")
    parser.add_argument("-d", "--dataset", type=str, default=None,
                      help="Path to dataset (defaults to DFC2023S)")
    
    args = parser.parse_args()
    test_batch_count(args.gpus, args.dataset)
