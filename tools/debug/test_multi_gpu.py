#!/usr/bin/env python3
import os
import sys
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Add the parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from im2height import Im2Height
from dfc2023_data import DFC2023Dataset

def test_multi_gpu(dataset_path=None, gpu_indices=[0, 1], batch_size=2, max_epochs=2):
    """
    Test that multi-GPU training works correctly
    
    Args:
        dataset_path: Path to the DFC2023 dataset
        gpu_indices: List of GPU indices to use [0,1,2,3]
        batch_size: Batch size per GPU
        max_epochs: Number of epochs to run (for testing)
    """
    if dataset_path is None:
        dataset_path = "/home/asfand/Ahmad/datasets/DFC2023S"
    
    print(f"=== Multi-GPU Training Test ===")
    print(f"Testing multi-GPU training with {len(gpu_indices)} GPUs: {gpu_indices}")
    print(f"Dataset: {dataset_path}")
    print(f"Batch size: {batch_size}")
    print(f"Max epochs: {max_epochs}")
    
    # Initialize datasets
    try:
        train_dataset = DFC2023Dataset(dataset_path, 'train', 'rgb', 'dsm')
        valid_dataset = DFC2023Dataset(dataset_path, 'valid', 'rgb', 'dsm')
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return
    
    # Get a sample to determine input channels
    sample_input, sample_target = train_dataset[0]
    in_channels = sample_input.shape[0]
    
    print(f"\n=== Dataset Info ===")
    print(f"- Input shape: {sample_input.shape}, using {in_channels} channels")
    print(f"- Target shape: {sample_target.shape}")
    print(f"- Training dataset size: {len(train_dataset)}")
    print(f"- Validation dataset size: {len(valid_dataset)}")
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )
    
    # Print dataloader information
    print(f"\n=== Dataloader Info ===")
    print(f"- Train loader: {len(train_loader)} batches")
    print(f"- Valid loader: {len(valid_loader)} batches")
    print(f"- Total train samples per epoch: {len(train_loader) * batch_size}")
    print(f"- Total valid samples per epoch: {len(valid_loader) * batch_size}")
    print(f"- Expected batches per GPU: {len(train_loader) / len(gpu_indices):.1f}")
    
    # Initialize model
    model = Im2Height(in_channels=in_channels, out_channels=1)
    
    # Set up callbacks
    output_dir = "tools/debug/test_multi_gpu_weights"
    os.makedirs(output_dir, exist_ok=True)
    
    early_stop = EarlyStopping(
        monitor='val_l1loss',
        patience=5,
        verbose=True,
        mode='min'
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='test-multi-gpu-{epoch:02d}-{val_l1loss:.4f}',
        save_top_k=1,
        verbose=True,
        monitor='val_l1loss',
        mode='min'
    )
    
    # Test different strategies
    strategies = ['ddp', 'ddp_find_unused_parameters_false']
    
    for strategy in strategies:
        print(f"\n=== Testing Strategy: {strategy} ===")
        
        # Create trainer with current strategy
        trainer = Trainer(
            devices=gpu_indices,
            accelerator="gpu",
            max_epochs=max_epochs,
            callbacks=[early_stop, checkpoint_callback],
            strategy=strategy,
            enable_progress_bar=True,
            logger=False,  # Disable logging for testing
            enable_checkpointing=True
        )
        
        # Fit model
        try:
            print(f"Starting training with {strategy}...")
            trainer.fit(model, train_loader, valid_loader)
            print(f"✓ Training with {strategy} completed successfully!")
        except Exception as e:
            print(f"❌ Training with {strategy} failed: {e}")
        
        # Reset model for next test
        model = Im2Height(in_channels=in_channels, out_channels=1)
    
    print(f"\n=== Multi-GPU Test Complete ===")
    print(f"Check {output_dir} for saved model weights")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test multi-GPU training functionality")
    parser.add_argument("-d", "--dataset", type=str, default=None,
                      help="Path to dataset (defaults to DFC2023S)")
    parser.add_argument("-g", "--gpus", type=str, default="0,1",
                      help="Comma-separated GPU indices (e.g., '0,1,2,3')")
    parser.add_argument("-b", "--batch_size", type=int, default=2,
                      help="Batch size per GPU")
    parser.add_argument("-e", "--epochs", type=int, default=2,
                      help="Number of epochs to run")
    
    args = parser.parse_args()
    
    # Parse GPU indices
    gpu_indices = [int(g.strip()) for g in args.gpus.split(',')]
    
    test_multi_gpu(args.dataset, gpu_indices, args.batch_size, args.epochs)
