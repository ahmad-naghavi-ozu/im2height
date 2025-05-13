import os
import argparse
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from im2height import Im2Height
from dfc2023_data import DFC2023Dataset


# load config with same parameters as the original implementation
# Using a smaller batch size to reduce GPU memory usage
load_config = {
    "batch_size": 2,  # Reduced from 6 to avoid out of memory errors
    "pin_memory": True,
    "num_workers": 4   # Reduced from 12 to lower memory usage
}


def run(dataset_path, output_dir="weights/dfc2023", input_type="rgb", target_type="dsm", 
        max_epochs=1000, patience=200, gpu_count=None):
    """
    Train the Im2Height model on DFC2023Amini dataset
    
    Args:
        dataset_path: Path to the DFC2023Amini dataset
        output_dir: Directory to save model weights
        input_type: Input data type ('rgb' or 'sar')
        target_type: Target data type (usually 'dsm')
        max_epochs: Maximum number of training epochs
        patience: Early stopping patience
        gpu_count: Number of GPUs to use (None for auto-detection)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize datasets
    train_dataset = DFC2023Dataset(dataset_path, 'train', input_type, target_type)
    valid_dataset = DFC2023Dataset(dataset_path, 'valid', input_type, target_type)
    
    # Initialize data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        shuffle=True, 
        **load_config
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        **load_config
    )
    
    # Check input sample to determine number of input channels
    sample_input, _ = train_dataset[0]
    
    # Ensure we're getting the channel count, not image dimensions
    # For RGB images this should be 3, for grayscale 1
    in_channels = sample_input.shape[0]  # Channel dimension is first in PyTorch tensors
    
    # Sanity check - if channels > 4, it's likely being confused with image dimensions
    if in_channels > 4:
        print(f"Warning: Detected unusually high number of channels: {in_channels}")
        print(f"Full tensor shape: {sample_input.shape}")
        print("Assuming RGB input (3 channels) - adjust manually if needed")
        in_channels = 3 if input_type == 'rgb' else 1
    
    out_channels = 1  # Output is always height map with one channel
    
    print(f"Using {in_channels} input channels for model")

    # Initialize model with detected number of channels
    model = Im2Height(in_channels=in_channels, out_channels=out_channels)

    # Set up trainer with callbacks for early stopping and model checkpointing
    early_stop = EarlyStopping(
        monitor='val_l1loss',
        patience=patience,
        verbose=False,
        mode='min'
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='best_run-{epoch:02d}-{val_l1loss:.4f}',
        save_top_k=5,
        verbose=True,
        monitor='val_l1loss',
        mode='min',
        save_last=True
    )
    
    # Set up trainer with callbacks
    # Handle different GPU specifications correctly
    if gpu_count is not None:
        # If user passes in "0", we interpret it as "use GPU 0", not "use 0 GPUs"
        if isinstance(gpu_count, str) and gpu_count.strip() == "0":
            trainer_devices = [0]  # Use GPU 0
        elif isinstance(gpu_count, int) and gpu_count == 0:
            trainer_devices = [0]  # Use GPU 0
        else:
            # If it's a comma-separated list like "0,1", convert to a list of integers
            if isinstance(gpu_count, str) and ',' in gpu_count:
                trainer_devices = [int(g.strip()) for g in gpu_count.split(',')]
            else:
                # Otherwise use the value as is
                trainer_devices = gpu_count
    else:
        trainer_devices = torch.cuda.device_count()
    
    trainer = Trainer(
        devices=trainer_devices,
        accelerator="gpu",
        num_nodes=1,
        default_root_dir=output_dir,
        max_epochs=max_epochs,
        callbacks=[early_stop, checkpoint_callback],
        gradient_clip_val=0.5,  # Add gradient clipping to improve stability
        accumulate_grad_batches=3,  # Accumulate gradients to simulate larger batch size
        precision=16  # Use mixed precision to reduce memory usage
    )

    # Train the model
    trainer.fit(model, train_loader, valid_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Im2Height model on DFC2023Amini dataset")
    parser.add_argument("-d", "--dataset_path", type=str, required=True,
                        help="Path to the DFC2023Amini dataset")
    parser.add_argument("-o", "--output_dir", type=str, default="weights/dfc2023",
                        help="Directory to save model weights")
    parser.add_argument("-i", "--input_type", type=str, default="rgb", choices=["rgb", "sar"],
                        help="Input data type")
    parser.add_argument("-t", "--target_type", type=str, default="dsm",
                        help="Target data type")
    parser.add_argument("-e", "--max_epochs", type=int, default=1000,
                        help="Maximum number of training epochs")
    parser.add_argument("-p", "--patience", type=int, default=200,
                        help="Early stopping patience")
    parser.add_argument("-g", "--gpu_count", type=int, default=None,
                        help="Number of GPUs to use (default: auto-detect)")
                        
    args = parser.parse_args()
    run(**vars(args))
