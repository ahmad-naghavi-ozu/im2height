import os
import argparse
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from im2height import Im2Height
from dfc2023_data import DFC2023Dataset


# Dynamic configuration based on input image size and available GPUs
def get_dynamic_config(image_size=(256, 256), num_gpus=1):
    """
    Calculate appropriate batch size and worker count based on image dimensions and available GPUs.
    The original paper used a batch size of 6 for 256x256 images on a single GPU.
    
    Args:
        image_size: Tuple of (width, height) of input images
        num_gpus: Number of GPUs being used for training
        
    Returns:
        Dictionary with batch_size and num_workers calculated dynamically
    """
    # Base values for 256x256 images from the paper (for a single GPU)
    BASE_SIZE = 256 * 256
    BASE_BATCH_SIZE = 6
    BASE_WORKERS = 12
    
    # Calculate the ratio of current image size to base size
    # Use squared relationship as memory usage grows quadratically with image size
    size_ratio = (image_size[0] * image_size[1]) / BASE_SIZE
    
    # Adjust batch size based on image dimensions (inverse relationship)
    # For larger images, use smaller batch size
    adjusted_batch_size = max(1, int(BASE_BATCH_SIZE / size_ratio))
    
    # Scale with number of GPUs - we can use more total batch size with more GPUs
    # But be conservative with the scaling (sqrt) to avoid out-of-memory errors
    if num_gpus > 1:
        adjusted_batch_size = max(1, int(adjusted_batch_size * (num_gpus ** 0.5)))
    
    # Adjust number of workers (less aggressive scaling)
    # Each GPU can handle its own workers
    adjusted_workers = max(2, int((BASE_WORKERS / (size_ratio ** 0.5)) * num_gpus))
    
    # For gradient accumulation, we want to simulate a larger effective batch size
    # With multiple GPUs, we need less accumulation as we're already parallelizing
    gradient_accum = max(1, int(BASE_BATCH_SIZE / adjusted_batch_size))
    
    print(f"Using dynamic configuration for image size {image_size} with {num_gpus} GPU(s):")
    print(f"  - Batch size: {adjusted_batch_size} (base: {BASE_BATCH_SIZE})")
    print(f"  - Workers: {adjusted_workers} (base: {BASE_WORKERS})")
    print(f"  - Gradient accumulation: {gradient_accum} (effective batch: {adjusted_batch_size * gradient_accum})")
    
    return {
        "config": {
            "batch_size": adjusted_batch_size,
            "pin_memory": True,
            "num_workers": adjusted_workers
        },
        "gradient_accum": gradient_accum
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
    
    # Check input sample to determine dimensions and channels
    sample_input, _ = train_dataset[0]
    
    # Get image dimensions (height, width) - channels are in first position for PyTorch tensors
    image_height, image_width = sample_input.shape[1], sample_input.shape[2]
    image_size = (image_width, image_height)
    
    # Count available GPUs or use the specified count
    num_available_gpus = 1  # Default to 1 GPU
    if gpu_count is not None:
        if gpu_count.strip() == "0":
            num_available_gpus = 1  # Using GPU 0
        elif ',' in gpu_count:
            num_available_gpus = len([int(g.strip()) for g in gpu_count.split(',')])
        elif gpu_count.strip().isdigit():
            num_available_gpus = max(1, int(gpu_count.strip()))
    else:
        num_available_gpus = torch.cuda.device_count()
    
    # Get dynamic configuration based on image size and number of GPUs
    dynamic_config = get_dynamic_config(image_size, num_available_gpus)
    load_config = dynamic_config["config"]
    gradient_accum = dynamic_config["gradient_accum"]
    
    # Initialize data loaders with dynamic configuration
    # For multi-GPU training, set proper batch size that accounts for all GPUs
    # This ensures each GPU processes the intended number of samples
    if num_available_gpus > 1:
        # For multi-GPU training, adjust the effective batch size
        # Each GPU should process (total samples / num_gpus) samples per epoch
        print(f"Adjusting dataloader for multi-GPU training: {num_available_gpus} GPUs")
        print(f"Total train dataset size: {len(train_dataset)}")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        shuffle=True, 
        **load_config
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        **load_config
    )
    
    print(f"Train loader has {len(train_loader)} batches with batch size {load_config['batch_size']}")
    print(f"Total samples to process per epoch: {len(train_loader) * load_config['batch_size']}")
    print(f"Validation loader has {len(valid_loader)} batches")
    
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
    
    # Determine appropriate memory saving techniques based on image size
    use_mixed_precision = (image_width >= 512 or image_height >= 512)
    precision = 16 if use_mixed_precision else 32
    
    # Set up trainer with callbacks
    # Handle different GPU specifications correctly
    if gpu_count is not None:
        # If user passes in "0", we interpret it as "use GPU 0", not "use 0 GPUs"
        if gpu_count.strip() == "0":
            trainer_devices = [0]  # Use GPU 0
        # If it's a comma-separated list like "0,1", convert to a list of integers
        elif ',' in gpu_count:
            trainer_devices = [int(g.strip()) for g in gpu_count.split(',')]
        # If it's a numeric string, interpret as number of GPUs
        elif gpu_count.strip().isdigit():
            num_gpus = int(gpu_count.strip())
            if num_gpus == 0:
                trainer_devices = [0]  # Use GPU 0
            else:
                # Use the specified number of GPUs
                trainer_devices = num_gpus
        else:
            # Fallback: just use all available GPUs
            print(f"Warning: Invalid GPU specification '{gpu_count}'. Using all available GPUs.")
            trainer_devices = torch.cuda.device_count()
    else:
        trainer_devices = torch.cuda.device_count()
    
    # For multi-GPU training, we need to ensure the full dataset is processed
    # rather than each GPU only seeing a portion of the data
    use_distributed_strategy = False
    if (isinstance(trainer_devices, list) and len(trainer_devices) > 1) or (isinstance(trainer_devices, int) and trainer_devices > 1):
        print(f"Multi-GPU training detected with {num_available_gpus} GPUs")
        print(f"Ensuring all {len(train_dataset)} training samples are processed")
        use_distributed_strategy = True
    
    # Configure the appropriate distributed strategy for PyTorch Lightning
    training_strategy = None
    if use_distributed_strategy:
        # Use simple string strategy instead of the custom strategy object
        # This is more compatible with different PyTorch Lightning versions
        training_strategy = 'ddp'
        print(f"Using distributed training strategy: {training_strategy}")
    
    trainer = Trainer(
        devices=trainer_devices,
        accelerator="gpu",
        num_nodes=1,
        default_root_dir=output_dir,
        max_epochs=max_epochs,
        callbacks=[early_stop, checkpoint_callback],
        gradient_clip_val=0.5,  # Add gradient clipping to improve stability
        accumulate_grad_batches=gradient_accum,  # Use dynamic gradient accumulation
        precision=precision,  # Use mixed precision for larger images
        strategy=training_strategy
    )

    # Train the model
    trainer.fit(model, train_loader, valid_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Im2Height model on DFC2023 dataset variants")
    parser.add_argument("-d", "--dataset_path", type=str, required=True,
                        help="Path to the DFC2023 dataset variant (e.g., DFC2023S, DFC2023A, DFC2023Asmall, DFC2023Amini)")
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
    parser.add_argument("-g", "--gpu_count", type=str, default=None,
                        help="GPUs to use (comma-separated indices or integer count, e.g., '0,1' or 2)")
                        
    args = parser.parse_args()
    run(**vars(args))
