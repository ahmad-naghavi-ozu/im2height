import os
import sys
import argparse
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from im2height import Im2Height
from data import Dataset


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
    size_ratio = (image_size[0] * image_size[1]) / BASE_SIZE
    
    # Adjust batch size based on image dimensions (inverse relationship)
    adjusted_batch_size = max(1, int(BASE_BATCH_SIZE / size_ratio))
    
    # Scale with number of GPUs (conservative scaling)
    if num_gpus > 1:
        adjusted_batch_size = max(1, int(adjusted_batch_size * (num_gpus ** 0.5)))
    
    # Adjust number of workers
    adjusted_workers = max(2, int((BASE_WORKERS / (size_ratio ** 0.5)) * num_gpus))
    
    # Gradient accumulation for effective batch size
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


def run(dataset_path=None, input_type="rgb", target_type="dsm", max_epochs=1000, 
        patience=200, gpu_count=None, output_dir="weights"):
    """
    Train the Im2Height model on any supported dataset format.
    
    Args:
        dataset_path: Path to dataset (auto-detects NPY or image format)
        input_type: Input data type ('rgb', 'sar', etc.) - for image format datasets
        target_type: Target data type ('dsm', etc.) - for image format datasets  
        max_epochs: Maximum number of training epochs
        patience: Early stopping patience
        gpu_count: GPU specification (None for auto-detect, '0,1' for specific GPUs)
        output_dir: Directory to save model weights
    """
    
    # Handle backward compatibility - check for old NPY structure first
    if dataset_path is None:
        # Try old NPY structure for backward compatibility
        if os.path.exists('data/train/x') and os.path.exists('data/train/y'):
            dataset_path = 'data'
            print("Using legacy NPY dataset structure")
        else:
            raise ValueError("No dataset path provided and no legacy data structure found")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize datasets using dataset class
    train_dataset = Dataset(dataset_path, 'train', input_type, target_type)
    
    # Try to find validation data, fall back to test if not available
    try:
        valid_dataset = Dataset(dataset_path, 'valid', input_type, target_type)
        print("Using validation dataset")
    except:
        try:
            valid_dataset = Dataset(dataset_path, 'test', input_type, target_type)
            print("Using test dataset for validation")
        except:
            raise ValueError("No validation or test data found")
    
    # Get dataset characteristics
    image_size = train_dataset.get_image_size()
    in_channels = train_dataset.get_input_channels()
    
    print(f"Dataset format: {train_dataset.dataset_format}")
    print(f"Input channels: {in_channels}")
    print(f"Image size: {image_size}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    
    # Determine GPU configuration
    num_available_gpus = 1
    if gpu_count is not None:
        if gpu_count.strip() == "0":
            num_available_gpus = 1
        elif ',' in gpu_count:
            num_available_gpus = len([int(g.strip()) for g in gpu_count.split(',')])
        elif gpu_count.strip().isdigit():
            num_available_gpus = max(1, int(gpu_count.strip()))
    else:
        num_available_gpus = torch.cuda.device_count()
    
    # Get dynamic configuration
    dynamic_config = get_dynamic_config(image_size, num_available_gpus)
    load_config = dynamic_config["config"]
    gradient_accum = dynamic_config["gradient_accum"]
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **load_config)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, **load_config)
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Validation loader: {len(valid_loader)} batches")
    
    # Initialize model
    model = Im2Height(in_channels=in_channels, out_channels=1)
    
    # Set up callbacks
    early_stop = EarlyStopping(
        monitor='val_l1loss',
        patience=patience,
        verbose=False,
        mode='min'
    )
    
    # Create output filename with dataset info
    dataset_name = os.path.basename(os.path.normpath(dataset_path)) if dataset_path != 'data' else 'legacy'
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename=f'{dataset_name}-best-{{epoch:02d}}-{{val_l1loss:.4f}}',
        save_top_k=5,
        verbose=True,
        monitor='val_l1loss',
        mode='min',
        save_last=True
    )
    
    # Configure trainer devices
    if gpu_count is not None:
        if gpu_count.strip() == "0":
            trainer_devices = [0]
        elif ',' in gpu_count:
            trainer_devices = [int(g.strip()) for g in gpu_count.split(',')]
        elif gpu_count.strip().isdigit():
            num_gpus = int(gpu_count.strip())
            trainer_devices = num_gpus if num_gpus > 0 else [0]
        else:
            print(f"Warning: Invalid GPU specification '{gpu_count}'. Using all available GPUs.")
            trainer_devices = torch.cuda.device_count()
    else:
        trainer_devices = torch.cuda.device_count()
    
    # Determine if we need distributed training
    use_distributed = False
    if (isinstance(trainer_devices, list) and len(trainer_devices) > 1) or \
       (isinstance(trainer_devices, int) and trainer_devices > 1):
        use_distributed = True
        training_strategy = 'ddp'
        print(f"Using distributed training with {num_available_gpus} GPUs")
    else:
        training_strategy = None
    
    # Use mixed precision for larger images
    use_mixed_precision = (image_size[0] >= 512 or image_size[1] >= 512)
    precision = 16 if use_mixed_precision else 32
    
    if use_mixed_precision:
        print("Using mixed precision training for large images")
    
    # Create trainer
    trainer = Trainer(
        devices=trainer_devices,
        accelerator="gpu",
        num_nodes=1,
        default_root_dir=output_dir,
        max_epochs=max_epochs,
        callbacks=[early_stop, checkpoint_callback],
        gradient_clip_val=0.5,
        accumulate_grad_batches=gradient_accum,
        precision=precision,
        strategy=training_strategy
    )
    
    # Train the model
    trainer.fit(model, train_loader, valid_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Im2Height model on any supported dataset format")
    parser.add_argument("-d", "--dataset_path", type=str, default=None,
                        help="Path to dataset (auto-detects NPY or image format)")
    parser.add_argument("-o", "--output_dir", type=str, default="weights",
                        help="Directory to save model weights")
    parser.add_argument("-i", "--input_type", type=str, default="rgb",
                        help="Input data type (for image format datasets)")
    parser.add_argument("-t", "--target_type", type=str, default="dsm",
                        help="Target data type (for image format datasets)")
    parser.add_argument("-e", "--max_epochs", type=int, default=1000,
                        help="Maximum number of training epochs")
    parser.add_argument("-p", "--patience", type=int, default=200,
                        help="Early stopping patience")
    parser.add_argument("-g", "--gpu_count", type=str, default=None,
                        help="GPUs to use (comma-separated indices or integer count)")
    
    args = parser.parse_args()
    
    # For backward compatibility - if no arguments, try legacy dataset
    if len(sys.argv) == 1:
        try:
            run()  # Try with no arguments for legacy support
        except:
            print("No arguments provided and legacy dataset not found.")
            print("Use --help for usage information.")
    else:
        run(**vars(args))
