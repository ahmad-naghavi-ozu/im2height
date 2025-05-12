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
load_config = {
    "batch_size": 6,
    "pin_memory": True,
    "num_workers": 12
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
    
    # Initialize data loaders
    train_loader = torch.utils.data.DataLoader(
        DFC2023Dataset(dataset_path, 'train', input_type, target_type), 
        shuffle=True, 
        **load_config
    )
    
    valid_loader = torch.utils.data.DataLoader(
        DFC2023Dataset(dataset_path, 'valid', input_type, target_type), 
        **load_config
    )

    # Initialize model
    model = Im2Height()

    # Set up trainer with callbacks for early stopping and model checkpointing
    trainer = Trainer(
        gpus=gpu_count if gpu_count is not None else torch.cuda.device_count(),
        num_nodes=1,
        default_root_dir=output_dir,
        max_epochs=max_epochs,
        early_stop_callback=EarlyStopping(
            monitor='val_l1loss',
            patience=patience,
            verbose=False,
            mode='min'
        ),
        checkpoint_callback=ModelCheckpoint(
            filepath=os.path.join(output_dir, 'best_run.ckpt'),
            save_top_k=5,
            verbose=True,
            monitor='val_l1loss',
            mode='min',
            save_last=True
        )
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
