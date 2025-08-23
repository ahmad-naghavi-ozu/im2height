import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from im2height import Im2Height
from data import PredictionDataset, prediction_collate_fn


def get_dynamic_predict_config(image_size=(256, 256), num_gpus=1):
    """Dynamic configuration for prediction based on image size and available GPUs."""
    BASE_SIZE = 256 * 256
    BASE_BATCH_SIZE = 32
    BASE_WORKERS = 32
    
    size_ratio = (image_size[0] * image_size[1]) / BASE_SIZE
    adjusted_batch_size = max(1, int(BASE_BATCH_SIZE / size_ratio))
    adjusted_workers = max(2, int(BASE_WORKERS / (size_ratio ** 0.5)))
    
    return {
        "batch_size": adjusted_batch_size,
        "pin_memory": True,
        "num_workers": adjusted_workers
    }


def run(dataset_path=None, input_files=None, output_dir="predictions", weights=None, 
        auto_find_weights=True, quiet=False):
    """
    Run predictions using trained Im2Height model for RGB to DSM conversion.
    
    Args:
        dataset_path: Path to dataset directory (for structured datasets)
        input_files: List of input files (for direct file prediction)
        output_dir: Directory to save predictions
        weights: Path to model weights (auto-finds if None)
        auto_find_weights: Whether to auto-find weights if not specified
        quiet: Suppress verbose output
    """
    
    # Handle input specification
    if dataset_path is None and input_files is None:
        # Try legacy prediction with files in current directory
        try:
            input_files = [f for f in os.listdir('.') if f.endswith('.npy')]
            if not input_files:
                raise ValueError("No input specified and no .npy files found in current directory")
            if not quiet:
                print(f"Using legacy mode: found {len(input_files)} .npy files")
        except:
            raise ValueError("Must specify either dataset_path or input_files")
    
    # Initialize prediction dataset
    if input_files is not None:
        prediction_dataset = PredictionDataset(input_files)
    else:
        prediction_dataset = PredictionDataset(dataset_path, 'test')
    
    # Get dataset characteristics
    if len(prediction_dataset) > 0:
        _, sample_tensor = prediction_dataset[0]
        in_channels = sample_tensor.shape[0]
        image_size = (sample_tensor.shape[2], sample_tensor.shape[1])  # (width, height)
    else:
        raise ValueError("No samples found for prediction")
    
    if not quiet:
        print(f"Input channels: {in_channels}")
        print(f"Image size: {image_size}")
        print(f"Prediction samples: {len(prediction_dataset)}")
    
    # Auto-find weights if not specified
    if weights is None and auto_find_weights:
        # Look for weights in common locations
        weight_patterns = [
            "weights/**/best*.ckpt",
            "weights/**/*best*.ckpt", 
            "*.ckpt",
            "best*.ckpt"
        ]
        
        import glob
        found_weights = []
        for pattern in weight_patterns:
            found_weights.extend(glob.glob(pattern, recursive=True))
        
        if found_weights:
            weights = sorted(found_weights)[-1]  # Use most recent
            if not quiet:
                print(f"Auto-found weights: {weights}")
        else:
            raise ValueError("No weights specified and none found automatically")
    
    # Load model with correct hyperparameters and proper device mapping
    # Handle GPU memory issues by trying GPU first, then falling back to CPU
    import torch
    device_count = torch.cuda.device_count()
    
    # Try GPU first, but handle OOM gracefully
    if device_count > 0:
        try:
            # Clear any existing GPU cache first
            torch.cuda.empty_cache()
            map_location = f'cuda:0'
            model = Im2Height.load_from_checkpoint(weights, in_channels=in_channels, out_channels=1, map_location=map_location)
            if not quiet:
                print(f"Loaded model from checkpoint: {weights}")
                print(f"Mapped to device: {map_location}")
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if not quiet:
                print(f"GPU loading failed ({e}), falling back to CPU...")
            map_location = 'cpu'
            model = Im2Height.load_from_checkpoint(weights, in_channels=in_channels, out_channels=1, map_location=map_location)
            if not quiet:
                print(f"Loaded model from checkpoint: {weights}")
                print(f"Mapped to device: {map_location} (fallback)")
    else:
        map_location = 'cpu'
        model = Im2Height.load_from_checkpoint(weights, in_channels=in_channels, out_channels=1, map_location=map_location)
        if not quiet:
            print(f"Loaded model from checkpoint: {weights}")
            print(f"Mapped to device: {map_location}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get dynamic configuration
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    load_config = get_dynamic_predict_config(image_size, num_gpus)
    
    # Create data loader
    prediction_loader = torch.utils.data.DataLoader(
        prediction_dataset, 
        collate_fn=prediction_collate_fn,
        **load_config
    )
    
    if not quiet:
        print(f"Prediction loader: {len(prediction_loader)} batches")
    
    # Set up trainer for prediction
    # Use the same device as the model was loaded on
    use_gpu = str(next(model.parameters()).device).startswith('cuda')
    
    trainer = Trainer(
        devices=1 if use_gpu else 1,
        accelerator="gpu" if use_gpu else "cpu",
        logger=False,
        enable_checkpointing=False
    )
    
    # Run predictions
    if not quiet:
        print("Running predictions...")
    predictions = trainer.predict(model, prediction_loader)
    
    # Save predictions
    if not quiet:
        print(f"Saving predictions to {output_dir}")
    for batch_idx, (file_paths, pred_batch) in enumerate(zip(
        [batch[0] for batch in prediction_loader], predictions)):
        
        for i, (file_path, pred) in enumerate(zip(file_paths, pred_batch)):
            # Generate output filename
            basename = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(output_dir, f"{basename}.npy")
            
            # Remove padding and save
            padding = 0  # Must match the padding used in PredictionDataset
            if padding > 0:
                pred_clean = pred[0, padding:-padding, padding:-padding]
            else:
                pred_clean = pred[0]
            
            np.save(output_path, pred_clean.cpu().numpy())
    
    if not quiet:
        print(f"Predictions completed! Results saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run predictions using trained Im2Height model")
    parser.add_argument("-d", "--dataset_path", type=str, default=None,
                        help="Path to dataset directory (for structured datasets)")
    parser.add_argument("-f", "--input_files", type=str, nargs="+", default=None,
                        help="List of input files for direct prediction")
    parser.add_argument("-o", "--output_dir", type=str, default="predictions",
                        help="Output directory for predictions")
    parser.add_argument("-w", "--weights", type=str, default=None,
                        help="Path to model weights (auto-finds if not specified)")
    parser.add_argument("--no_auto_weights", action="store_true",
                        help="Disable automatic weight finding")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")
    
    # Backward compatibility arguments
    parser.add_argument("--input", type=str, nargs="+", default=None,
                        help="Legacy: Input file paths")
    parser.add_argument("--output", type=str, default=None,
                        help="Legacy: Output directory")
    
    args = parser.parse_args()
    
    # Handle backward compatibility
    if args.input is not None or args.output is not None:
        # Legacy interface
        run(
            input_files=args.input, 
            output_dir=args.output or "predictions", 
            weights=args.weights,
            quiet=args.quiet
        )
    else:
        # New interface
        run(
            dataset_path=args.dataset_path,
            input_files=args.input_files,
            output_dir=args.output_dir,
            weights=args.weights,
            auto_find_weights=not args.no_auto_weights,
            quiet=args.quiet
        )
