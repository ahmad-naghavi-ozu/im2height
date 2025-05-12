import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from im2height import Im2Height
from dfc2023_data import DFC2023PredictionDataset


# load config with same parameters as the original implementation
load_config = {
    "batch_size": 32,
    "pin_memory": True,
    "num_workers": 32
}


def run(dataset_path, output_dir, weights, split="test", input_type="rgb"):
    """
    Run predictions on DFC2023Amini dataset using a trained Im2Height model
    
    Args:
        dataset_path: Path to the DFC2023Amini dataset
        output_dir: Directory to save predictions
        weights: Path to the trained model weights
        split: Dataset split to run predictions on ('test', 'valid', or 'train')
        input_type: Input data type ('rgb' or 'sar')
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize dataset
    prediction_dataset = DFC2023PredictionDataset(dataset_path, split, input_type)
    
    # Set input channels based on input_type - don't try to detect from data
    # RGB images use 3 channels, SAR images use 1 channel
    in_channels = 3 if input_type == 'rgb' else 1
    
    print(f"Using {in_channels} input channels for prediction based on input_type '{input_type}'.")
    
    # Load the trained model
    try:
        # Check if the weights file exists
        if not os.path.exists(weights):
            print(f"Warning: Specified weights file {weights} doesn't exist.")
            
            # Try to find a suitable checkpoint in the weights directory
            weights_dir = os.path.dirname(weights)
            if os.path.isdir(weights_dir):
                possible_checkpoints = [os.path.join(weights_dir, f) for f in os.listdir(weights_dir) 
                                       if f.endswith('.ckpt') and os.path.isfile(os.path.join(weights_dir, f))]
                
                if possible_checkpoints:
                    # Sort by modification time (newest first)
                    possible_checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                    weights = possible_checkpoints[0]
                    print(f"Using most recent checkpoint: {weights}")
                else:
                    raise FileNotFoundError(f"No checkpoint files found in {weights_dir}")
            else:
                raise FileNotFoundError(f"Weights directory {weights_dir} doesn't exist")
        
        model = Im2Height.load_from_checkpoint(weights)
        print(f"Successfully loaded model from {weights}")
        
        # Check if the model's input channels match the data
        if hasattr(model, 'in_channels') and model.in_channels != in_channels:
            print(f"Warning: Model was trained with {model.in_channels} channels, but input has {in_channels} channels.")
            print("Creating a new model with the correct number of input channels.")
            model = Im2Height(in_channels=in_channels, out_channels=1)
            # Load weights manually, skipping the first conv layer
            checkpoint = torch.load(weights)
            model_dict = model.state_dict()
            
            # Get the state dict from the checkpoint
            checkpoint_state_dict = checkpoint.get('state_dict', checkpoint)
            
            # Filter out the incompatible layers from the loaded weights
            filtered_dict = {k: v for k, v in checkpoint_state_dict.items() 
                           if k in model_dict and model_dict[k].shape == v.shape}
            
            print(f"Compatible parameters: {len(filtered_dict)}/{len(model_dict)}")
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict, strict=False)
            
            # Verify the model architecture and input shape
            print(f"Model input channels: {model.in_channels}")
            print(f"Model first convolutional layer weights shape: {model.conv1.conv1.weight.shape}")
            print("Created new model and transferred compatible weights")
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Creating a new model with {in_channels} input channels.")
        model = Im2Height(in_channels=in_channels, out_channels=1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Set model to evaluation mode
    model.eval()

    # Initialize data loader with smaller batch size to avoid memory issues
    loader_config = load_config.copy()
    loader_config["batch_size"] = 8  # Reduce batch size to prevent potential memory issues
    
    data_loader = torch.utils.data.DataLoader(
        prediction_dataset, 
        **loader_config
    )

    # Define a function to inspect tensor details
    def inspect_tensor(tensor, name=""):
        print(f"\n{name} Tensor inspection:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Type: {tensor.dtype}")
        print(f"  Device: {tensor.device}")
        print(f"  Min/Max values: {tensor.min().item():.4f}/{tensor.max().item():.4f}")
        print(f"  Mean/Std: {tensor.mean().item():.4f}/{tensor.std().item():.4f}")
        
        # Check if channels dimension is correct based on input_type
        if tensor.shape[1] != in_channels and tensor.shape[0] != in_channels:
            print(f"  WARNING: Neither dimension matches expected channels ({in_channels})")
            # Try to fix tensor dimensions if needed
            if tensor.shape[0] > 100:  # Likely the image height/width, not channels
                print("  Attempting to correct tensor dimensions...")
                if in_channels == 3:  # RGB
                    corrected = tensor[:3].unsqueeze(0)  # Take first 3 channels and add batch dim
                    print(f"  Corrected shape: {corrected.shape}")
                    return corrected
                else:  # SAR (1 channel)
                    corrected = tensor[:1].unsqueeze(0)  # Take first channel and add batch dim
                    print(f"  Corrected shape: {corrected.shape}")
                    return corrected
        return tensor
    
    # Run predictions and save results
    for i, (filenames, tensors) in enumerate(data_loader):
        try:
            # Print info about the first few batches for debugging
            if i < 2:
                print(f"\nBatch {i+1}")
                print(f"Filenames: {[os.path.basename(f) for f in filenames]}")
                print(f"Input tensors shape: {tensors.shape}")
                
                # Check the first tensor in detail
                inspect_tensor(tensors[0], "First sample")
            
            with torch.no_grad():
                tensors = tensors.to(device)
                
                # For the first few batches, verify tensor shape and adjust if needed
                if i < 2 and (tensors.shape[1] != in_channels):
                    print(f"WARNING: Input tensor has wrong channel dimension: {tensors.shape}")
                    # Create a new tensor with the right dimensions
                    if in_channels == 3:  # RGB
                        tensors = tensors[:, :3]  # Keep only first 3 channels
                    else:  # SAR
                        tensors = tensors[:, :1]  # Keep only first channel
                    print(f"Corrected shape: {tensors.shape}")
                
                predictions = model(tensors)
                print(f"Output predictions shape: {predictions.shape}")
            
            for filename, img in zip(filenames, predictions.cpu().detach().numpy()):
                # Create filename based on the input filename
                output_filename = os.path.basename(filename).rsplit('.', 1)[0] + '_pred.npy'
                output_path = os.path.join(output_dir, output_filename)
                
                # Save prediction
                np.save(output_path, img[0])
                print(f"Saved prediction to {output_path}")
        
        except Exception as e:
            print(f"Error processing batch {i+1}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run predictions on DFC2023Amini dataset using Im2Height model")
    parser.add_argument("-d", "--dataset_path", type=str, required=True,
                        help="Path to the DFC2023Amini dataset")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Directory to save predictions")
    parser.add_argument("-w", "--weights", type=str, required=True,
                        help="Path to the trained model weights")
    parser.add_argument("-s", "--split", type=str, default="test", choices=["train", "valid", "test"],
                        help="Dataset split to run predictions on")
    parser.add_argument("-i", "--input_type", type=str, default="rgb", choices=["rgb", "sar"],
                        help="Input data type")
                        
    args = parser.parse_args()
    run(**vars(args))
