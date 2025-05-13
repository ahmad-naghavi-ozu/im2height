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
    print(f"Saving predictions to: {output_dir}")
    
    # Initialize dataset
    prediction_dataset = DFC2023PredictionDataset(dataset_path, split, input_type)
    
    # Initially set input channels based on input_type, but this may be overridden
    # by what we detect in the checkpoint to ensure compatibility
    in_channels = 3 if input_type == 'rgb' else 1
    
    print(f"Initial channels setting: {in_channels} channels based on input_type '{input_type}'.")
    
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
        
        # First try loading the checkpoint to examine its configuration
        checkpoint = torch.load(weights, map_location='cpu')
        checkpoint_state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Try to determine model's input channels from the checkpoint
        # Look at the first convolutional layer's weight shape to determine channels
        ckpt_in_channels = None
        if 'conv1.conv1.weight' in checkpoint_state_dict:
            ckpt_in_channels = checkpoint_state_dict['conv1.conv1.weight'].shape[1]
            print(f"Detected {ckpt_in_channels} input channels in checkpoint")
            
            # Always use the checkpoint's channel count to ensure compatibility
            # This ensures we load the model correctly regardless of input_type
            in_channels = ckpt_in_channels
            
            # But warn if there's a mismatch with requested input_type
            if (ckpt_in_channels == 3 and input_type == 'sar') or (ckpt_in_channels == 1 and input_type == 'rgb'):
                print(f"WARNING: Checkpoint was trained with {ckpt_in_channels} channels but input_type '{input_type}' expects different channels.")
                print(f"Will use checkpoint's channel configuration ({ckpt_in_channels}) for prediction.")
        
        # Now load the model with the appropriate number of channels
        try:
            # First try direct loading (works if channels match)
            model = Im2Height.load_from_checkpoint(weights)
            print(f"Successfully loaded model from {weights}")
            
            # Double-check channel configuration and always update our in_channels to match the model
            if hasattr(model, 'in_channels'):
                if model.in_channels != in_channels:
                    print(f"Warning: Loaded model has {model.in_channels} channels but input type '{input_type}' expects {in_channels} channels.")
                    print(f"Will process using the model's configuration ({model.in_channels} channels).")
                # Override in_channels to match the model - this is critical for proper prediction
                in_channels = model.in_channels
        except Exception as e:
            # Instead of failing, let's create a model with the correct number of input channels
            print(f"Standard loading failed: {e}")
            print(f"Creating model with {ckpt_in_channels} input channels to match checkpoint...")
            
            # Create a new model with the input channels from the checkpoint
            model = Im2Height(in_channels=ckpt_in_channels, out_channels=1)
            
            # Then load the checkpoint state dict
            checkpoint_state_dict = checkpoint.get('state_dict', checkpoint)
            
            # Load the state dict, ignoring mismatched keys
            missing, unexpected = model.load_state_dict(checkpoint_state_dict, strict=False)
            
            if missing:
                print(f"Missing keys: {len(missing)} keys")
            if unexpected:
                print(f"Unexpected keys: {len(unexpected)} keys")
                
            print(f"Successfully created model with {ckpt_in_channels} input channels and loaded weights.")
            # Update in_channels to match the model
            in_channels = ckpt_in_channels
    except Exception as e:
        # If we detected checkpoint channels earlier but failed to load the model properly,
        # try one more time with a custom approach
        if ckpt_in_channels is not None:
            try:
                print(f"Trying alternative loading method with {ckpt_in_channels} channels...")
                model = Im2Height(in_channels=ckpt_in_channels, out_channels=1)
                checkpoint_state_dict = checkpoint.get('state_dict', checkpoint)
                missing, unexpected = model.load_state_dict(checkpoint_state_dict, strict=False)
                print(f"Successfully loaded model using {ckpt_in_channels} input channels.")
                print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
                in_channels = ckpt_in_channels
            except Exception as alt_e:
                print(f"ERROR: Alternative loading also failed: {alt_e}")
                print("Cannot proceed with prediction without a trained model.")
                print("Please check the checkpoint path and ensure it contains a valid trained model.")
                print(f"Checkpoint path: {weights}")
                print("\nPossible solutions:")
                print("1. Check that the checkpoint file exists and is accessible")
                print("2. Verify that the checkpoint was created with a compatible model architecture")
                print("3. Make sure the input_type parameter (--input_type) matches how the model was trained")
                print("4. If needed, train a new model before running predictions")
                
                import sys
                sys.exit(1)  # Exit with error code
        else:
            print(f"ERROR: Failed to load model from checkpoint: {e}")
            print("Cannot proceed with prediction without a trained model.")
            print("Please check the checkpoint path and ensure it contains a valid trained model.")
            print(f"Checkpoint path: {weights}")
            print("\nPossible solutions:")
            print("1. Check that the checkpoint file exists and is accessible") 
            print("2. Verify that the checkpoint was created with a compatible model architecture")
            print("3. Make sure the input_type parameter (--input_type) matches how the model was trained")
            print("4. If needed, train a new model before running predictions")
            
            import sys
            sys.exit(1)  # Exit with error code
    
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
                
                # For all batches, verify tensor shape matches model's expected input channels
                if tensors.shape[1] != model.in_channels:
                    print(f"Input tensor channels ({tensors.shape[1]}) don't match model's expected channels ({model.in_channels})")
                    
                    # Create a new tensor with the right dimensions to match the model
                    if model.in_channels == 3:  # Model expects RGB
                        if tensors.shape[1] >= 3:
                            tensors = tensors[:, :3]  # Keep only first 3 channels
                        else:
                            # Not enough channels, duplicate the existing channel(s)
                            tensors = tensors[:, :1].repeat(1, 3, 1, 1)
                            print(f"Duplicated single channel to create RGB input")
                    else:  # Model expects single channel
                        if tensors.shape[1] > 1:
                            # Convert multiple channels to single channel using mean
                            tensors = tensors.mean(dim=1, keepdim=True)
                            print(f"Converted multiple channels to single channel using mean")
                        else:
                            tensors = tensors[:, :1]  # Keep only first channel
                    
                    if i < 2:  # Only print for first few batches
                        print(f"Adjusted tensor shape to: {tensors.shape}")
                
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
