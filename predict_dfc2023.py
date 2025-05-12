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
    
    # Check if we have any samples to determine input channels
    if len(prediction_dataset) > 0:
        _, sample_tensor = prediction_dataset[0]
        in_channels = sample_tensor.shape[0]
    else:
        in_channels = 3 if input_type == 'rgb' else 1  # Default based on input_type
    
    print(f"Using {in_channels} input channels for prediction.")
    
    # Load the trained model
    try:
        model = Im2Height.load_from_checkpoint(weights)
        # Check if the model's input channels match the data
        if hasattr(model, 'in_channels') and model.in_channels != in_channels:
            print(f"Warning: Model was trained with {model.in_channels} channels, but input has {in_channels} channels.")
            print("Creating a new model with the correct number of input channels.")
            model = Im2Height(in_channels=in_channels, out_channels=1)
            # Load weights manually, skipping the first conv layer
            checkpoint = torch.load(weights)
            model_dict = model.state_dict()
            # Filter out the first convolution layer from the loaded weights
            filtered_dict = {k: v for k, v in checkpoint['state_dict'].items() if 'conv1.conv.weight' not in k and 'conv1.conv.bias' not in k}
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict, strict=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Creating a new model with {in_channels} input channels.")
        model = Im2Height(in_channels=in_channels, out_channels=1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Set model to evaluation mode
    model.eval()

    # Initialize data loader
    data_loader = torch.utils.data.DataLoader(
        prediction_dataset, 
        **load_config
    )

    # Run predictions and save results
    for filenames, tensors in data_loader:
        with torch.no_grad():
            tensors = tensors.to(device)
            predictions = model(tensors)
        
        for filename, img in zip(filenames, predictions.cpu().detach().numpy()):
            # Create filename based on the input filename
            output_filename = os.path.basename(filename).rsplit('.', 1)[0] + '_pred.npy'
            output_path = os.path.join(output_dir, output_filename)
            
            # Save prediction
            np.save(output_path, img[0])
            print(f"Saved prediction to {output_path}")


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
