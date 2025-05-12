import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from im2height import Im2Height
from data import NpyPredictionDataset


load_config = {
	"batch_size": 32,
	"pin_memory": True,
	"num_workers": 32
}

def run(input, output, weights):

	# Initialize dataset to check input shape
	prediction_dataset = NpyPredictionDataset(input)
	
	# Check if we have any samples to determine input channels
	if len(prediction_dataset) > 0:
		_, sample_tensor = prediction_dataset[0]
		in_channels = sample_tensor.shape[0]
	else:
		in_channels = 1  # Default to 1 channel if no samples
	
	# load weights and manually set in_channels if needed
	try:
		model = Im2Height.load_from_checkpoint(weights)
		# Check if the model's input channels match the data
		if model.in_channels != in_channels:
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

	data_loader = torch.utils.data.DataLoader(prediction_dataset, **load_config)

	# predict and store
	for filenames, tensors in data_loader:
		
		with torch.no_grad():
			tensors = tensors.to(device)
			predictions = model(tensors)
		
		for filename, img in zip(filenames, predictions.cpu().detach().numpy()):
			np.save(f"{output}/{os.path.basename(filename)}", img[0])



if __name__ == '__main__':

	DESCRIPTION = """
	Command line interface for batch compatible generic model prediction.

	Usage:
		$ python predict.py -i path/to/my/files/*.npy -o my/output/path -w pth/to/weight.ckpt

	Performs predictions for all .npy files obtained through shell globbing
	and serialises the outputs as specified in the main routine below.
	"""

	parser = argparse.ArgumentParser(description=DESCRIPTION)
	parser.add_argument("-i", "--input", type=str, help="Input file paths", required=True, nargs="+")
	parser.add_argument("-o", "--output", type=str, help="Output directory", required=True)
	parser.add_argument("-w", "--weights", type=str, help="Weights path", required=True)
	args = parser.parse_args()
	run(**vars(args))
