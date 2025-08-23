"""
Contains dataset classes for NPY format datasets.
Training and prediction ONLY work with NPY format - image datasets must be preprocessed first.
"""

import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageFile
from albumentations import HorizontalFlip, VerticalFlip, Rotate, RandomRotate90, RandomBrightnessContrast, GaussNoise
from augmenter import Augmenter
from scipy.ndimage import gaussian_gradient_magnitude

# Ensure PIL can handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


def prediction_collate_fn(batch):
	"""
	Custom collate function for prediction data.
	Handles batching of (file_path, tensor) tuples.
	"""
	file_paths, tensors = zip(*batch)
	# Stack tensors into a batch
	tensor_batch = torch.stack(tensors, dim=0)
	return list(file_paths), tensor_batch


class NpyDataset(torch.utils.data.Dataset):
	'''
	A supervised learning dataset class to handle serialised
	numpy data, for example images.

	Data consists of float `.npy` files of fixed shape.
	Observations and labels are given by different folders
	containing files with same names.
	'''
	def __init__(self, x_dir, y_dir):
		"""
		Instantiate .npy file dataset.

		:param x_dir: (str) observation directory
		:param y_dir: (str) label directory
		"""

		self.x_dir = x_dir
		self.y_dir = y_dir

		# sort is needed for order in data
		self.x_list = np.sort(os.listdir(x_dir))
		self.y_list = np.sort(os.listdir(y_dir))

		transforms = [
			VerticalFlip(p=.2),
			HorizontalFlip(p=.2),
			RandomRotate90(p=.3)]
			#RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=.2),
			#GaussNoise(var_limit=(0.0, 20.0), p=.2)]

		self.augmenter = Augmenter(list_of_transforms=transforms, p=.9)


	def __len__(self):
		return len(self.x_list)

	def __getitem__(self, idx: int) -> tuple:

		img_name = os.path.join(self.x_dir, self.x_list[idx])
		img = np.rollaxis(np.load(img_name), 0, 3)
		#print("img", img.shape)

		padding = 0
		img = np.pad(img, ((padding,padding),(padding,padding),(0,0)), "reflect") # pad to reach side of 2**n

		label_name = os.path.join(self.y_dir, self.y_list[idx])
		label = np.rollaxis(np.load(label_name), 0, 3)
		label = label-label.min()
		#print("label", label.shape)

		label = np.pad(label, ((padding,padding),(padding,padding),(0,0)), "reflect")

		# albumentations needs channel last
		img, label = self.augmenter(img, label)

		# pytorch needs channels first
		img_tensor = torch.Tensor(img).permute((2, 0, 1))
		label_tensor = torch.Tensor(label).permute((2, 0, 1))

		return img_tensor, label_tensor


class NpyPredictionDataset(torch.utils.data.Dataset):
	'''
	A dataset class to handle prediction on serialised numpy data,
	for example images.

	Data consists of float `.npy` files of fixed shape.
	'''
	def __init__(self, files):
		"""
		Instantiate .npy file dataset.

		:param files: (list) list of files to predict on
		"""

		self.files = files

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx: int) -> tuple:
		padding = 3
		img = np.rollaxis(np.load(self.files[idx]), 0, 3)
		img = np.pad(img, ((padding,padding),(padding,padding),(0,0)), "reflect")
		img = torch.Tensor(img).permute((2, 0, 1))
		return self.files[idx], img


class Dataset(torch.utils.data.Dataset):
	'''
	A dataset class that works with NPY format datasets only.
	Image format datasets must be preprocessed to NPY format first using preprocess.py.
	
	Expected structure:
	- NPY format: dataset_path/train/x/, dataset_path/train/y/ directories with .npy files
	
	Any attempt to use image format datasets will result in an error with instructions 
	to run preprocessing first.
	'''
	
	def __init__(self, dataset_path, split='train'):
		"""
		Initialize dataset for RGB to DSM conversion (NPY format only).
		
		:param dataset_path: (str) path to dataset - NPY format with train/x, train/y directories
		:param split: (str) 'train', 'valid', or 'test'
		"""
		
		self.dataset_path = dataset_path
		self.split = split
		self.dataset_format = self._detect_format()
		
		# Initialize based on detected format
		if self.dataset_format == 'npy':
			self._init_npy_format()
		else:
			# This should never happen due to the error checking in _detect_format()
			raise ValueError("Only NPY format is supported for training and prediction")
			
		# Set up augmentations
		transforms = [
			VerticalFlip(p=.2),
			HorizontalFlip(p=.2),
			RandomRotate90(p=.3)
		]
		# Only apply augmentations during training
		self.augmenter = Augmenter(list_of_transforms=transforms, p=.9 if split == 'train' else 0)
	
	def _detect_format(self):
		"""Automatically detect dataset format based on directory structure."""
		
		# Check for NPY format: look for x/ and y/ subdirectories in split folder
		npy_x_path = os.path.join(self.dataset_path, self.split, 'x')
		npy_y_path = os.path.join(self.dataset_path, self.split, 'y')
		
		if os.path.exists(npy_x_path) and os.path.exists(npy_y_path):
			return 'npy'
		
		# Check for image format: look for rgb and dsm subdirectories
		img_input_path = os.path.join(self.dataset_path, self.split, 'rgb')
		img_target_path = os.path.join(self.dataset_path, self.split, 'dsm')
		
		if os.path.exists(img_input_path) and os.path.exists(img_target_path):
			# Image format detected - this is not supported for training/prediction
			raise ValueError(
				f"\n❌ ERROR: Image format dataset detected, but training/prediction requires NPY format.\n"
				f"📁 Found: {img_input_path} and {img_target_path}\n"
				f"🔧 SOLUTION: Run preprocessing first:\n"
				f"   ./run.sh --action preprocess --dataset {self.dataset_path}\n"
				f"   OR\n"
				f"   python preprocess.py -d {self.dataset_path}\n"
				f"⚠️  Training and prediction ONLY work with NPY format files."
			)
		
		# Fallback: assume image format if split directory exists
		split_path = os.path.join(self.dataset_path, self.split)
		if os.path.exists(split_path):
			raise ValueError(
				f"\n❌ ERROR: No NPY format detected for dataset: {self.dataset_path}\n"
				f"📁 Looking for: {npy_x_path} and {npy_y_path}\n"
				f"🔧 SOLUTION: Run preprocessing first:\n"
				f"   ./run.sh --action preprocess --dataset {self.dataset_path}\n"
				f"⚠️  Training and prediction ONLY work with NPY format files."
			)
		
		raise ValueError(f"Could not detect dataset format for {self.dataset_path}")
	
	def _init_npy_format(self):
		"""Initialize for NPY format dataset."""
		self.x_dir = os.path.join(self.dataset_path, self.split, 'x')
		self.y_dir = os.path.join(self.dataset_path, self.split, 'y')
		
		# Get sorted file lists
		self.x_list = np.sort(os.listdir(self.x_dir))
		self.y_list = np.sort(os.listdir(self.y_dir))
		
		print(f"Detected NPY format dataset: {len(self.x_list)} samples")
	
	def __len__(self):
		# Only NPY format is supported
		return len(self.x_list)
	
	def __getitem__(self, idx: int) -> tuple:
		# Only NPY format is supported
		return self._get_npy_item(idx)
	
	def _get_npy_item(self, idx: int) -> tuple:
		"""Get item from NPY format dataset."""
		img_name = os.path.join(self.x_dir, self.x_list[idx])
		img = np.rollaxis(np.load(img_name), 0, 3)
		
		padding = 0
		img = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), "reflect")
		
		label_name = os.path.join(self.y_dir, self.y_list[idx])
		label = np.rollaxis(np.load(label_name), 0, 3)
		label = label - label.min()
		
		label = np.pad(label, ((padding, padding), (padding, padding), (0, 0)), "reflect")
		
		# Apply augmentations (albumentations needs channel last)
		img, label = self.augmenter(img, label)
		
		# Convert to PyTorch tensors (channels first)
		img_tensor = torch.Tensor(img).permute((2, 0, 1))
		label_tensor = torch.Tensor(label).permute((2, 0, 1))
		
		return img_tensor, label_tensor
	
	def get_input_channels(self):
		"""Get the number of input channels by examining a sample."""
		if len(self) == 0:
			return 1  # Default fallback
		
		sample_input, _ = self[0]
		return sample_input.shape[0]  # Channel dimension is first in PyTorch tensors
	
	def get_image_size(self):
		"""Get the image dimensions by examining a sample."""
		if len(self) == 0:
			return (256, 256)  # Default fallback
		
		sample_input, _ = self[0]
		return (sample_input.shape[2], sample_input.shape[1])  # (width, height)


class PredictionDataset(torch.utils.data.Dataset):
	'''
	A prediction dataset that handles NPY format only.
	Image format datasets must be preprocessed to NPY first.
	'''
	
	def __init__(self, dataset_path, split='test'):
		"""
		Initialize prediction dataset for RGB to DSM conversion (NPY format only).
		
		:param dataset_path: (str) path to dataset or list of files
		:param split: (str) dataset split ('test', 'valid', etc.)
		"""
		
		# Handle both directory paths and file lists
		if isinstance(dataset_path, list):
			# Direct file list provided - check if files are NPY
			self.files = dataset_path
			self.dataset_format = 'file_list'
			# Verify all files are NPY format
			non_npy_files = [f for f in self.files if not f.endswith('.npy')]
			if non_npy_files:
				raise ValueError(
					f"\n❌ ERROR: Non-NPY files detected in file list for prediction.\n"
					f"📁 Non-NPY files: {non_npy_files[:3]}{'...' if len(non_npy_files) > 3 else ''}\n"
					f"🔧 SOLUTION: Run preprocessing first to convert to NPY format.\n"
					f"⚠️  Prediction ONLY works with NPY format files."
				)
		else:
			# Directory path provided
			self.dataset_path = dataset_path
			self.split = split
			self.dataset_format = self._detect_format()
			self._init_file_list()
	
	def _detect_format(self):
		"""Detect dataset format for prediction (NPY only)."""
		# Check for NPY format
		npy_x_path = os.path.join(self.dataset_path, self.split, 'x')
		if os.path.exists(npy_x_path):
			return 'npy'
		
		# Check for image format - this is not supported
		img_input_path = os.path.join(self.dataset_path, self.split, 'rgb')
		if os.path.exists(img_input_path):
			raise ValueError(
				f"\n❌ ERROR: Image format dataset detected for prediction, but only NPY format is supported.\n"
				f"📁 Found: {img_input_path}\n"
				f"🔧 SOLUTION: Run preprocessing first:\n"
				f"   ./run.sh --action preprocess --dataset {self.dataset_path}\n"
				f"   OR\n"
				f"   python preprocess.py -d {self.dataset_path}\n"
				f"⚠️  Prediction ONLY works with NPY format files."
			)
		
		raise ValueError(
			f"\n❌ ERROR: No NPY format detected for prediction dataset: {self.dataset_path}\n"
			f"📁 Looking for: {npy_x_path}\n"
			f"🔧 SOLUTION: Run preprocessing first:\n"
			f"   ./run.sh --action preprocess --dataset {self.dataset_path}\n"
			f"⚠️  Prediction ONLY works with NPY format files."
		)
	
	def _init_file_list(self):
		"""Initialize file list (NPY format only)."""
		if self.dataset_format == 'npy':
			x_dir = os.path.join(self.dataset_path, self.split, 'x')
			self.files = sorted([os.path.join(x_dir, f) for f in os.listdir(x_dir) if f.endswith('.npy')])
		else:
			# This should never happen due to error checking in _detect_format()
			raise ValueError("Only NPY format is supported for prediction")
	
	def __len__(self):
		return len(self.files)
	
	def __getitem__(self, idx: int) -> tuple:
		file_path = self.files[idx]
		
		# Only NPY files are supported
		if not file_path.endswith('.npy'):
			raise ValueError(f"Only NPY files are supported for prediction. Got: {file_path}")
		
		# Load NPY file
		img = np.rollaxis(np.load(file_path), 0, 3)
		
		# Keep consistent with training data (no padding)
		padding = 0
		img = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), "reflect")
		
		# Convert to tensor
		img_tensor = torch.Tensor(img).permute((2, 0, 1))
		
		return file_path, img_tensor
