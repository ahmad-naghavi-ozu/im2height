"""
Contains unified dataset classes for multiple dataset formats
Supports both NPY files (original Im2Height) and image files (DFC2023, etc.)
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


class UnifiedDataset(torch.utils.data.Dataset):
	'''
	A unified dataset class that automatically detects and handles different dataset formats:
	- NPY format (original Im2Height): separate x/ and y/ directories with .npy files
	- Image format (DFC2023, etc.): structured directories with image files and DSM data
	
	Supports dynamic channel detection (1, 3, or N channels) for any input format.
	'''
	
	def __init__(self, dataset_path, split='train', input_type='rgb', target_type='dsm'):
		"""
		Initialize unified dataset with automatic format detection.
		
		:param dataset_path: (str) path to dataset - can be:
			- NPY format: path containing train/x, train/y, test/x, test/y directories
			- Image format: path containing train/rgb, train/dsm, etc. directories
		:param split: (str) 'train', 'valid', or 'test'
		:param input_type: (str) input modality ('rgb', 'sar', etc.) - only used for image format
		:param target_type: (str) target modality ('dsm', etc.) - only used for image format
		"""
		
		self.dataset_path = dataset_path
		self.split = split
		self.input_type = input_type
		self.target_type = target_type
		self.dataset_format = self._detect_format()
		
		# Initialize based on detected format
		if self.dataset_format == 'npy':
			self._init_npy_format()
		else:  # image format
			self._init_image_format()
			
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
		
		# Check for image format: look for input_type and target_type subdirectories
		img_input_path = os.path.join(self.dataset_path, self.split, self.input_type)
		img_target_path = os.path.join(self.dataset_path, self.split, self.target_type)
		
		if os.path.exists(img_input_path) and os.path.exists(img_target_path):
			return 'image'
		
		# Fallback: assume image format if split directory exists
		split_path = os.path.join(self.dataset_path, self.split)
		if os.path.exists(split_path):
			return 'image'
		
		raise ValueError(f"Could not detect dataset format for {self.dataset_path}")
	
	def _init_npy_format(self):
		"""Initialize for NPY format dataset."""
		self.x_dir = os.path.join(self.dataset_path, self.split, 'x')
		self.y_dir = os.path.join(self.dataset_path, self.split, 'y')
		
		# Get sorted file lists
		self.x_list = np.sort(os.listdir(self.x_dir))
		self.y_list = np.sort(os.listdir(self.y_dir))
		
		print(f"Detected NPY format dataset: {len(self.x_list)} samples")
	
	def _init_image_format(self):
		"""Initialize for image format dataset with optional NPY conversion."""
		self.input_dir = os.path.join(self.dataset_path, self.split, self.input_type)
		self.target_dir = os.path.join(self.dataset_path, self.split, self.target_type)
		
		# Get sorted file lists
		self.input_files = sorted([f for f in os.listdir(self.input_dir) 
								  if os.path.isfile(os.path.join(self.input_dir, f))])
		self.target_files = sorted([f for f in os.listdir(self.target_dir) 
								   if os.path.isfile(os.path.join(self.target_dir, f))])
		
		# Set up NPY cache directories for faster loading
		dataset_name = os.path.basename(os.path.normpath(self.dataset_path))
		script_dir = os.path.dirname(os.path.abspath(__file__))
		self.data_dir = os.path.join(script_dir, "data", dataset_name)
		self.input_npy_dir = os.path.join(self.data_dir, self.split, "x")
		self.target_npy_dir = os.path.join(self.data_dir, self.split, "y")
		os.makedirs(self.input_npy_dir, exist_ok=True)
		os.makedirs(self.target_npy_dir, exist_ok=True)
		
		print(f"Detected image format dataset: {len(self.input_files)} samples")
		if len(self.input_files) != len(self.target_files):
			print(f"Warning: Input files ({len(self.input_files)}) != target files ({len(self.target_files)})")
	
	def __len__(self):
		if self.dataset_format == 'npy':
			return len(self.x_list)
		else:
			return len(self.input_files)
	
	def __getitem__(self, idx: int) -> tuple:
		if self.dataset_format == 'npy':
			return self._get_npy_item(idx)
		else:
			return self._get_image_item(idx)
	
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
	
	def _get_image_item(self, idx: int) -> tuple:
		"""Get item from image format dataset with NPY caching."""
		input_path = os.path.join(self.input_dir, self.input_files[idx])
		file_basename = os.path.splitext(os.path.basename(input_path))[0]
		
		# Check NPY cache first
		input_npy_path = os.path.join(self.input_npy_dir, f"{file_basename}.npy")
		target_npy_path = os.path.join(self.target_npy_dir, f"{file_basename}.npy")
		
		# Load or convert input
		if os.path.exists(input_npy_path):
			img = np.load(input_npy_path)
		else:
			img = self._load_and_cache_image(input_path, input_npy_path)
		
		# Load or convert target
		target_path = os.path.join(self.target_dir, self.target_files[idx])
		if os.path.exists(target_npy_path):
			label = np.load(target_npy_path)
		else:
			label = self._load_and_cache_image(target_path, target_npy_path)
		
		# Ensure proper channel dimensions
		if len(img.shape) == 2:
			img = np.expand_dims(img, axis=2)
		if len(label.shape) == 2:
			label = np.expand_dims(label, axis=2)
		
		# Normalize label
		label = label - label.min()
		
		# Add padding if needed
		padding = 0
		img = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), "reflect")
		label = np.pad(label, ((padding, padding), (padding, padding), (0, 0)), "reflect")
		
		# Handle channel order - convert to channels last for augmentation
		if img.shape[0] == 1 or img.shape[0] == 3:  # If channels first
			img = np.transpose(img, (1, 2, 0))
		if label.shape[0] == 1:  # If channels first
			label = np.transpose(label, (1, 2, 0))
		
		# Apply augmentations
		img, label = self.augmenter(img, label)
		
		# Ensure contiguous arrays
		img = np.ascontiguousarray(img)
		label = np.ascontiguousarray(label)
		
		# Convert to PyTorch tensors (channels first)
		img_tensor = torch.Tensor(img).permute((2, 0, 1))
		label_tensor = torch.Tensor(label).permute((2, 0, 1))
		
		return img_tensor, label_tensor
	
	def _load_and_cache_image(self, image_path, npy_path):
		"""Load image and cache as NPY for faster future loading."""
		if image_path.endswith(('.jpg', '.png', '.tif', '.tiff')):
			img = np.array(Image.open(image_path))
		else:  # Assume it's already a numpy file
			img = np.load(image_path)
		
		# Cache the converted image
		np.save(npy_path, img)
		return img
	
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


class UnifiedPredictionDataset(torch.utils.data.Dataset):
	'''
	A unified prediction dataset that handles both NPY and image formats.
	'''
	
	def __init__(self, dataset_path, split='test', input_type='rgb'):
		"""
		Initialize unified prediction dataset.
		
		:param dataset_path: (str) path to dataset or list of files
		:param split: (str) dataset split ('test', 'valid', etc.)
		:param input_type: (str) input modality for image format datasets
		"""
		
		# Handle both directory paths and file lists
		if isinstance(dataset_path, list):
			# Direct file list provided
			self.files = dataset_path
			self.dataset_format = 'file_list'
		else:
			# Directory path provided
			self.dataset_path = dataset_path
			self.split = split
			self.input_type = input_type
			self.dataset_format = self._detect_format()
			self._init_file_list()
	
	def _detect_format(self):
		"""Detect dataset format for prediction."""
		# Check for NPY format
		npy_x_path = os.path.join(self.dataset_path, self.split, 'x')
		if os.path.exists(npy_x_path):
			return 'npy'
		
		# Check for image format
		img_input_path = os.path.join(self.dataset_path, self.split, self.input_type)
		if os.path.exists(img_input_path):
			return 'image'
		
		raise ValueError(f"Could not detect dataset format for prediction: {self.dataset_path}")
	
	def _init_file_list(self):
		"""Initialize file list based on detected format."""
		if self.dataset_format == 'npy':
			x_dir = os.path.join(self.dataset_path, self.split, 'x')
			self.files = sorted([os.path.join(x_dir, f) for f in os.listdir(x_dir)])
		else:  # image format
			input_dir = os.path.join(self.dataset_path, self.split, self.input_type)
			self.files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) 
								if os.path.isfile(os.path.join(input_dir, f))])
	
	def __len__(self):
		return len(self.files)
	
	def __getitem__(self, idx: int) -> tuple:
		file_path = self.files[idx]
		
		# Load image based on format
		if file_path.endswith('.npy'):
			img = np.rollaxis(np.load(file_path), 0, 3)
		else:
			img = np.array(Image.open(file_path))
			if len(img.shape) == 2:
				img = np.expand_dims(img, axis=2)
		
		# Add padding
		padding = 3
		img = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), "reflect")
		
		# Convert to tensor
		img_tensor = torch.Tensor(img).permute((2, 0, 1))
		
		return file_path, img_tensor
