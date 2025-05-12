import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from nadam import Nadam
from ssim import ssim



class Pool(LightningModule):

	def __init__(self, kernel_size=2, stride=2, **kwargs):

		super(Pool, self).__init__()

		self.pool_fn = nn.MaxPool2d(kernel_size, stride, **kwargs)

	def forward(self, x, *args, **kwargs):

		size = x.size()
		x, indices = self.pool_fn(x, **kwargs)

		return x, indices, size


class Unpool(LightningModule):

	def __init__(self, fn, kernel_size=2, stride=2, **kwargs):

		super(Unpool, self).__init__()

		self.pool_fn = nn.MaxUnpool2d(kernel_size, stride, **kwargs)

	def forward(self, x, indices, output_size, *args, **kwargs):

		return self.pool_fn(x, indices=indices, output_size=output_size, *args, **kwargs)

class Block(LightningModule):
	""" A Block performs three rounds of conv, batchnorm, relu
	"""
	def __init__(self, fn, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
		super(Block, self).__init__()

		self.conv1 = fn(in_channels, out_channels, kernel_size, stride, padding)
		self.conv_rest = fn(out_channels, out_channels, kernel_size, stride, padding)
		self.bn = nn.BatchNorm2d(out_channels)
		# following similar setup https://github.com/hysts/pytorch_resnet
		self.identity = nn.Sequential()  # identity
		if in_channels != out_channels:
			self.identity.add_module(
				'conv',
				nn.Conv2d(
					in_channels,
					out_channels,
					kernel_size=1,
					stride=stride,  # downsample
					padding=0,
					bias=False))
			self.identity.add_module('bn', nn.BatchNorm2d(out_channels))  # BN


	def forward(self, x):

		y = F.relu(self.bn(self.conv1(x)))
		y = F.relu(self.bn(self.conv_rest(y)))
		y = self.bn(self.conv_rest(y))
		identity = self.identity(x)
		y = F.relu(y + identity)

		return y


class Im2Height(LightningModule):
	""" Im2Height Fully Residual Convolutional-Deconvolutional Network
		implementation based on https://arxiv.org/abs/1802.10249
		
		Modified to support dynamic input channels (1 or 3) for both grayscale and RGB inputs
		while maintaining backward compatibility with original implementation
	"""

	def __init__(self, in_channels=1, out_channels=1):

		super(Im2Height, self).__init__()
		
		# Save number of input channels
		self.in_channels = in_channels
		self.out_channels = out_channels

		# Convolutions
		self.conv1 = Block(nn.Conv2d, in_channels, 64)
		self.conv2 = Block(nn.Conv2d, 64, 128)
		self.conv3 = Block(nn.Conv2d, 128, 256)
		self.conv4 = Block(nn.Conv2d, 256, 512)

		# Deconvolutions
		self.deconv1 = Block(nn.ConvTranspose2d, 512, 256)
		self.deconv2 = Block(nn.ConvTranspose2d, 256, 128)
		self.deconv3 = Block(nn.ConvTranspose2d, 128, 64)
		self.deconv4 = Block(nn.ConvTranspose2d, 128, out_channels) # note this is residual merge

		self.pool = Pool(2, 2, return_indices=True)
		self.unpool = Unpool(2, 2)


	def forward(self, x):

		# Convolve
		x = self.conv1(x)
		# Residual skip connection
		x_conv_input = x.clone()
		x, indices1, size1 = self.pool(x)
		x, indices2, size2 = self.pool(self.conv2(x))
		x, indices3, size3 = self.pool(self.conv3(x))
		x, indices4, size4 = self.pool(self.conv4(x))

		# Deconvolve
		x = self.unpool(x, indices4, indices3.size())
		x = self.deconv1(x)
		x = self.unpool(x, indices3, indices2.size())
		x = self.deconv2(x)
		x = self.unpool(x, indices2, indices1.size())
		x = self.deconv3(x)
		x = self.unpool(x, indices1, x_conv_input.size())

		# Concatenate with residual skip connection
		x = torch.cat((x, x_conv_input), dim=1)
		x = self.deconv4(x)

		return x


	# lightning implementations
	def training_step(self, batch, batch_idx):

		x, y = batch
		y_pred = self(x)
		l1loss = F.l1_loss(y_pred, y)
		l2loss = F.mse_loss(y_pred, y)
		tensorboard_logs = { 'l1loss': l1loss, 'l2loss': l2loss }

		return { 'loss': l1loss, 'log': tensorboard_logs }

	def configure_optimizers(self):

		return Nadam(self.parameters(), lr=2e-5, schedule_decay=4e-3)
		#return torch.optim.SGD(self.parameters(), lr=1e-3)


	# validation
	def validation_step(self, batch, batch_idx):

		x, y = batch
		y_pred = self(x)

		l1loss = F.l1_loss(y_pred, y)
		l2loss = F.mse_loss(y_pred, y)
		ssim_loss = ssim(y_pred, y)

		tensorboard_logs = { 'val_l1loss': l1loss, 'val_l2loss': l2loss, 'val_ssimloss': ssim_loss }

		return tensorboard_logs

	def on_validation_epoch_end(self):
		# Get outputs from validation steps
		outputs = self.trainer.callback_metrics
		
		# Log metrics for the epoch
		self.log("val_l1loss", outputs.get("val_l1loss", 0))
		self.log("val_l2loss", outputs.get("val_l2loss", 0))
		self.log("val_ssimloss", outputs.get("val_ssimloss", 0))



if __name__ == "__main__":
	net = Im2Height()
	print(net)
