import os
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from im2height import Im2Height
from data import NpyDataset


# load data
load_config = {
	"batch_size": 6,
	"pin_memory": True,
	"num_workers": 12
}


def run():

	#torch.multiprocessing.freeze_support()
	train_dataset = NpyDataset('data/train/x', 'data/train/y')
	test_dataset = NpyDataset('data/test/x', 'data/test/y')
	
	train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **load_config)
	test_loader = torch.utils.data.DataLoader(test_dataset, **load_config)

	# For backward compatibility:
	# Original implementation used single-channel inputs, so we'll check
	# if we need to update for multi-channel support
	sample_input, _ = train_dataset[0]
	in_channels = sample_input.shape[0]  # Channel dimension is first in PyTorch tensors
	
	print(f"Detected input with {in_channels} channel(s)")
	
	# training
	model = Im2Height(in_channels=in_channels, out_channels=1)

	trainer = Trainer(
		gpus=torch.cuda.device_count(),
		num_nodes=1,
		default_root_dir='weights/',
		max_epochs=1000,
		early_stop_callback=EarlyStopping(
			monitor='val_l1loss',
			patience=200,
			verbose=False,
			mode='min'
		),
		checkpoint_callback=ModelCheckpoint(
			filepath='weights/best_run.ckpt',
			save_top_k=5,
			verbose=True,
			monitor='val_l1loss',
			mode='min',
			save_last=True
			#prefix=''
		)
	)

	trainer.fit(model, train_loader, test_loader)


if __name__ == '__main__':
	run()
