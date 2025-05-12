# Running Im2Height on DFC2023Amini Dataset

This document provides instructions on how to run the Im2Height model on the DFC2023Amini dataset structure.

## Dataset Structure

The DFC2023Amini dataset has the following structure:

```
DFC2023Amini/
├── test/
│   ├── dsm/  (Digital Surface Model data - elevation information)
│   ├── rgb/  (RGB optical imagery)
│   ├── sar/  (Synthetic Aperture Radar imagery)
│   └── sem/  (Semantic segmentation masks/labels)
├── train/
│   ├── dsm/
│   ├── rgb/
│   ├── sar/
│   └── sem/
└── valid/
    ├── dsm/
    ├── rgb/
    ├── sar/
    └── sem/
```

For this implementation:
- RGB imagery is used as input (`rgb` directory), with SAR as a possible alternative input
- Digital Surface Models (DSM) are used as targets (`dsm` directory)
- Semantic segmentation data (`sem` directory) is not used in this implementation

## Preprocessing the Dataset

The Im2Height model expects input data in .npy format. You can use the `preprocess_dfc2023.py` script to convert the dataset to the required format:

```
python preprocess_dfc2023.py --dataset_path /path/to/DFC2023Amini --output_path /path/to/processed_dataset
```

This will create a new directory structure compatible with the original implementation:

```
processed_dataset/
├── train/
│   ├── x/ (input data as .npy files)
│   └── y/ (target data as .npy files)
├── valid/
│   ├── x/
│   └── y/
└── test/
    ├── x/
    └── y/
```

### Preprocessing options

- `--dataset_path`: Path to the DFC2023Amini dataset (required)
- `--output_path`: Path to save the processed .npy files (required)
- `--input_type`: Input data type to use (`rgb` or `sar`, default: `rgb`)
- `--target_type`: Target data type to use (default: `dsm`)

## Training

You have two options for training the model:

### Option 1: Using the original training script with preprocessed data

After preprocessing the data, you can use the original training script:

```
python train.py
```

Make sure to update the paths in the script if your preprocessed data is not in the default `data/` directory:

```python
train_loader = torch.utils.data.DataLoader(NpyDataset('path/to/processed_dataset/train/x', 'path/to/processed_dataset/train/y'), shuffle=True, **load_config)
test_loader = torch.utils.data.DataLoader(NpyDataset('path/to/processed_dataset/test/x', 'path/to/processed_dataset/test/y'), **load_config)
```

### Option 2: Using the DFC2023-specific training script

Alternatively, you can use the `train_dfc2023.py` script that handles the dataset conversion on-the-fly:

```
python train_dfc2023.py --dataset_path /path/to/DFC2023Amini --output_dir weights/dfc2023 --input_type rgb
```

#### Options for train_dfc2023.py

- `--dataset_path`: Path to the DFC2023Amini dataset (required)
- `--output_dir`: Directory to save model weights (default: `weights/dfc2023`)
- `--input_type`: Input data type to use (`rgb` or `sar`, default: `rgb`)
- `--target_type`: Target data type to use (default: `dsm`)
- `--max_epochs`: Maximum number of training epochs (default: 1000)
- `--patience`: Early stopping patience (default: 200)
- `--gpu_count`: Number of GPUs to use (default: auto-detect)

## Prediction

You have two options for running predictions:

### Option 1: Using the original prediction script with preprocessed data

If you've preprocessed your data using `preprocess_dfc2023.py`, you can use the original prediction script:

```
python predict.py -i path/to/processed_dataset/test/x/*.npy -o predictions -w weights/best_run.ckpt
```

### Option 2: Using the DFC2023-specific prediction script

Alternatively, use the `predict_dfc2023.py` script that handles the dataset format on-the-fly:

```
python predict_dfc2023.py --dataset_path /path/to/DFC2023Amini --output_dir predictions --weights weights/dfc2023/best_run.ckpt
```

#### Options for predict_dfc2023.py

- `--dataset_path`: Path to the DFC2023Amini dataset (required)
- `--output_dir`: Directory to save predictions (required)
- `--weights`: Path to the trained model weights (required)
- `--split`: Dataset split to run predictions on (`test`, `valid`, or `train`, default: `test`)
- `--input_type`: Input data type to use (`rgb` or `sar`, default: `rgb`)

## Notes

- The original Im2Height model architecture expects single-channel input and is preserved unchanged
- For RGB images, only the first channel is used (not averaged to grayscale, which would lose information)
- The `.npy` format is used for efficient data loading and is the native format expected by the original implementation
- You can either preprocess the dataset once using `preprocess_dfc2023.py` or use the on-the-fly conversion in the DFC2023-specific scripts
- The semantic segmentation data (`sem` directory) is not used as this implementation is focused on height prediction only
- Results are saved as numpy (.npy) files in the specified output directory
