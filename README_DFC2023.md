# Running Im2Height on DFC2023 Dataset Variants

This document provides instructions on how to run the Im2Height model on various DFC2023 dataset structures (DFC2023Amini, DFC2023S, DFC2023A, DFC2023Asmall).

## Dataset Structure

All DFC2023 dataset variants (DFC2023Amini, DFC2023S, DFC2023A, DFC2023Asmall) follow the same structure:

```
DFC2023<VariantName>/
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

```bash
python preprocess_dfc2023.py --dataset_path /home/asfand/Ahmad/datasets/DFC2023Amini
# Or use any other variant:
# python preprocess_dfc2023.py --dataset_path /home/asfand/Ahmad/datasets/DFC2023Asmall
```

This will automatically generate NPY files in the `data/` directory of your IM2HEIGHT project with the structure expected by the original implementation.

This will create a new directory structure compatible with the original implementation in your project root:

```
data/
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

- `--dataset_path`: Path to the DFC2023 dataset variant (e.g., DFC2023Amini, DFC2023S, DFC2023A, DFC2023Asmall) (required)
- `--output_path`: Path to save the processed .npy files (optional, defaults to `data/` in project root)
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

```bash
python train_dfc2023.py --dataset_path /home/asfand/Ahmad/datasets/DFC2023Amini --output_dir weights/dfc2023 --input_type rgb
# Or use any other variant:
# python train_dfc2023.py --dataset_path /home/asfand/Ahmad/datasets/DFC2023Asmall --output_dir weights/DFC2023Asmall --input_type rgb
```

#### Options for train_dfc2023.py

- `--dataset_path`: Path to the DFC2023 dataset variant (e.g., DFC2023Amini, DFC2023S, DFC2023A, DFC2023Asmall) (required)
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

```bash
python predict.py -i data/test/x/*.npy -o predictions -w weights/best_run.ckpt
```

### Option 2: Using the DFC2023-specific prediction script

Alternatively, use the `predict_dfc2023.py` script that handles the dataset format on-the-fly:

```bash
python predict_dfc2023.py --dataset_path /home/asfand/Ahmad/datasets/DFC2023Amini --output_dir predictions --weights weights/dfc2023/best_run.ckpt
# Or use any other variant:
# python predict_dfc2023.py --dataset_path /home/asfand/Ahmad/datasets/DFC2023Asmall --output_dir predictions/DFC2023Asmall --weights weights/DFC2023Asmall/best_run.ckpt
```

#### Options for predict_dfc2023.py

- `--dataset_path`: Path to the DFC2023 dataset variant (e.g., DFC2023Amini, DFC2023S, DFC2023A, DFC2023Asmall) (required)
- `--output_dir`: Directory to save predictions (required)
- `--weights`: Path to the trained model weights (required)
- `--split`: Dataset split to run predictions on (`test`, `valid`, or `train`, default: `test`)
- `--input_type`: Input data type to use (`rgb` or `sar`, default: `rgb`)

## Notes

- The Im2Height model architecture has been updated to support both single-channel and multi-channel inputs
- For RGB images, all three channels are now preserved (unlike the original implementation that only used one channel)
- TIF/TIFF file formats commonly used in remote sensing are now supported
- The `.npy` format is used for efficient data loading and is the native format expected by the original implementation
- You can either preprocess the dataset once using `preprocess_dfc2023.py` or use the on-the-fly conversion in the DFC2023-specific scripts
- The semantic segmentation data (`sem` directory) is not used as this implementation is focused on height prediction only
- Results are saved as numpy (.npy) files in the specified output directory

## One-Step Preprocessing and Training

For convenience, you can use the `preprocess_and_train.py` script to run both preprocessing and training in one step:

```bash
python preprocess_and_train.py --dataset_path /home/asfand/Ahmad/datasets/DFC2023Amini
# Or use any other variant:
# python preprocess_and_train.py --dataset_path /home/asfand/Ahmad/datasets/DFC2023Asmall
```

This script will first convert the DFC2023 dataset variant to NPY format in the project's `data/` directory, then train the model using that data.
