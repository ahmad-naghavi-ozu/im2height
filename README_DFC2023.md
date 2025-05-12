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
- RGB or SAR imagery is used as input (`rgb` or `sar` directories)
- Digital Surface Models (DSM) are used as targets (`dsm` directory)

## Training

To train the Im2Height model on the DFC2023Amini dataset, use the `train_dfc2023.py` script:

```
python train_dfc2023.py --dataset_path /path/to/DFC2023Amini --output_dir weights/dfc2023 --input_type rgb
```

### Options

- `--dataset_path`: Path to the DFC2023Amini dataset (required)
- `--output_dir`: Directory to save model weights (default: `weights/dfc2023`)
- `--input_type`: Input data type to use (`rgb` or `sar`, default: `rgb`)
- `--target_type`: Target data type to use (default: `dsm`)
- `--max_epochs`: Maximum number of training epochs (default: 1000)
- `--patience`: Early stopping patience (default: 200)
- `--gpu_count`: Number of GPUs to use (default: auto-detect)

## Prediction

To run predictions using a trained model on the DFC2023Amini dataset, use the `predict_dfc2023.py` script:

```
python predict_dfc2023.py --dataset_path /path/to/DFC2023Amini --output_dir predictions --weights weights/dfc2023/best_run.ckpt
```

### Options

- `--dataset_path`: Path to the DFC2023Amini dataset (required)
- `--output_dir`: Directory to save predictions (required)
- `--weights`: Path to the trained model weights (required)
- `--split`: Dataset split to run predictions on (`test`, `valid`, or `train`, default: `test`)
- `--input_type`: Input data type to use (`rgb` or `sar`, default: `rgb`)

## Notes

- This implementation preserves the original Im2Height model architecture and only adapts the data loading to work with the DFC2023Amini dataset structure
- Results are saved as numpy (.npy) files in the specified output directory
- The model expects single-channel input, so RGB images are converted to grayscale automatically
