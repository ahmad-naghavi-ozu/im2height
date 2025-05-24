# Unified Im2Height Implementation

This document describes the unified Im2Height implementation that consolidates the original NPY-based code with DFC2023-specific enhancements, eliminating code duplication while preserving all functionality.

## Overview

The unified implementation provides:

- **Single codebase** supporting multiple dataset formats (NPY and image files)
- **Automatic format detection** and configuration
- **Dynamic channel support** (1, 3, or N channels)
- **Multi-GPU training** with dynamic batch sizing
- **Mixed precision** for large images (>256x256)
- **Backward compatibility** with legacy workflows
- **Unified preprocessing, training, and prediction** pipeline

## Key Files

### Core Implementation
- `data.py` - Unified dataset classes (`UnifiedDataset`, `UnifiedPredictionDataset`)
- `train.py` - Unified training script with dynamic configuration
- `predict_unified.py` - Enhanced prediction script
- `preprocess_unified.py` - Unified preprocessing script
- `run_unified.sh` - Unified run script for complete workflows

### Legacy Support (Deprecated but Maintained)
- `train_dfc2023.py` - DFC2023-specific training (deprecated)
- `predict_dfc2023.py` - DFC2023-specific prediction (deprecated)
- `preprocess_dfc2023.py` - DFC2023-specific preprocessing (deprecated)
- `dfc2023_data.py` - DFC2023-specific dataset classes (deprecated)
- `run_dfc2023.sh` - DFC2023-specific run script (deprecated)

## Quick Start

### 1. Dataset Information
```bash
# Check dataset format and structure
./run_unified.sh --action info --dataset /path/to/dataset
```

### 2. Complete Pipeline (Recommended)
```bash
# Run complete pipeline: preprocess + train + predict
./run_unified.sh --action all --dataset /path/to/DFC2023Amini

# With custom settings
./run_unified.sh --action all --dataset /path/to/dataset --gpus 0,1 --patience 50
```

### 3. Individual Steps

#### Preprocessing
```bash
# Auto-detect format and preprocess if needed
./run_unified.sh --action preprocess --dataset /path/to/dataset

# Force reprocessing
./run_unified.sh --action preprocess --dataset /path/to/dataset --force
```

#### Training
```bash
# Train with auto-configuration
./run_unified.sh --action train --dataset /path/to/dataset

# Train with specific GPUs and patience
./run_unified.sh --action train --dataset /path/to/dataset --gpus 0,1 --patience 100
```

#### Prediction
```bash
# Unified Im2Height Implementation

This document describes the unified Im2Height implementation that consolidates the original NPY-based code with DFC2023-specific enhancements, eliminating code duplication while preserving all functionality.

## Dataset Formats

### NPY Format (Processed)
```
data/
  <dataset_name>/
    train/
      x/           # Input NPY files
      y/           # Target NPY files
    valid/
      x/
      y/
    test/
      x/
      y/
```

### Image Format (Raw)
```
<dataset_path>/
  train/
    rgb/           # Input images (jpg, png, tif)
    dsm/           # Target images
  valid/
    rgb/
    dsm/
  test/
    rgb/
    dsm/
```

## Command Reference

### Unified Run Script (`run_unified.sh`)

```bash
./run_unified.sh [OPTIONS]

Actions:
  info          # Display dataset information
  preprocess    # Convert to NPY format if needed
  train         # Train the model
  predict       # Run predictions
  all          # Complete pipeline

Options:
  -d, --dataset PATH     # Dataset path
  -i, --input TYPE       # Input type (rgb, sar, etc.)
  -t, --target TYPE      # Target type (dsm, etc.)
  -g, --gpus GPUS        # GPU indices (e.g., "0,1")
  -p, --patience N       # Early stopping patience
  -e, --epochs N         # Maximum epochs
  -w, --weights PATH     # Model weights (prediction)
  -o, --output DIR       # Output directory (prediction)
  --force               # Force reprocessing
  --quiet               # Suppress verbose output
```

## Examples

### Example 1: DFC2023 Dataset
```bash
# Complete pipeline with DFC2023Amini
./run_unified.sh --action all --dataset /home/user/datasets/DFC2023Amini

# Training only with custom settings
./run_unified.sh --action train --dataset /home/user/datasets/DFC2023Amini \
  --gpus 0,1 --patience 50 --epochs 500
```

### Example 2: Custom Dataset
```bash
# Check custom dataset
./run_unified.sh --action info --dataset /path/to/custom_dataset

# Process with SAR input
./run_unified.sh --action all --dataset /path/to/sar_dataset \
  --input sar --target elevation
```

### Example 3: Prediction Only
```bash
# Auto-find weights and predict
./run_unified.sh --action predict --dataset /path/to/test_data

# Use specific weights
./run_unified.sh --action predict --dataset /path/to/test_data \
  --weights weights/DFC2023S/epoch_100_best.ckpt
```

## Migration from Legacy Code

### From DFC2023-specific workflow:
```bash
# Old way (deprecated)
./run_dfc2023.sh --action all --dataset /path/to/DFC2023dataset

# New way (unified)
./run_unified.sh --action all --dataset /path/to/DFC2023dataset
```

### From original NPY workflow:
```bash
# Old way
python train.py  # with hardcoded settings

# New way (unified)
./run_unified.sh --action train --dataset /path/to/npy_data
```

## Troubleshooting

### Dataset Not Found
```bash
# Check dataset structure
./run_unified.sh --action info --dataset /path/to/dataset
```

### Memory Issues
```bash
# Use fewer GPUs or smaller batch size
./run_unified.sh --action train --dataset /path --gpus 0
```

The unified implementation maintains full backward compatibility while providing enhanced functionality and better performance.

# Complete pipeline (preprocess + train + predict)
./run_unified.sh --action all --dataset /path/to/dataset
```

### Using Individual Scripts

#### 1. Preprocessing (Optional for Image Datasets)

```bash
# Preprocess any image dataset
python preprocess_unified.py -d /path/to/dataset

# Force reprocess existing NPY files
python preprocess_unified.py -d /path/to/dataset --force

# Get dataset info without processing
python preprocess_unified.py -d /path/to/dataset --info-only
```

#### 2. Training

```bash
# Train with automatic configuration
python train.py --dataset_path /path/to/dataset

# Train with specific GPU setup
python train.py --dataset_path /path/to/dataset --gpus 0,1

# Train with custom parameters
python train.py --dataset_path /path/to/dataset --max_epochs 500 --patience 100
```

#### 3. Prediction

```bash
# Predict with automatic weight finding
python predict_unified.py --dataset_path /path/to/dataset

# Predict with specific weights
python predict_unified.py --dataset_path /path/to/dataset --weights path/to/model.ckpt
```

## Dataset Format Support

### Image Format (DFC2023, Custom Datasets)
```
dataset/
├── train/
│   ├── rgb/          # Input images (.jpg, .png, .tif, .tiff)
│   └── dsm/          # Target images
├── valid/
│   ├── rgb/
│   └── dsm/
└── test/
    ├── rgb/
    └── dsm/
```

### NPY Format (Original Paper, Preprocessed)
```
data/dataset_name/
├── train/
│   ├── x/            # Input NPY files
│   └── y/            # Target NPY files
├── valid/
│   ├── x/
│   └── y/
└── test/
    ├── x/
    └── y/
```

## Dynamic Configuration

The unified implementation automatically configures itself based on:

- **Image Dimensions**: Detected from first sample
- **GPU Count**: Auto-detected or specified
- **Available Memory**: Dynamic batch size adjustment
- **Channel Count**: Auto-detected (1, 3, or N channels)

### Example Configurations

| Image Size | GPU Count | Batch Size | Workers | Mixed Precision |
|------------|-----------|------------|---------|-----------------|
| 256x256    | 1         | 6          | 12      | No              |
| 256x256    | 2         | 12         | 24      | No              |
| 512x512    | 1         | 2          | 4       | Yes             |
| 512x512    | 2         | 4          | 8       | Yes             |
| 1024x1024  | 1         | 1          | 2       | Yes             |

## Migration from DFC2023-Specific Implementation

If you were using the old DFC2023-specific scripts, you can migrate seamlessly:

### Old Way
```bash
./run_dfc2023.sh --action all --dataset /path/to/dataset
```

### New Way (Unified)
```bash
./run_unified.sh --action all --dataset /path/to/dataset
```

The unified implementation provides all the same functionality with enhanced capabilities.

## Advanced Usage

### Custom Input/Target Types
```bash
# Use SAR input instead of RGB
./run_unified.sh --action all --dataset /path/to/dataset --input sar

# Custom target type
./run_unified.sh --action all --dataset /path/to/dataset --target elevation
```

### GPU Configuration
```bash
# Use specific GPUs
./run_unified.sh --action train --dataset /path/to/dataset --gpus 0,1,2

# Single GPU training
./run_unified.sh --action train --dataset /path/to/dataset --gpus 0
```

### Training Parameters
```bash
# Extended training with early stopping
./run_unified.sh --action train --dataset /path/to/dataset --epochs 2000 --patience 300

# Quick training for testing
./run_unified.sh --action train --dataset /path/to/dataset --epochs 50 --patience 10
```

## Output Organization

The unified implementation organizes outputs by dataset name:

```
weights/
└── dataset_name/
    ├── im2height_model_best.ckpt
    ├── im2height_model_last.ckpt
    └── training_log.txt

predictions/
└── dataset_name/
    ├── pred_sample1.npy
    ├── pred_sample2.npy
    └── ...

data/                    # NPY cache for image datasets
└── dataset_name/
    ├── train/
    ├── valid/
    └── test/
```

## Backward Compatibility

The unified implementation maintains full backward compatibility:

- **Existing NPY datasets**: Work without modification
- **Training scripts**: Old `train_dfc2023.py` still available as fallback
- **Model weights**: Compatible between old and new implementations
- **Prediction format**: Same output format as original implementation

## Performance Improvements

The unified implementation includes several performance optimizations:

1. **NPY Caching**: Image datasets converted to NPY for faster loading
2. **Dynamic Memory Management**: Optimal batch sizes based on available memory
3. **Mixed Precision**: Automatic FP16 for large images
4. **Efficient Data Loading**: Optimized DataLoader configurations
5. **GPU Memory Optimization**: Better CUDA memory management

## Troubleshooting

### Common Issues

**"No valid dataset splits found"**
- Check that dataset follows expected directory structure
- Verify input_type and target_type match directory names

**"CUDA out of memory"**
- The system automatically reduces batch size, but for very large images you may need to use fewer GPUs
- Try `--gpus 0` for single GPU training

**"No model weights found"**
- Ensure training completed successfully
- Check the `weights/dataset_name/` directory for `.ckpt` files

### Debug Mode
```bash
# Verbose output for debugging
./run_unified.sh --action all --dataset /path/to/dataset

# Quiet mode for scripting
./run_unified.sh --action all --dataset /path/to/dataset --quiet
```

## Contributing

When contributing to the unified implementation:

1. Test with both NPY and image format datasets
2. Ensure backward compatibility is maintained
3. Update this README with new features
4. Test multi-GPU configurations if possible

## Files in Unified Implementation

### Core Scripts
- `run_unified.sh`: Main entry point script
- `preprocess_unified.py`: Unified preprocessing for all formats
- `train.py`: Enhanced training with unified dataset support
- `predict_unified.py`: Unified prediction script
- `data.py`: UnifiedDataset class with automatic format detection

### Legacy Scripts (Maintained for Compatibility)
- `run_dfc2023.sh`: Original DFC2023-specific script
- `train_dfc2023.py`: Original DFC2023 training script
- `predict_dfc2023.py`: Original DFC2023 prediction script
- `preprocess_dfc2023.py`: Original DFC2023 preprocessing script

The legacy scripts are preserved to ensure existing workflows continue to work while users migrate to the unified implementation.
