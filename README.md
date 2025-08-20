# Enhanced Im2Height: Multi-Dataset Height Estimation

Enhanced version of the Im2Height model with support for multiple datasets, improved GPU handling, and comprehensive workflow management.

## ⚠️ **IMPORTANT: NPY-Only Workflow**

**Training and prediction ONLY work with NPY format files.** If your dataset contains raw images, you **MUST** run preprocessing first to convert them to NPY format.

## 🚀 Quick Start

### **Workflow Overview:**
1. **Check dataset** → `./run.sh --action info --dataset /path/to/dataset`
2. **Preprocess** → `./run.sh --action preprocess --dataset /path/to/dataset` (if needed)
3. **Train** → `./run.sh --action train --dataset /path/to/dataset`
4. **Predict** → `./run.sh --action predict --dataset /path/to/dataset`

### **Complete Pipeline:**
```bash
# Run everything automatically
./run.sh --action all --dataset /path/to/your/dataset --gpus 0,1
```

## 📋 Supported Dataset Formats

### **NPY Format (Required for Training/Prediction):**
```
dataset/
├── train/
│   ├── x/          # Input images as .npy files
│   └── y/          # Target height maps as .npy files
├── test/
│   ├── x/          # Test input images as .npy files
│   └── y/          # Test target height maps as .npy files
└── valid/          # Optional validation split
    ├── x/
    └── y/
```

### **Raw Image Format (Requires Preprocessing):**
```
dataset/
├── train/
│   ├── rgb/        # RGB images (.jpg, .png, .tif)
│   └── dsm/        # Height maps (.tif, .npy)
├── test/
│   ├── rgb/
│   └── dsm/
└── valid/
    ├── rgb/
    └── dsm/
```

## 🔧 Installation

1. **Clone and setup:**
```bash
git clone <your-repo>
cd IM2HEIGHT
pip install -r requirements.txt
```

2. **Make scripts executable:**
```bash
chmod +x run.sh
```

## 📖 Usage Guide

### **1. Check Dataset Information**
```bash
./run.sh --action info --dataset /path/to/dataset
```
This will show:
- Dataset format (NPY or image)
- Split information (train/test/valid)
- File counts and structure

### **2. Preprocessing (Required for Image Datasets)**
```bash
# Convert image dataset to NPY format
./run.sh --action preprocess --dataset /path/to/image_dataset

# Force reprocessing even if NPY files exist
./run.sh --action preprocess --dataset /path/to/dataset --force
```

### **3. Training**
```bash
# Basic training
./run.sh --action train --dataset /path/to/npy_dataset

# Training with specific GPU configuration
./run.sh --action train --dataset /path/to/dataset --gpus 0,1

# Training with custom parameters
./run.sh --action train --dataset /path/to/dataset \
  --patience 50 --epochs 500 --gpus 0
```

### **4. Prediction**
```bash
# Prediction with automatic weight detection
./run.sh --action predict --dataset /path/to/dataset

# Prediction with specific weights
./run.sh --action predict --dataset /path/to/dataset \
  --weights weights/model_best.ckpt --output custom_predictions/
```

### **5. Complete Pipeline**
```bash
# Run preprocess + train + predict automatically
./run.sh --action all --dataset /path/to/dataset --gpus 0,1
```

## 🎯 Enhanced Features

### **Multi-Dataset Support**
- **DFC2023mini**: Street View imagery dataset
- **DFC2023S**: Satellite imagery dataset  
- **Custom datasets**: Any image format with corresponding height maps
- **Legacy NPY**: Original Im2Height format

### **Improved GPU Handling**
- **Multi-GPU training** with automatic configuration
- **Memory optimization** and CUDA cache management
- **Dynamic GPU selection** via command line
- **Memory-efficient loading** with lazy evaluation

### **Robust Preprocessing**
- **Automatic format detection** for input datasets
- **Efficient NPY conversion** with caching
- **Multiple image formats** (.jpg, .png, .tif, .tiff)
- **Progress tracking** and error handling

### **Enhanced Training**
- **Early stopping** with configurable patience
- **Model checkpointing** with best model selection
- **Learning rate scheduling** and optimization
- **Validation monitoring** and logging

## 🛠️ Command Reference

### **run.sh Options:**
```bash
Usage: ./run.sh [OPTIONS]

Actions:
  info                      Display dataset information and format
  preprocess                Convert images to NPY format  
  train                     Train the model (requires NPY format)
  predict                   Run predictions (requires NPY format)
  all                       Run complete pipeline

Options:
  -d, --dataset PATH        Path to dataset
  -i, --input TYPE          Input data type: rgb, sar, etc. (default: rgb)
  -t, --target TYPE         Target data type: dsm, etc. (default: dsm)
  -g, --gpus GPUs           GPU indices to use (e.g. '0,1')
  -p, --patience NUMBER     Early stopping patience (default: 200)
  -e, --epochs NUMBER       Maximum training epochs (default: 1000)
  -w, --weights PATH        Path to model weights (for prediction)
  -o, --output DIR          Output directory (for prediction)
  --force                   Force reprocess even if NPY files exist
  --quiet                   Suppress verbose output
```

## 📊 Model Architecture

Based on the original Im2Height paper ([arXiv:1802.10249](https://arxiv.org/abs/1802.10249)):

- **Fully residual** convolutional-deconvolutional network
- **Skip connections** for preserving fine details and boundaries  
- **End-to-end training** without post-processing
- **Multi-channel support** (RGB, SAR, etc.)

## 🔍 Troubleshooting

### **Common Issues:**

**"ERROR: Image format dataset detected"**
```bash
# Solution: Run preprocessing first
./run.sh --action preprocess --dataset /path/to/dataset
```

**GPU memory errors during training**
```bash
# Solution: Use fewer GPUs or reduce batch size
./run.sh --action train --dataset /path/to/dataset --gpus 0
```

**No model weights found for prediction**
```bash
# Solution: Train a model first or specify weights path
./run.sh --action train --dataset /path/to/dataset
# Then predict:
./run.sh --action predict --dataset /path/to/dataset
```

## 📁 Directory Structure

```
IM2HEIGHT/
├── run.sh                      # Main workflow script
├── data.py                     # Enhanced dataset handling (NPY-only)
├── preprocess.py               # Multi-dataset preprocessing
├── train.py                    # Enhanced training script
├── predict.py                  # Prediction script
├── im2height.py               # Enhanced model implementation
├── augmenter.py               # Data augmentation
├── requirements.txt           # Dependencies
├── data/                      # Processed datasets (NPY format)
├── weights/                   # Trained model weights
├── predictions/              # Prediction outputs
└── tools/                    # Visualization and metrics tools
    ├── read_pdf.py            # PDF text extraction utility
    ├── metrics.py             # Performance evaluation metrics
    └── visualize_results_*.ipynb # Result visualization notebooks
```

## 📚 Citation

If you use this enhanced version, please cite both the original paper and acknowledge the enhancements:

```bibtex
@article{mou2018im2height,
  title={IM2HEIGHT: Height Estimation from Single Monocular Imagery via Fully Residual Convolutional-Deconvolutional Network},
  author={Mou, Lichao and Zhu, Xiao Xiang},
  journal={arXiv preprint arXiv:1802.10249},
  year={2018}
}
```

## 🤝 Contributing

This enhanced version includes:
- Multi-dataset support and preprocessing pipeline
- Improved GPU handling and memory management  
- Comprehensive workflow management with comprehensive error handling
- Enhanced documentation and user experience

## 📄 License

This project maintains compatibility with the original Im2Height implementation while adding significant enhancements for production use and research applications.


