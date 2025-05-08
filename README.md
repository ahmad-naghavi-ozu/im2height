# IM2HEIGHT - Height Estimation from Single Monocular Imagery

PyTorch (Lightning) implementation of Im2Height: [arXiv reference](https://arxiv.org/abs/1802.10249)

## Overview

This project implements a fully residual convolutional-deconvolutional network architecture for estimating height (DSM - Digital Surface Model) from a single monocular remote sensing image. The approach is based on the research paper "IM2HEIGHT: Height Estimation from Single Monocular Imagery via Fully Residual Convolutional-Deconvolutional Network" by Lichao Mou and Xiao Xiang Zhu.

### Problem Statement

Height estimation from a single monocular image is an inherently ambiguous and technically ill-posed problem, with a large source of uncertainty coming from the overall scale. This implementation addresses this challenge using deep learning techniques.

### Network Architecture

The network consists of two main components:
- **Convolutional sub-network**: Transforms the input remote sensing image to high-level multidimensional feature representation
- **Deconvolutional sub-network**: Generates height map from the extracted features

Key features of the architecture:
- Fully residual learning to improve optimization
- Skip connection between the first residual block and the next-to-last block to preserve fine edge details and boundaries
- End-to-end training without any additional post-processing steps

## Dataset Requirements

The model expects the following data structure:
```
data/
  ├── train/
  │   ├── x/  # RGB aerial/satellite images stored as .npy files
  │   └── y/  # Corresponding DSM (height) data stored as .npy files
  └── test/
      ├── x/  # Test RGB images stored as .npy files
      └── y/  # Test DSM data stored as .npy files
```

Each RGB image should correspond to a height map of the same spatial dimensions.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/IM2HEIGHT.git
cd IM2HEIGHT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model from scratch:

```bash
python train.py
```

Training parameters are configured in `train.py`, including:
- Batch size: 6
- Number of workers: 12
- Max epochs: 1000
- Early stopping with patience of 200 epochs
- Model checkpoints saved based on validation loss

### Prediction

To generate height maps from RGB images using a trained model:

```bash
python predict.py --input <input_directory> --output <output_directory> --weights <path_to_model_weights>
```

## Model Performance

According to the paper, the model was evaluated using these metrics:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Structural Similarity Index (SSIM)

The model with skip connections significantly outperforms the basic residual conv-deconv network, particularly in preserving object boundaries and fine details.

## Applications

The estimated height maps can be used for:
1. Instance segmentation of buildings
2. 3D scene understanding
3. Urban planning and monitoring
4. Change detection when combined with temporal data

## Citation

If you use this code for your research, please cite:

```bibtex
@article{mou2018im2height,
  title={IM2HEIGHT: Height Estimation from Single Monocular Imagery via Fully Residual Convolutional-Deconvolutional Network},
  author={Mou, Lichao and Zhu, Xiao Xiang},
  journal={arXiv preprint arXiv:1802.10249},
  year={2018}
}
```

## License

[Include license information here]


