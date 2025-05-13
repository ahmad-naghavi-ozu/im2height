#!/bin/bash
# filepath: /home/asfand/Ahmad/IM2HEIGHT/run_dfc2023.sh

# Default settings
DATASET_PATH="/home/asfand/Ahmad/datasets/DFC2023Amini"
INPUT_TYPE="rgb"
TARGET_TYPE="dsm"
ACTION="train"  # Default action: train
GPUS=""        # Default: use all available GPUs
PATIENCE="200"  # Default early stopping patience

# Note: The script now uses dynamic configuration based on input image dimensions.
# For 256x256 images (original paper), it will use batch_size=6, workers=12
# For 512x512 images (DFC2023), it will automatically adjust to lower values
# to optimize memory usage and performance.

# Help function
function show_help {
    echo "Usage: ./run_dfc2023.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help                Show this help message"
    echo "  -a, --action ACTION       Action to perform: preprocess, train, predict, or all (default: train)"
    echo "  -d, --dataset PATH        Path to DFC2023Amini dataset (default: $DATASET_PATH)"
    echo "  -i, --input TYPE          Input data type: rgb or sar (default: $INPUT_TYPE)"
    echo "  -t, --target TYPE         Target data type (default: $TARGET_TYPE)"
    echo "  -w, --weights PATH        Path to model weights (for prediction only)"
    echo "  -o, --output DIR          Output directory (for prediction only)"
    echo "  -g, --gpus GPUs           Comma-separated list of GPU indices to use (e.g. '0,1' for first two GPUs)"
    echo "  -p, --patience NUMBER     Early stopping patience value for training (default: $PATIENCE)"
    echo ""
    echo "Examples:"
    echo "  ./run_dfc2023.sh --action preprocess"
    echo "  ./run_dfc2023.sh --action train"
    echo "  ./run_dfc2023.sh --action train --patience 20"
    echo "  ./run_dfc2023.sh --action predict --weights weights/dfc2023/best_run.ckpt --output predictions"
    echo "  ./run_dfc2023.sh --action all"
    echo "  ./run_dfc2023.sh --action all --patience 20"
}

# Process command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -a|--action)
            ACTION="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        -i|--input)
            INPUT_TYPE="$2"
            shift 2
            ;;
        -t|--target)
            TARGET_TYPE="$2"
            shift 2
            ;;
        -w|--weights)
            WEIGHTS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -g|--gpus)
            GPUS="$2"
            shift 2
            ;;
        -p|--patience)
            PATIENCE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Create necessary directories
mkdir -p data
mkdir -p weights/dfc2023

# Perform the requested action
case $ACTION in
    preprocess)
        echo "=== Preprocessing DFC2023Amini dataset ==="
        python preprocess_dfc2023.py --dataset_path "$DATASET_PATH" --input_type "$INPUT_TYPE" --target_type "$TARGET_TYPE"
        ;;
    train)
        echo "=== Training Im2Height model on DFC2023Amini dataset ==="
        if [ -z "$GPUS" ]; then
            python train_dfc2023.py --dataset_path "$DATASET_PATH" --input_type "$INPUT_TYPE" --target_type "$TARGET_TYPE" --patience "$PATIENCE"
        else
            python train_dfc2023.py --dataset_path "$DATASET_PATH" --input_type "$INPUT_TYPE" --target_type "$TARGET_TYPE" --gpu_count "$GPUS" --patience "$PATIENCE"
        fi
        ;;
    predict)
        if [ -z "$WEIGHTS" ]; then
            echo "Error: Model weights path is required for prediction."
            echo "Use --weights to specify the path to model weights."
            exit 1
        fi
        if [ -z "$OUTPUT_DIR" ]; then
            OUTPUT_DIR="predictions"
            echo "No output directory specified. Using default: $OUTPUT_DIR"
        fi
        echo "=== Running predictions on DFC2023Amini dataset ==="
        python predict_dfc2023.py --dataset_path "$DATASET_PATH" --output_dir "$OUTPUT_DIR" --weights "$WEIGHTS" --input_type "$INPUT_TYPE"
        ;;
    all)
        echo "=== Running full pipeline: preprocess, train, predict ==="
        # Preprocess
        echo "Step 1: Preprocessing dataset..."
        python preprocess_dfc2023.py --dataset_path "$DATASET_PATH" --input_type "$INPUT_TYPE" --target_type "$TARGET_TYPE"
        
        # Clear CUDA cache before training
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        
        # Train
        echo "Step 2: Training model..."
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Avoid memory fragmentation
        if [ -z "$GPUS" ]; then
            python train_dfc2023.py --dataset_path "$DATASET_PATH" --input_type "$INPUT_TYPE" --target_type "$TARGET_TYPE" --patience "$PATIENCE"
        else
            python train_dfc2023.py --dataset_path "$DATASET_PATH" --input_type "$INPUT_TYPE" --target_type "$TARGET_TYPE" --gpu_count "$GPUS" --patience "$PATIENCE"
        fi
        
        # Clear CUDA cache again before prediction
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        
        # Find the best model weights
        BEST_WEIGHTS=$(find weights/dfc2023 -name "*best*.ckpt" | sort | tail -n 1)
        if [ -z "$BEST_WEIGHTS" ]; then
            echo "No model weights found. Prediction step skipped."
        else
            # Predict
            echo "Step 3: Running predictions with model $BEST_WEIGHTS..."
            python predict_dfc2023.py --dataset_path "$DATASET_PATH" --output_dir "predictions" --weights "$BEST_WEIGHTS" --input_type "$INPUT_TYPE"
        fi
        ;;
    *)
        echo "Unknown action: $ACTION"
        show_help
        exit 1
        ;;
esac

echo "Done!"
