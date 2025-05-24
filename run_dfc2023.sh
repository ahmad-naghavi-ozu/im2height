#!/bin/bash
# filepath: /home/asfand/Ahmad/IM2HEIGHT/run_dfc2023.sh

# Default settings
DATASET_PATH="/home/asfand/Ahmad/datasets/DFC2023S"
INPUT_TYPE="rgb"
TARGET_TYPE="dsm"
ACTION="train"  # Default action: train
GPUS=""        # Default: use all available GPUs
PATIENCE="200"  # Default early stopping patience

# Note: The script now uses dynamic configuration based on input image dimensions.
# For 256x256 images (original paper), it will use batch_size=6, workers=12
# For 512x512 images (DFC2023), it will automatically adjust to lower values
# to optimize memory usage and performance.
#
# Data for each dataset (e.g., DFC2023S, DFC2023A, DFC2023Asmall, DFC2023Amini) will be stored in 
# separate subdirectories within the data/ folder to keep different dataset versions organized.

# Help function
function show_help {
    echo "Usage: ./run_dfc2023.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help                Show this help message"
    echo "  -a, --action ACTION       Action to perform: preprocess, train, predict, or all (default: train)"
    echo "  -d, --dataset PATH        Path to dataset (e.g., DFC2023S, DFC2023A, DFC2023Asmall, DFC2023Amini) (default: $DATASET_PATH)"
    echo "  -i, --input TYPE          Input data type: rgb or sar (default: $INPUT_TYPE)"
    echo "  -t, --target TYPE         Target data type (default: $TARGET_TYPE)"
    echo "  -w, --weights PATH        Path to model weights (for prediction only)"
    echo "  -o, --output DIR          Output directory (for prediction only)"
    echo "  -g, --gpus GPUs           Comma-separated list of GPU indices to use (e.g. '0,1' for first two GPUs)"
    echo "                            This also affects batch size and worker count for optimal performance"
    echo "  -p, --patience NUMBER     Early stopping patience value for training (default: $PATIENCE)"
    echo ""
    echo "Examples:"
    echo "  ./run_dfc2023.sh --action preprocess"
    echo "  ./run_dfc2023.sh --action train"
    echo "  ./run_dfc2023.sh --action train --patience 20"
    echo "  ./run_dfc2023.sh --action predict                                    # Auto-finds best weights"
    echo "  ./run_dfc2023.sh --action predict --weights path/to/specific.ckpt    # Use specific weights"
    echo "  ./run_dfc2023.sh --action predict --output custom_predictions        # Custom output directory"
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

# Extract dataset name from the path (last folder name)
DATASET_NAME=$(basename "$DATASET_PATH")

# Create necessary directories
mkdir -p "data/${DATASET_NAME}"
mkdir -p "weights/${DATASET_NAME}"

# Perform the requested action
case $ACTION in
    preprocess)
        echo "=== Preprocessing ${DATASET_NAME} dataset ==="
        python preprocess_dfc2023.py --dataset_path "$DATASET_PATH" --input_type "$INPUT_TYPE" --target_type "$TARGET_TYPE"
        ;;
    train)
        echo "=== Training Im2Height model on ${DATASET_NAME} dataset ==="
        if [ -z "$GPUS" ]; then
            python train_dfc2023.py --dataset_path "$DATASET_PATH" --input_type "$INPUT_TYPE" --target_type "$TARGET_TYPE" --patience "$PATIENCE" --output_dir "weights/${DATASET_NAME}"
        else
            python train_dfc2023.py --dataset_path "$DATASET_PATH" --input_type "$INPUT_TYPE" --target_type "$TARGET_TYPE" --gpu_count "$GPUS" --patience "$PATIENCE" --output_dir "weights/${DATASET_NAME}"
        fi
        ;;
    predict)
        if [ -z "$WEIGHTS" ]; then
            # Automatically find the best model weights
            BEST_WEIGHTS=$(find "weights/${DATASET_NAME}" -name "*best*.ckpt" | sort | tail -n 1)
            if [ -z "$BEST_WEIGHTS" ]; then
                echo "Error: No model weights found in weights/${DATASET_NAME}/"
                echo "Please specify weights path using --weights option, or train a model first."
                exit 1
            else
                echo "No weights specified. Using automatically found best model: $BEST_WEIGHTS"
                WEIGHTS="$BEST_WEIGHTS"
            fi
        fi
        if [ -z "$OUTPUT_DIR" ]; then
            OUTPUT_DIR="predictions/${DATASET_NAME}"
            echo "No output directory specified. Using default: $OUTPUT_DIR"
        fi
        echo "=== Running predictions on ${DATASET_NAME} dataset ==="
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
            python train_dfc2023.py --dataset_path "$DATASET_PATH" --input_type "$INPUT_TYPE" --target_type "$TARGET_TYPE" --patience "$PATIENCE" --output_dir "weights/${DATASET_NAME}"
        else
            python train_dfc2023.py --dataset_path "$DATASET_PATH" --input_type "$INPUT_TYPE" --target_type "$TARGET_TYPE" --gpu_count "$GPUS" --patience "$PATIENCE" --output_dir "weights/${DATASET_NAME}"
        fi
        
        # Clear CUDA cache again before prediction
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        
        # Find the best model weights
        BEST_WEIGHTS=$(find "weights/${DATASET_NAME}" -name "*best*.ckpt" | sort | tail -n 1)
        if [ -z "$BEST_WEIGHTS" ]; then
            echo "No model weights found. Prediction step skipped."
        else
            # Predict
            echo "Step 3: Running predictions with model $BEST_WEIGHTS..."
            python predict_dfc2023.py --dataset_path "$DATASET_PATH" --output_dir "predictions/${DATASET_NAME}" --weights "$BEST_WEIGHTS" --input_type "$INPUT_TYPE"
        fi
        ;;
    *)
        echo "Unknown action: $ACTION"
        show_help
        exit 1
        ;;
esac

echo "Done!"
