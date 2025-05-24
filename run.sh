#!/bin/bash
#
# Run script for Im2Height model that supports multiple dataset formats.
#
# This script can handle:
# 1. Legacy NPY datasets (original paper format)
# 2. Image datasets (DFC2023 and similar formats)
# 3. Automatic format detection and configuration
# 4. Multi-GPU training with dynamic configuration
# 5. Comprehensive preprocessing, training, and prediction pipeline
#

# Default settings
DATASET_PATH="/home/asfand/Ahmad/datasets/DFC2023Amini"
INPUT_TYPE="rgb"
TARGET_TYPE="dsm"
ACTION="train"  # Default action: train
GPUS=""        # Default: use all available GPUs
PATIENCE="200"  # Default early stopping patience
MAX_EPOCHS="1000"  # Default maximum epochs

# Help function
function show_help {
    echo "Usage: ./run.sh [OPTIONS]"
    echo ""
    echo "⚠️  IMPORTANT: Im2Height ONLY works with NPY format for training/prediction!"
    echo "   If your dataset contains images, you MUST run preprocessing first."
    echo ""
    echo "Im2Height runner supporting multiple dataset formats:"
    echo "  - Legacy NPY format (original paper) - ready for training"
    echo "  - Image format (DFC2023, custom datasets) - requires preprocessing first"
    echo "  - Automatic format detection and configuration"
    echo ""
    echo "Options:"
    echo "  -h, --help                Show this help message"
    echo "  -a, --action ACTION       Action to perform: preprocess, train, predict, info, or all (default: train)"
    echo "  -d, --dataset PATH        Path to dataset (default: $DATASET_PATH)"
    echo "  -i, --input TYPE          Input data type: rgb, sar, etc. (default: $INPUT_TYPE)"
    echo "  -t, --target TYPE         Target data type: dsm, etc. (default: $TARGET_TYPE)"
    echo "  -w, --weights PATH        Path to model weights (for prediction only)"
    echo "  -o, --output DIR          Output directory (for prediction only)"
    echo "  -g, --gpus GPUs           Comma-separated list of GPU indices to use (e.g. '0,1')"
    echo "  -p, --patience NUMBER     Early stopping patience for training (default: $PATIENCE)"
    echo "  -e, --epochs NUMBER       Maximum training epochs (default: $MAX_EPOCHS)"
    echo "  --force                   Force reprocess even if NPY files exist"
    echo "  --quiet                   Suppress verbose output"
    echo ""
    echo "Actions:"
    echo "  info                      Display dataset information and format"
    echo "  preprocess                Convert images to NPY format (MANDATORY for image datasets)"
    echo "  train                     Train the model (automatically uses NPY format if available)"
    echo "  predict                   Run predictions (automatically uses NPY format if available)"
    echo "  all                       Run complete pipeline: preprocess + train + predict"
    echo ""
    echo "Examples:"
    echo "  # Check dataset format and structure"
    echo "  ./run.sh --action info --dataset /path/to/dataset"
    echo "  "
    echo "  # Convert image dataset to NPY (REQUIRED for image datasets)"
    echo "  ./run.sh --action preprocess --dataset /path/to/DFC2023Amini"
    echo "  "
    echo "  # Train on dataset (automatically uses NPY if available)"
    echo "  ./run.sh --action train --dataset /path/to/DFC2023Amini --patience 20"
    echo "  "
    echo "  # Predict on dataset (automatically uses NPY if available)"
    echo "  ./run.sh --action predict --dataset /path/to/DFC2023Amini --weights weights/best.ckpt"
    echo "  "
    echo "  # Complete pipeline for image dataset"
    echo "  ./run.sh --action all --dataset /path/to/dataset --gpus 0,1"
    echo ""
    echo "WORKFLOW:"
    echo "  1. info      → Check dataset format"
    echo "  2. preprocess → Convert images to NPY (if needed)"
    echo "  3. train     → Train model (auto-detects NPY format)"
    echo "  4. predict   → Generate predictions (auto-detects NPY format)"
    echo ""
    echo "SMART PATH DETECTION:"
    echo "  • train/predict actions automatically use preprocessed NPY data when available"
    echo "  • Falls back to original path if NPY data not found"
    echo "  • NPY format: data/DATASET_NAME/ (faster loading, preprocessed)"
    echo "  • Original format: your specified dataset path"
}

# Process command-line arguments
FORCE_FLAG=""
QUIET_FLAG=""

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
        -e|--epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --force)
            FORCE_FLAG="--force"
            shift
            ;;
        --quiet)
            QUIET_FLAG="--quiet"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate dataset path
if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: Dataset path does not exist: $DATASET_PATH"
    exit 1
fi

# Extract dataset name from the path (last folder name)
DATASET_NAME=$(basename "$DATASET_PATH")

# Create necessary directories
mkdir -p "data/${DATASET_NAME}"
mkdir -p "weights/${DATASET_NAME}"
mkdir -p "predictions/${DATASET_NAME}"

# Function to check if Python script exists
check_script() {
    if [ ! -f "$1" ]; then
        echo "Error: Required script not found: $1"
        echo "Please ensure you're running this from the Im2Height project directory."
        exit 1
    fi
}

# Check for required scripts
check_script "preprocess.py"
check_script "train.py"
check_script "predict.py"

# Perform the requested action
case $ACTION in
    info)
        echo "=== Dataset Information ==="
        python preprocess.py --dataset_path "$DATASET_PATH" --input_type "$INPUT_TYPE" --target_type "$TARGET_TYPE" --info-only $QUIET_FLAG
        ;;
    preprocess)
        echo "=== Preprocessing ${DATASET_NAME} dataset ==="
        python preprocess.py --dataset_path "$DATASET_PATH" --input_type "$INPUT_TYPE" --target_type "$TARGET_TYPE" $FORCE_FLAG $QUIET_FLAG
        ;;
    train)
        echo "=== Training Im2Height model on ${DATASET_NAME} dataset ==="
        
        # Check if NPY dataset exists, otherwise use original path
        NPY_DATASET_PATH="data/${DATASET_NAME}"
        if [ -d "$NPY_DATASET_PATH" ]; then
            echo "Using preprocessed NPY dataset: $NPY_DATASET_PATH"
            TRAINING_DATASET_PATH="$NPY_DATASET_PATH"
        else
            echo "NPY dataset not found at $NPY_DATASET_PATH, using original path: $DATASET_PATH"
            TRAINING_DATASET_PATH="$DATASET_PATH"
        fi
        
        # Build training command
        CMD="python train.py --dataset_path \"$TRAINING_DATASET_PATH\" --input_type \"$INPUT_TYPE\" --target_type \"$TARGET_TYPE\" --max_epochs \"$MAX_EPOCHS\" --patience \"$PATIENCE\" --output_dir \"weights/${DATASET_NAME}\""
        
        # Add GPU specification if provided
        if [ ! -z "$GPUS" ]; then
            CMD="$CMD --gpu_count \"$GPUS\""
        fi
        
        # Add quiet flag if specified
        if [ ! -z "$QUIET_FLAG" ]; then
            CMD="$CMD --quiet"
        fi
        
        # Execute training command
        eval $CMD
        ;;
    predict)
        # Check if NPY dataset exists, otherwise use original path
        NPY_DATASET_PATH="data/${DATASET_NAME}"
        if [ -d "$NPY_DATASET_PATH" ]; then
            echo "Using preprocessed NPY dataset: $NPY_DATASET_PATH"
            PREDICTION_DATASET_PATH="$NPY_DATASET_PATH"
        else
            echo "NPY dataset not found at $NPY_DATASET_PATH, using original path: $DATASET_PATH"
            PREDICTION_DATASET_PATH="$DATASET_PATH"
        fi
        
        # Handle weights parameter
        if [ -z "$WEIGHTS" ]; then
            # Automatically find the best model weights
            BEST_WEIGHTS=$(find "weights/${DATASET_NAME}" -name "*best*.ckpt" -o -name "*best*.pth" | sort | tail -n 1)
            if [ -z "$BEST_WEIGHTS" ]; then
                echo "Error: No model weights found in weights/${DATASET_NAME}/"
                echo "Please specify weights path using --weights option, or train a model first."
                exit 1
            else
                echo "No weights specified. Using automatically found best model: $BEST_WEIGHTS"
                WEIGHTS="$BEST_WEIGHTS"
            fi
        fi
        
        # Handle output directory
        if [ -z "$OUTPUT_DIR" ]; then
            OUTPUT_DIR="predictions/${DATASET_NAME}"
            echo "No output directory specified. Using default: $OUTPUT_DIR"
        fi
        
        echo "=== Running predictions on ${DATASET_NAME} dataset ==="
        
        # Build prediction command
        CMD="python predict.py --dataset_path \"$PREDICTION_DATASET_PATH\" --output_dir \"$OUTPUT_DIR\" --weights \"$WEIGHTS\" --input_type \"$INPUT_TYPE\""
        
        # Add quiet flag if specified
        if [ ! -z "$QUIET_FLAG" ]; then
            CMD="$CMD --quiet"
        fi
        
        # Set GPU visibility if specified
        if [ ! -z "$GPUS" ]; then
            echo "Using GPUs: $GPUS"
            export CUDA_VISIBLE_DEVICES="$GPUS"
        fi
        
        # Execute prediction command
        eval $CMD
        ;;
    all)
        echo "=== Running full pipeline: preprocess, train, predict ==="
        
        # Step 1: Preprocess
        echo "Step 1: Preprocessing dataset..."
        python preprocess.py --dataset_path "$DATASET_PATH" --input_type "$INPUT_TYPE" --target_type "$TARGET_TYPE" $FORCE_FLAG $QUIET_FLAG
        
        if [ $? -ne 0 ]; then
            echo "Error during preprocessing. Aborting pipeline."
            exit 1
        fi
        
        # After preprocessing, use the NPY dataset path for training and prediction
        NPY_DATASET_PATH="data/${DATASET_NAME}"
        
        # Clear CUDA cache before training
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        
        # Step 2: Train
        echo "Step 2: Training model..."
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Avoid memory fragmentation
        
        # Build training command
        CMD="python train.py --dataset_path \"$NPY_DATASET_PATH\" --input_type \"$INPUT_TYPE\" --target_type \"$TARGET_TYPE\" --max_epochs \"$MAX_EPOCHS\" --patience \"$PATIENCE\" --output_dir \"weights/${DATASET_NAME}\""
        
        # Add GPU specification if provided
        if [ ! -z "$GPUS" ]; then
            CMD="$CMD --gpu_count \"$GPUS\""
        fi
        
        # Add quiet flag if specified
        if [ ! -z "$QUIET_FLAG" ]; then
            CMD="$CMD --quiet"
        fi
        
        # Execute training command
        eval $CMD
        
        if [ $? -ne 0 ]; then
            echo "Error during training. Aborting pipeline."
            exit 1
        fi
        
        # Clear CUDA cache again before prediction
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        
        # Step 3: Predict
        echo "Step 3: Running predictions..."
        
        # Find the best model weights
        BEST_WEIGHTS=$(find "weights/${DATASET_NAME}" -name "*best*.ckpt" -o -name "*best*.pth" | sort | tail -n 1)
        if [ -z "$BEST_WEIGHTS" ]; then
            echo "Warning: No model weights found. Prediction step skipped."
        else
            echo "Using model weights: $BEST_WEIGHTS"
            
            # Build prediction command
            CMD="python predict.py --dataset_path \"$NPY_DATASET_PATH\" --output_dir \"predictions/${DATASET_NAME}\" --weights \"$BEST_WEIGHTS\" --input_type \"$INPUT_TYPE\""
            
            # Add quiet flag if specified
            if [ ! -z "$QUIET_FLAG" ]; then
                CMD="$CMD --quiet"
            fi
            
            # Execute prediction command
            eval $CMD
        fi
        ;;
    *)
        echo "Unknown action: $ACTION"
        echo "Valid actions: info, preprocess, train, predict, all"
        show_help
        exit 1
        ;;
esac

echo "Done!"
