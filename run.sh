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
DATASET_PATH="/home/asfand/Ahmad/datasets/Huawei_Contest"
INPUT_TYPE="rgb"
TARGET_TYPE="dsm"
ACTION="train"  # Default action: train
GPUS="1,2"        # Default: use all available GPUs
PATIENCE="20"  # Default early stopping patience
MAX_EPOCHS="200"  # Default maximum epochs
BATCH_SIZE=""  # Default: use dynamic calculation (empty = auto)
RESUME_CHECKPOINT=""  # Default: no checkpoint resumption

# Track background processes for cleanup
BACKGROUND_PIDS=()
SCRIPT_PID=$$

# Enhanced cleanup function to kill background processes and clear GPU memory
cleanup() {
    echo "🧹 Starting cleanup process..."
    
    # Kill any background processes we started
    for pid in "${BACKGROUND_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "🔄 Killing tracked background process: $pid"
            kill -TERM "$pid" 2>/dev/null || true
            sleep 2
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    
    # Find and kill any Im2Height related processes
    echo "🔍 Searching for Im2Height related processes..."
    
    # Get all python processes that might be related to Im2Height
    IM2HEIGHT_PIDS=$(ps aux | grep -E "(train\.py|im2height|Im2Height)" | grep -v grep | grep -v "$$" | awk '{print $2}' || true)
    
    if [ ! -z "$IM2HEIGHT_PIDS" ]; then
        echo "🎯 Found Im2Height processes: $IM2HEIGHT_PIDS"
        for pid in $IM2HEIGHT_PIDS; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "🔄 Killing Im2Height process: $pid"
                kill -TERM "$pid" 2>/dev/null || true
                sleep 1
                kill -KILL "$pid" 2>/dev/null || true
            fi
        done
    fi
    
    # Find and kill GPU processes started by this user's Im2Height
    echo "🔍 Searching for GPU processes..."
    GPU_PIDS=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null | \
               grep -E "(python.*train|im2height)" | awk -F',' '{print $1}' | tr -d ' ' || true)
    
    if [ ! -z "$GPU_PIDS" ]; then
        echo "🎯 Found GPU processes: $GPU_PIDS"
        for pid in $GPU_PIDS; do
            # Check if this process belongs to current user
            if ps -o user= -p "$pid" 2>/dev/null | grep -q "$(whoami)"; then
                if kill -0 "$pid" 2>/dev/null; then
                    echo "🔄 Killing GPU process: $pid"
                    kill -TERM "$pid" 2>/dev/null || true
                    sleep 1
                    kill -KILL "$pid" 2>/dev/null || true
                fi
            fi
        done
    fi
    
    # Wait for processes to die
    echo "⏳ Waiting for processes to terminate..."
    sleep 3
    
    # Clear CUDA cache
    echo "🗑️ Clearing CUDA cache..."
    python -c "
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print('✅ CUDA cache cleared')
else:
    print('⚠️ CUDA not available')
" 2>/dev/null || echo "❌ Failed to clear CUDA cache"
    
    # Final verification
    echo "🔍 Final verification..."
    REMAINING_PIDS=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null | \
                     grep -E "(python.*train|im2height)" | awk -F',' '{print $1}' | tr -d ' ' || true)
    
    if [ ! -z "$REMAINING_PIDS" ]; then
        echo "⚠️ Warning: Some processes may still be running: $REMAINING_PIDS"
    else
        echo "✅ All GPU processes cleaned up successfully"
    fi
    
    echo "🧹 Cleanup completed."
}

# Set up signal handlers for cleanup
trap cleanup EXIT
trap cleanup INT
trap cleanup TERM

# Create PID file for this script
PID_FILE="/tmp/im2height_${USER}_$$.pid"
echo $$ > "$PID_FILE"

# Function to clean up PID file
cleanup_pid_file() {
    rm -f "$PID_FILE" 2>/dev/null || true
}

# Add PID file cleanup to main cleanup
original_cleanup=$(declare -f cleanup)
eval "${original_cleanup/echo \"🧹 Cleanup completed.\"/cleanup_pid_file; echo \"🧹 Cleanup completed.\"}"

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
    echo "  -a, --action ACTION       Action to perform: preprocess, train, predict, evaluate, info, or all (default: train)"
    echo "  -d, --dataset PATH        Path to dataset (default: $DATASET_PATH)"
    echo "  -i, --input TYPE          Input data type: rgb, sar, etc. (default: $INPUT_TYPE)"
    echo "  -t, --target TYPE         Target data type: dsm, etc. (default: $TARGET_TYPE)"
    echo "  -w, --weights PATH        Path to model weights (for prediction only)"
    echo "  -o, --output DIR          Output directory (for prediction only)"
    echo "  -g, --gpus GPUs           Comma-separated list of GPU indices to use (e.g. '0,1')"
    echo "  -p, --patience NUMBER     Early stopping patience for training (default: $PATIENCE)"
    echo "  -e, --epochs NUMBER       Maximum training epochs (default: $MAX_EPOCHS)"
    echo "  -b, --batch-size NUMBER   Batch size per GPU (default: dynamic calculation based on image size)"
    echo "  -r, --resume PATH         Resume training from checkpoint file (e.g., weights/dataset/last.ckpt)"
    echo "  --force                   Force reprocess even if NPY files exist"
    echo "  --quiet                   Suppress verbose output"
    echo ""
    echo "Actions:"
    echo "  info                      Display dataset information and format"
    echo "  preprocess                Convert images to NPY format (MANDATORY for image datasets)"
    echo "  train                     Train the model (automatically uses NPY format if available)"
    echo "  predict                   Run predictions (automatically uses NPY format if available)"
    echo "  evaluate                  Evaluate model predictions against ground truth"
    echo "  all                       Run complete pipeline: preprocess + train + predict + evaluate"
    echo "  cleanup                   Clean up GPU processes and memory"
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
    echo "  # Train with specific batch size (useful for large images or limited GPU memory)"
    echo "  ./run.sh --action train --dataset /path/to/DFC2023Amini --batch-size 2"
    echo "  "
    echo "  # Resume training from last checkpoint"
    echo "  ./run.sh --action train --dataset /path/to/DFC2023Amini --resume weights/DFC2023Amini/last.ckpt"
    echo "  "
    echo "  # Predict on dataset (automatically uses NPY if available)"
    echo "  ./run.sh --action predict --dataset /path/to/DFC2023Amini --weights weights/best.ckpt"
    echo "  "
    echo "  # Evaluate predictions against ground truth"
    echo "  ./run.sh --action evaluate --dataset /path/to/DFC2023Amini"
    echo "  "
    echo "  # Complete pipeline for image dataset with custom batch size"
    echo "  ./run.sh --action all --dataset /path/to/dataset --gpus 0,1 --batch-size 4"
    echo ""
    echo "MEMORY USAGE NOTES:"
    echo "  • Batch size automatically adjusts based on image size"
    echo "  • For large images (>256x256), use smaller batch sizes to avoid OOM errors"
    echo "  • 500x500 images: try --batch-size 2 or 4"
    echo "  • 1024x1024 images: try --batch-size 1"
    echo "  • Memory usage scales quadratically with image size"
    echo ""
    echo "WORKFLOW:"
    echo "  1. info      → Check dataset format"
    echo "  2. preprocess → Convert images to NPY (if needed)"
    echo "  3. train     → Train model (auto-detects NPY format)"
    echo "  4. predict   → Generate predictions (auto-detects NPY format)"
    echo "  5. evaluate  → Evaluate predictions against ground truth"
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
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -r|--resume)
            RESUME_CHECKPOINT="$2"
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

# Function to check for and optionally kill existing GPU processes
check_gpu_processes() {
    echo "🔍 Checking for existing GPU processes..."
    
    # Get current user's processes using GPUs
    GPU_PROCESSES=$(nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv,noheader,nounits 2>/dev/null | grep -E "(python|train\.py)" || true)
    
    if [ ! -z "$GPU_PROCESSES" ]; then
        echo "🎯 Found existing GPU processes:"
        echo "$GPU_PROCESSES"
        
        # Extract PIDs of processes that might conflict
        ALL_PYTHON_PIDS=$(echo "$GPU_PROCESSES" | awk -F',' '{print $1}' | tr -d ' ' || true)
        
        # Filter for Im2Height related processes
        IM2HEIGHT_PIDS=""
        for pid in $ALL_PYTHON_PIDS; do
            if ps -p "$pid" -o args= 2>/dev/null | grep -q -E "(train\.py|im2height|Im2Height)"; then
                IM2HEIGHT_PIDS="$IM2HEIGHT_PIDS $pid"
            fi
        done
        
        if [ ! -z "$IM2HEIGHT_PIDS" ]; then
            echo "⚠️ Found potentially conflicting Im2Height processes:"
            for pid in $IM2HEIGHT_PIDS; do
                echo "  PID $pid: $(ps -p $pid -o args= 2>/dev/null | head -c 80)..."
            done
            echo ""
            
            read -p "🤔 Do you want to kill these processes before continuing? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                for pid in $IM2HEIGHT_PIDS; do
                    echo "🔄 Killing process $pid..."
                    kill -TERM "$pid" 2>/dev/null || true
                    sleep 2
                    kill -KILL "$pid" 2>/dev/null || true
                done
                echo "⏳ Waiting for GPU memory to clear..."
                sleep 3
                echo "✅ Process cleanup completed"
            else
                echo "⚠️ Continuing with existing processes. This may cause conflicts."
            fi
        else
            echo "ℹ️ No Im2Height processes found, GPU processes belong to other applications"
        fi
    else
        echo "✅ No existing GPU processes found"
    fi
}

# Check for required scripts
check_script "preprocess.py"
check_script "train.py"
check_script "predict.py"
check_script "evaluate.py"

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
        
        # Check for existing GPU processes
        check_gpu_processes
        
        # Check if NPY dataset exists, otherwise use original path
        NPY_DATASET_PATH="data/${DATASET_NAME}"
        if [ -d "$NPY_DATASET_PATH" ]; then
            echo "Using preprocessed NPY dataset: $NPY_DATASET_PATH"
            TRAINING_DATASET_PATH="$NPY_DATASET_PATH"
        else
            echo "NPY dataset not found at $NPY_DATASET_PATH, using original path: $DATASET_PATH"
            TRAINING_DATASET_PATH="$DATASET_PATH"
        fi
        
        # Clear CUDA cache before training
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        
        # Build training command
        CMD="python train.py --dataset_path \"$TRAINING_DATASET_PATH\" --input_type \"$INPUT_TYPE\" --target_type \"$TARGET_TYPE\" --max_epochs \"$MAX_EPOCHS\" --patience \"$PATIENCE\" --output_dir \"weights/${DATASET_NAME}\""
        
        # Add batch size if specified
        if [ ! -z "$BATCH_SIZE" ]; then
            CMD="$CMD --batch_size \"$BATCH_SIZE\""
        fi
        
        # Add GPU specification if provided
        if [ ! -z "$GPUS" ]; then
            CMD="$CMD --gpu_count \"$GPUS\""
        fi
        
        # Add quiet flag if specified
        if [ ! -z "$QUIET_FLAG" ]; then
            CMD="$CMD --quiet"
        fi
        
        # Add resume checkpoint if specified
        if [ ! -z "$RESUME_CHECKPOINT" ]; then
            if [ -f "$RESUME_CHECKPOINT" ]; then
                CMD="$CMD --resume_from_checkpoint \"$RESUME_CHECKPOINT\""
                echo "Resuming training from checkpoint: $RESUME_CHECKPOINT"
            else
                echo "Warning: Checkpoint file not found: $RESUME_CHECKPOINT"
                echo "Starting training from scratch..."
            fi
        fi
        
        # Execute training command and capture PID
        echo "Starting training with command: $CMD"
        
        # Use a more robust execution method
        (
            # Set up error handling within the subshell
            set -e
            exec $CMD
        ) &
        
        TRAIN_PID=$!
        BACKGROUND_PIDS+=($TRAIN_PID)
        
        echo "🚀 Training started with PID: $TRAIN_PID"
        echo "📊 Monitoring training progress..."
        
        # Monitor the training process
        while kill -0 $TRAIN_PID 2>/dev/null; do
            sleep 10
            # Optional: Add progress monitoring here
        done
        
        # Wait for training to complete and get exit code
        wait $TRAIN_PID
        TRAIN_EXIT_CODE=$?
        
        echo "🏁 Training completed with exit code: $TRAIN_EXIT_CODE"
        
        if [ $TRAIN_EXIT_CODE -ne 0 ]; then
            echo "❌ Training failed with exit code: $TRAIN_EXIT_CODE"
            echo "🧹 Initiating cleanup due to training failure..."
            cleanup
            exit $TRAIN_EXIT_CODE
        else
            echo "✅ Training completed successfully"
        fi
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
        else
            # For prediction, ensure we have at least one GPU visible
            echo "No specific GPUs specified for prediction, using all available"
        fi
        
        # Execute prediction command
        eval $CMD
        ;;
    evaluate)
        echo "=== Evaluating predictions for ${DATASET_NAME} dataset ==="
        
        # Check if predictions exist
        PREDICTIONS_DIR="predictions/${DATASET_NAME}"
        if [ ! -d "$PREDICTIONS_DIR" ]; then
            echo "Error: Predictions directory not found: $PREDICTIONS_DIR"
            echo "Please run predictions first using: ./run.sh --action predict --dataset $DATASET_PATH"
            exit 1
        fi
        
        # Check if there are any prediction files
        PRED_COUNT=$(find "$PREDICTIONS_DIR" -name "*.npy" | wc -l)
        if [ "$PRED_COUNT" -eq 0 ]; then
            echo "Error: No prediction files (.npy) found in $PREDICTIONS_DIR"
            echo "Please run predictions first using: ./run.sh --action predict --dataset $DATASET_PATH"
            exit 1
        fi
        
        echo "Found $PRED_COUNT prediction files in $PREDICTIONS_DIR"
        
        # Build evaluation command
        CMD="python evaluate.py --dataset \"$DATASET_PATH\" --predictions \"$PREDICTIONS_DIR\""
        
        # Add quiet flag if specified
        if [ ! -z "$QUIET_FLAG" ]; then
            CMD="$CMD --quiet"
        fi
        
        # Set output directory for evaluation results
        EVAL_OUTPUT_DIR="evaluations"
        mkdir -p "$EVAL_OUTPUT_DIR"
        CMD="$CMD --output \"$EVAL_OUTPUT_DIR/evaluation_${DATASET_NAME}.csv\""
        
        echo "Running evaluation with command: $CMD"
        
        # Execute evaluation command
        eval $CMD
        
        if [ $? -eq 0 ]; then
            echo "✅ Evaluation completed successfully"
            echo "📊 Results saved to: $EVAL_OUTPUT_DIR/evaluation_${DATASET_NAME}.csv"
            echo "📋 Terminal output saved to: $EVAL_OUTPUT_DIR/evaluation_${DATASET_NAME}.txt"
        else
            echo "❌ Evaluation failed"
            exit 1
        fi
        ;;
    all)
        echo "=== Running full pipeline: preprocess, train, predict, evaluate ==="
        
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
        
        # Add batch size if specified
        if [ ! -z "$BATCH_SIZE" ]; then
            CMD="$CMD --batch_size \"$BATCH_SIZE\""
        fi
        
        # Add GPU specification if provided
        if [ ! -z "$GPUS" ]; then
            CMD="$CMD --gpu_count \"$GPUS\""
        fi
        
        # Add quiet flag if specified
        if [ ! -z "$QUIET_FLAG" ]; then
            CMD="$CMD --quiet"
        fi
        
        # Add resume checkpoint if specified
        if [ ! -z "$RESUME_CHECKPOINT" ]; then
            if [ -f "$RESUME_CHECKPOINT" ]; then
                CMD="$CMD --resume_from_checkpoint \"$RESUME_CHECKPOINT\""
                echo "Resuming training from checkpoint: $RESUME_CHECKPOINT"
            else
                echo "Warning: Checkpoint file not found: $RESUME_CHECKPOINT"
                echo "Starting training from scratch..."
            fi
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
            
            # Set GPU visibility if specified
            if [ ! -z "$GPUS" ]; then
                echo "Using GPUs for prediction: $GPUS"
                export CUDA_VISIBLE_DEVICES="$GPUS"
            else
                # For prediction, don't restrict GPU visibility to avoid device mapping issues
                echo "Using all available GPUs for prediction"
                unset CUDA_VISIBLE_DEVICES
            fi
            
            # Execute prediction command
            eval $CMD
            
            if [ $? -eq 0 ]; then
                echo "✅ Prediction completed successfully"
                
                # Step 4: Evaluate predictions
                echo "Step 4: Evaluating predictions..."
                
                # Build evaluation command
                EVAL_CMD="python evaluate.py --dataset \"$DATASET_PATH\" --predictions \"predictions/${DATASET_NAME}\""
                
                # Add quiet flag if specified
                if [ ! -z "$QUIET_FLAG" ]; then
                    EVAL_CMD="$EVAL_CMD --quiet"
                fi
                
                # Set output directory for evaluation results
                EVAL_OUTPUT_DIR="evaluations"
                mkdir -p "$EVAL_OUTPUT_DIR"
                EVAL_CMD="$EVAL_CMD --output \"$EVAL_OUTPUT_DIR/evaluation_${DATASET_NAME}.csv\""
                
                echo "Running evaluation with command: $EVAL_CMD"
                
                # Execute evaluation command
                eval $EVAL_CMD
                
                if [ $? -eq 0 ]; then
                    echo "✅ Evaluation completed successfully"
                    echo "📊 Results saved to: $EVAL_OUTPUT_DIR/evaluation_${DATASET_NAME}.csv"
                else
                    echo "⚠️ Evaluation failed, but pipeline continues"
                fi
            else
                echo "❌ Prediction failed, skipping evaluation"
            fi
        fi
        ;;
    cleanup)
        echo "=== Manual GPU Cleanup ==="
        
        # Show current GPU status
        echo "Current GPU status:"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
        
        # Check for our processes
        check_gpu_processes
        
        # Force cleanup
        cleanup
        
        echo "GPU cleanup completed. New status:"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
        ;;
    *)
        echo "Unknown action: $ACTION"
        echo "Valid actions: info, preprocess, train, predict, evaluate, all, cleanup"
        show_help
        exit 1
        ;;
esac

echo "Done!"
