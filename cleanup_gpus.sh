#!/bin/bash
#
# GPU Cleanup Script for Im2Height
# This script helps clean up orphaned GPU processes and clear GPU memory
#

echo "=== GPU Cleanup Script ==="

# Function to show current GPU status
show_gpu_status() {
    echo "Current GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
    while IFS=, read -r gpu_id name mem_used mem_total utilization; do
        mem_used=$(echo $mem_used | tr -d ' ')
        mem_total=$(echo $mem_total | tr -d ' ')
        utilization=$(echo $utilization | tr -d ' ')
        percentage=$((mem_used * 100 / mem_total))
        printf "GPU %s: %s - Memory: %s/%s MB (%.1f%%) - Utilization: %s%%\n" \
               "$gpu_id" "$name" "$mem_used" "$mem_total" "$percentage" "$utilization"
    done
    echo
}

# Function to find and kill GPU processes
cleanup_gpu_processes() {
    echo "Searching for GPU processes..."
    
    # Get all compute processes
    GPU_PROCESSES=$(nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv,noheader,nounits 2>/dev/null || true)
    
    if [ -z "$GPU_PROCESSES" ]; then
        echo "No GPU compute processes found."
        return 0
    fi
    
    echo "Found GPU processes:"
    echo "$GPU_PROCESSES"
    echo
    
    # Filter for Python/training processes
    PYTHON_PROCESSES=$(echo "$GPU_PROCESSES" | grep -E "(python|train)" || true)
    
    if [ -z "$PYTHON_PROCESSES" ]; then
        echo "No Python training processes found on GPUs."
        return 0
    fi
    
    echo "Python/training processes using GPUs:"
    echo "$PYTHON_PROCESSES"
    echo
    
    # Extract PIDs
    PIDS=$(echo "$PYTHON_PROCESSES" | awk -F',' '{print $1}' | tr -d ' ')
    
    if [ -z "$PIDS" ]; then
        echo "No PIDs found."
        return 0
    fi
    
    echo "Process details:"
    for pid in $PIDS; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "PID $pid: $(ps -p $pid -o comm= -o args= 2>/dev/null || echo 'Process not found')"
        fi
    done
    echo
    
    read -p "Do you want to kill these processes? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        for pid in $PIDS; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "Killing process $pid..."
                kill -TERM "$pid" 2>/dev/null || true
                sleep 2
                if kill -0 "$pid" 2>/dev/null; then
                    echo "Force killing process $pid..."
                    kill -KILL "$pid" 2>/dev/null || true
                fi
            fi
        done
        echo "Waiting for GPU memory to clear..."
        sleep 3
    else
        echo "Skipping process cleanup."
    fi
}

# Function to clear CUDA cache
clear_cuda_cache() {
    echo "Clearing CUDA cache..."
    python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('CUDA cache cleared successfully')
else:
    print('CUDA not available')
" 2>/dev/null || echo "Failed to clear CUDA cache"
    echo
}

# Main execution
echo "Starting GPU cleanup..."
echo

# Show initial status
show_gpu_status

# Cleanup processes
cleanup_gpu_processes

# Clear CUDA cache
clear_cuda_cache

# Show final status
echo "Final GPU status:"
show_gpu_status

echo "GPU cleanup completed!"
