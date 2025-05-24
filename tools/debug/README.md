# Debug Tools for IM2HEIGHT

This directory contains debugging and testing utilities for the IM2HEIGHT project, specifically focused on multi-GPU training validation and dataset analysis.

## Available Tools

### 1. `test_batch_count.py`
**Purpose**: Verify correct batch counting across multiple GPUs and ensure all training samples are processed.

**Usage**:
```bash
# Test with 2 GPUs
python tools/debug/test_batch_count.py -g "0,1"

# Test with 4 GPUs
python tools/debug/test_batch_count.py -g "0,1,2,3"

# Test with custom dataset
python tools/debug/test_batch_count.py -g "0,1" -d "/path/to/dataset"
```

**Features**:
- Analyzes dataset coverage and batch distribution
- Tests both standard and enhanced DDP strategies
- Reports potential data loss in multi-GPU scenarios
- Validates that all samples are processed per epoch

### 2. `check_dataloader.py`
**Purpose**: Analyze dataloader configuration, memory usage, and batch consistency.

**Usage**:
```bash
# Check default dataset
python tools/debug/check_dataloader.py

# Check specific dataset
python tools/debug/check_dataloader.py -d "/path/to/dataset"
```

**Features**:
- Dynamic configuration analysis
- Memory usage estimation
- Batch size consistency checking
- GPU-specific configuration validation

### 3. `check_dataset.py`
**Purpose**: Verify dataset integrity and structure across train/valid/test splits.

**Usage**:
```bash
# Check default dataset
python tools/debug/check_dataset.py

# Check specific dataset
python tools/debug/check_dataset.py -d "/path/to/dataset"
```

**Features**:
- Dataset split verification
- Sample shape and dtype validation
- Directory structure analysis
- File count and integrity checking

### 4. `test_multi_gpu.py`
**Purpose**: Full end-to-end multi-GPU training test with different strategies.

**Usage**:
```bash
# Test with 2 GPUs for 2 epochs
python tools/debug/test_multi_gpu.py -g "0,1" -e 2

# Test with 4 GPUs and custom batch size
python tools/debug/test_multi_gpu.py -g "0,1,2,3" -b 4 -e 1

# Test with custom dataset
python tools/debug/test_multi_gpu.py -d "/path/to/dataset" -g "0,1"
```

**Features**:
- Tests multiple DDP strategies
- Validates complete training pipeline
- Saves test model weights
- Comprehensive error reporting

### 5. `validate_enhancements.py`
**Purpose**: Comprehensive validation of all training enhancements from the multi-GPU branch.

**Usage**:
```bash
# Run complete validation suite
python tools/debug/validate_enhancements.py
```

**Features**:
- Tests enhanced training configuration
- Validates multi-GPU detection logic
- Analyzes dynamic configuration for different GPU counts
- Comprehensive dataset coverage analysis
- Verifies all enhancements are working correctly

## Quick Start

To validate your multi-GPU setup:

```bash
# 1. Check dataset integrity
python tools/debug/check_dataset.py

# 2. Analyze dataloader configuration
python tools/debug/check_dataloader.py

# 3. Test batch counting (most important for multi-GPU)
python tools/debug/test_batch_count.py -g "0,1"

# 4. Run full multi-GPU test
python tools/debug/test_multi_gpu.py -g "0,1" -e 1

# 5. Validate all enhancements
python tools/debug/validate_enhancements.py
```

## Common Issues and Solutions

### Issue: Not all samples processed in multi-GPU training
**Solution**: The `test_batch_count.py` tool will identify this and suggest using the enhanced DDP strategy implemented in the main training script.

### Issue: Memory errors with large images
**Solution**: Use `check_dataloader.py` to analyze memory requirements and adjust batch sizes accordingly.

### Issue: Inconsistent batch sizes
**Solution**: `check_dataloader.py` will detect this and help identify the cause.

### Issue: Dataset loading errors
**Solution**: `check_dataset.py` provides detailed diagnostics for dataset structure and file accessibility.

## Integration with Main Training

These tools validate the same configurations used in `train_dfc2023.py`:
- Dynamic batch size adjustment based on image size and GPU count
- Enhanced DDP strategy for better sample coverage
- Memory-optimized configurations for different hardware setups

## Output Files

Debug tools may create temporary files and outputs:

**Tracked in Git (shared with team):**
- ✅ All `.py` debug scripts
- ✅ `README.md` documentation
- ✅ Core functionality and utilities

**Ignored by Git (local only):**
- ❌ `tools/debug/outputs/` - Test outputs and results
- ❌ `tools/debug/logs/` - Debug logs and traces  
- ❌ `tools/debug/*.log` - Individual log files
- ❌ `tools/debug/temp_*` - Temporary files
- ❌ `tools/debug/test_multi_gpu_weights/` - Model checkpoints from tests
- ❌ `tools/debug/__pycache__/` - Python cache files

This ensures that:
1. **Debug tools are shared** - Team members can use the same debugging utilities
2. **Outputs stay local** - Large files, logs, and temporary data don't clutter the repository
3. **Clean development** - Each developer's test outputs remain private
