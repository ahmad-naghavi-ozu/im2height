# Debug Tools

This directory contains debugging and diagnostic utilities for the Im2Height pipeline.

## Available Tools

### `debug_channels.py`
**Purpose**: Validates input channel detection and image dimensions across the pipeline.

**Usage**:
```bash
# Check channel detection for a dataset
python tools/debug/debug_channels.py --dataset data/DFC2023Amini

# Check specific image file
python tools/debug/debug_channels.py --image path/to/image.tif

# Check with custom parameters
python tools/debug/debug_channels.py --dataset data/DFC2023Amini --input_type rgb --target_type dsm
```

**Features**:
- Detects RGB vs SAR vs other input types automatically
- Validates image dimensions and padding consistency
- Checks for common preprocessing issues
- Reports dataset statistics and format information

**Output**: Console logging with validation results and recommendations.

## Adding New Debug Tools

When creating new debug utilities:

1. **Naming**: Use descriptive names with `debug_` prefix
2. **Documentation**: Include docstrings and help text
3. **Error Handling**: Provide clear error messages and recovery suggestions
4. **Integration**: Follow the same argument patterns as existing tools
5. **Output**: Use consistent logging format for easy parsing

## Output Directories

Debug tools may create temporary output in:
- `outputs/` - Excluded from Git tracking
- `logs/` - Excluded from Git tracking  
- `temp_*` files - Excluded from Git tracking

All debug script outputs are automatically excluded from Git via the project `.gitignore`.
