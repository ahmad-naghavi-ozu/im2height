# IM2HEIGHT Updates

## Output Consistency

1. **Standardized output directories**: Consolidated all prediction outputs into a single "predictions" directory
2. **Consistent padding**: Fixed padding discrepancy between training and prediction to ensure consistent output dimensions
3. **Streamlined workflow**: Eliminated redundant folders to improve project organization

## Dynamic Configuration Based on Image Size

The training script now dynamically adjusts parameters based on input image dimensions:

1. **Auto-scaling batch size**: Batch size is automatically reduced for larger images to prevent memory issues
2. **Dynamic gradient accumulation**: For larger images, gradient accumulation is used to maintain effective batch size
3. **Worker scaling**: Number of data loader workers scales with image dimensions
4. **Precision adaptation**: For 512x512 or larger images, mixed precision (16-bit) is automatically used
5. **Memory optimization**: Memory management is automatically tuned based on image size

## Multi-Channel Input Support

The original implementation was designed to work with single-channel inputs. This update adds support for:

1. **Multi-channel inputs**: The network now accepts both single-channel and three-channel (RGB) inputs dynamically
2. **TIF file format**: Added support for TIF/TIFF files commonly used in remote sensing
3. **Expected data structure**: Converted files are now stored according to the expected structure in README.md:
4. **Backward compatibility**: All changes maintain compatibility with the original implementation
   ```
   data/
     ├── train/
     │   ├── x/  # RGB aerial/satellite images stored as .npy files
     │   └── y/  # Corresponding DSM (height) data stored as .npy files
     └── test/
         ├── x/  # Test RGB images stored as .npy files
         └── y/  # Test DSM data stored as .npy files
   ```

## Key Changes

1. Modified `Im2Height` class to accept dynamic number of input channels (1 or 3)
2. Updated `DFC2023Dataset` and `DFC2023PredictionDataset` classes to:
   - Preserve all channels in RGB images
   - Support TIF format
   - Save NPY files in the expected data structure
   - Cache converted files to improve performance

3. Added safeguards in prediction scripts to handle model loading when input channels differ
4. Maintained backward compatibility with original implementation and NPY datasets

## Usage with DFC2023 Dataset Variants

The code now correctly handles the DFC2023 dataset structure variants (DFC2023Amini, DFC2023S, DFC2023A, DFC2023Asmall) and saves converted NPY files for future use. The directory structure will be:

```
data/
  ├── train/
  │   ├── x/  # Converted RGB images as NPY files
  │   └── y/  # Converted DSM data as NPY files
  ├── valid/
  │   ├── x/
  │   └── y/
  └── test/
      ├── x/
      └── y/
```

This folder is excluded from Git tracking as mentioned.
