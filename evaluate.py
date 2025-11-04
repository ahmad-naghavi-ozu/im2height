#!/usr/bin/env python3
"""
IM2HEIGHT Evaluation Script
Comprehensive evaluation of height estimation models with building-specific metrics.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import glob


def r2_score(y, yhat, eps=1e-8):
    """
    Compute the coefficient of determination (R²) for regression tasks.
    
    R² represents the proportion of variance in the dependent variable that is 
    predictable from the independent variable(s). It ranges from -∞ to 1, where:
    - 1.0 indicates perfect prediction
    - 0.0 indicates the model performs no better than a horizontal line at the mean
    - Negative values indicate the model performs worse than the mean
    
    Args:
        y (np.ndarray): Ground truth values (1D or 2D array)
        yhat (np.ndarray): Predicted values (1D or 2D array)
        eps (float): Small epsilon value to avoid division by zero
        
    Returns:
        float: R² score computed across all pixels, or None if input is empty
    """
    # Flatten arrays to ensure 1D computation across all pixels
    y_flat = y.flatten()
    yhat_flat = yhat.flatten()
    
    if y_flat.size == 0:
        return None
    
    # Residual sum of squares
    ss_res = np.sum((y_flat - yhat_flat) ** 2)
    
    # Total sum of squares
    ss_tot = np.sum((y_flat - np.mean(y_flat)) ** 2)
    
    # Handle edge case where all ground truth values are the same
    if ss_tot < eps:
        return 0.0
    
    return 1.0 - ss_res / (ss_tot + eps)

class HeightRegressionMetrics(object):
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics to initial state."""
        self.mse = 0.0
        self.rmse = 0.0
        self.abs = 0.0
        self.rmse_building = 0.0
        # rmse_matched removed - not applicable for IM2HEIGHT
        self.total_high_rise_rmse = 0.0
        self.total_mid_rise_rmse = 0.0
        self.total_low_rise_rmse = 0.0
        self.delta1_sum = 0.0
        self.delta2_sum = 0.0
        self.delta3_sum = 0.0
        self.r2_sum = 0.0
        self.total_samples = 0
        self.img_sample = 0
        self.count_mid_rise = 0
        self.count_high_rise = 0
        self.count_low_rise = 0
       

    def add_batch(self, gt_image, pre_image, gt_mask, pred_mask, eps=1e-5):
        assert gt_image.shape == pre_image.shape, "Shape mismatch: gt_image shape {}, pre_image shape {}".format(
            gt_image.shape, pre_image.shape)
        
        delta_gt_image = gt_image.copy()
        delta_pre_image = pre_image.copy()
        pre_image[pre_image <= 0] = eps
        gt_image[gt_image <= 0] = eps
        valid_mask = ((gt_image > 0) | (pre_image > 0))
        #print("Valid Mask", valid_mask.shape)
        building_mask = np.expand_dims((gt_mask == 1), axis=0)  # Buildings are 1 in your dataset
        #print("Building Mask", building_mask.shape)
        # Note: matched_building_mask not needed since IM2HEIGHT doesn't predict building masks
        if valid_mask.sum() > 0:
            
            mse_i = np.nanmean((gt_image[valid_mask] - pre_image[valid_mask]) ** 2)
            rmse_i = np.sqrt(mse_i)
            abs_i = np.nanmean(np.abs(gt_image[valid_mask] - pre_image[valid_mask]))

        if building_mask.sum() > 0:
            rmse_b = (np.nanmean((pre_image[building_mask] - gt_image[building_mask]) ** 2)) ** 0.5
        else:
            rmse_b = 0.0

        # Note: rmse_matched is not applicable for IM2HEIGHT since it doesn't predict building masks
        # Only ground truth building mask is available
        rmse_m = 0.0  # Placeholder - not used in final metrics


        low_rise_building_mask = (gt_image >= 1) & (gt_image < 15)
        mid_rise_building_mask = (gt_image >= 15) & (gt_image < 40)
        high_rise_building_mask = gt_image >= 40
        
        low_rise = gt_image[low_rise_building_mask]
        mid_rise = gt_image[mid_rise_building_mask]
        high_rise = gt_image[high_rise_building_mask]

        low_rise_pred = pre_image[low_rise_building_mask]
        mid_rise_pred = pre_image[mid_rise_building_mask]
        high_rise_pred = pre_image[high_rise_building_mask]

        if high_rise.size > 0 and high_rise_pred.size > 0:
            high_rise_mse = np.nanmean((high_rise - high_rise_pred) ** 2)
            high_rise_rmse = np.sqrt(high_rise_mse)
            self.total_high_rise_rmse += high_rise_rmse
            self.count_high_rise += 1  
        else:
            high_rise_rmse = None

        if mid_rise.size > 0 and mid_rise_pred.size > 0:
            mid_rise_mse = np.nanmean((mid_rise - mid_rise_pred) ** 2)
            mid_rise_rmse = np.sqrt(mid_rise_mse)
            self.total_mid_rise_rmse += mid_rise_rmse
            self.count_mid_rise += 1  
        else:
            mid_rise_rmse = None
        
        if low_rise.size > 0 and low_rise_pred.size > 0:
            low_rise_mse = np.nanmean((low_rise - low_rise_pred) ** 2)
            low_rise_rmse = np.sqrt(low_rise_mse)
            self.total_low_rise_rmse += low_rise_rmse
            self.count_low_rise += 1
        else:
            low_rise_rmse = None

        
        self.mse += mse_i
        self.rmse += rmse_i
        self.rmse_building += rmse_b
        # rmse_matched removed - not applicable for IM2HEIGHT
        self.abs += abs_i

        # DELTA METRICS
        delta_pre_image[delta_pre_image <= 0] = eps
        # delta_pre_image[delta_pre_image < 0] = 999
        delta_gt_image[delta_gt_image <= 0] = eps
        maxRatio = np.maximum(delta_pre_image / delta_gt_image, delta_gt_image / delta_pre_image)
        self.delta1_sum += (maxRatio < 1.25).mean()
        self.delta2_sum += (maxRatio < 1.25 ** 2).mean()
        self.delta3_sum += (maxRatio < 1.25 ** 3).mean()
        
        # R² SCORE METRIC
        r2 = r2_score(gt_image, pre_image, eps=eps)
        if r2 is not None:
            self.r2_sum += r2
        
        self.img_sample += 1

    def calculate_metrics(self):
        #mse = np.nanmean(self.mse_list)
       
        #mae = np.nanmean(self.abs_list)
        #rmse = np.nanmean(self.rmse_list)
        #rmse_building = np.nanmean(self.rmse_building_list)
        
        #delta1 = np.nanmean(self.delta1)#self.delta1 / self.total_samples
        #delta2 = np.nanmean(self.delta2)#self.delta2 / self.total_samples
        #delta3 = np.nanmean(self.delta3)#self.delta3 / self.total_samples
        mse = self.mse / self.img_sample
        rmse = self.rmse / self.img_sample
        rmse_building = self.rmse_building / self.img_sample
        # rmse_matched removed - not applicable for IM2HEIGHT
        high_rise_rmse = self.total_high_rise_rmse / self.count_high_rise if self.count_high_rise > 0 else 0
        mid_rise_rmse = self.total_mid_rise_rmse / self.count_mid_rise if self.count_mid_rise > 0 else 0
        low_rise_rmse = self.total_low_rise_rmse / self.count_low_rise if self.count_low_rise > 0 else 0
        mae = self.abs / self.img_sample
        delta1 = self.delta1_sum / self.img_sample
        delta2 = self.delta2_sum / self.img_sample
        delta3 = self.delta3_sum / self.img_sample
        r2 = self.r2_sum / self.img_sample
        
        return mse, rmse, rmse_building, high_rise_rmse, mid_rise_rmse, low_rise_rmse, mae, delta1, delta2, delta3, r2


def load_image_data(file_path):
    """
    Load image data from various formats (NPY, TIFF, PNG, etc.)
    
    Args:
        file_path: Path to image file
        
    Returns:
        numpy array with image data
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.npy':
        return np.load(file_path)
    elif file_ext in ['.tif', '.tiff']:
        try:
            from PIL import Image
            import tifffile
            # Try tifffile first for better handling of scientific TIFF files
            try:
                return tifffile.imread(file_path)
            except:
                # Fallback to PIL
                with Image.open(file_path) as img:
                    return np.array(img)
        except ImportError:
            # If neither available, try with PIL only
            from PIL import Image
            with Image.open(file_path) as img:
                return np.array(img)
    elif file_ext in ['.png', '.jpg', '.jpeg']:
        from PIL import Image
        with Image.open(file_path) as img:
            return np.array(img)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


def load_data_files(dataset_path, split='test'):
    """
    Load RGB, DSM, and semantic mask files from original dataset structure.
    Supports both image formats (.tif, .png, .jpg) and NPY formats.
    
    Args:
        dataset_path: Path to original dataset directory (not preprocessed NPY)
        split: Data split ('train', 'valid', 'test')
        
    Returns:
        List of tuples (rgb_file, dsm_file, sem_file)
    """
    rgb_dir = os.path.join(dataset_path, split, 'rgb')
    dsm_dir = os.path.join(dataset_path, split, 'dsm') 
    sem_dir = os.path.join(dataset_path, split, 'sem')
    
    if not all(os.path.exists(d) for d in [rgb_dir, dsm_dir, sem_dir]):
        missing_dirs = [d for d in [rgb_dir, dsm_dir, sem_dir] if not os.path.exists(d)]
        raise FileNotFoundError(f"Required directories not found in {dataset_path}/{split}: {missing_dirs}")
    
    # Get all image files (support multiple formats)
    image_extensions = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg', '*.npy']
    rgb_files = []
    
    for ext in image_extensions:
        rgb_files.extend(glob.glob(os.path.join(rgb_dir, ext)))
    
    rgb_files = sorted(rgb_files)
    
    if not rgb_files:
        raise FileNotFoundError(f"No image files found in {rgb_dir}")
    
    data_files = []
    for rgb_file in rgb_files:
        base_name = os.path.basename(rgb_file)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # Find corresponding DSM and SEM files (may have different extensions)
        dsm_file = None
        sem_file = None
        
        # Search for DSM file with matching basename
        for ext in image_extensions:
            potential_dsm = os.path.join(dsm_dir, f"{name_without_ext}.{ext.replace('*', '').replace('.', '')}")
            if os.path.exists(potential_dsm):
                dsm_file = potential_dsm
                break
        
        # Search for SEM file with matching basename  
        for ext in image_extensions:
            potential_sem = os.path.join(sem_dir, f"{name_without_ext}.{ext.replace('*', '').replace('.', '')}")
            if os.path.exists(potential_sem):
                sem_file = potential_sem
                break
        
        if dsm_file and sem_file:
            data_files.append((rgb_file, dsm_file, sem_file))
        else:
            print(f"Warning: Missing files for {name_without_ext} (DSM: {dsm_file is not None}, SEM: {sem_file is not None})")
    
    return data_files


def evaluate_dataset(dataset_path, predictions_path, split='test', verbose=True):
    """
    Evaluate model predictions against ground truth.
    
    Args:
        dataset_path: Path to ground truth dataset
        predictions_path: Path to model predictions
        split: Data split to evaluate ('test', 'valid', 'train')
        verbose: Whether to print progress
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Load ground truth files
    data_files = load_data_files(dataset_path, split)
    if not data_files:
        raise ValueError(f"No data files found in {dataset_path}/{split}")
    
    if verbose:
        print(f"Found {len(data_files)} files to evaluate")
    
    # Initialize metrics calculator
    metrics = HeightRegressionMetrics()
    
    processed_count = 0
    error_count = 0
    
    for i, (rgb_file, gt_dsm_file, gt_sem_file) in enumerate(data_files):
        base_name = os.path.basename(gt_dsm_file)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # Look for prediction file (should be .npy format in predictions directory)
        pred_file = os.path.join(predictions_path, f"{name_without_ext}.npy")
        
        if not os.path.exists(pred_file):
            if verbose:
                print(f"Warning: Prediction file not found: {pred_file}")
            error_count += 1
            continue
        
        try:
            # Load data with format detection
            gt_dsm = load_image_data(gt_dsm_file)
            pred_dsm = np.load(pred_file)  # Predictions are always in NPY format
            gt_sem = load_image_data(gt_sem_file)
            
            # Handle shape mismatches and dimensions
            if gt_dsm.ndim == 3 and gt_dsm.shape[0] == 1:
                gt_dsm = gt_dsm.squeeze(0)
            if pred_dsm.ndim == 3 and pred_dsm.shape[0] == 1:
                pred_dsm = pred_dsm.squeeze(0)
            if gt_sem.ndim == 3 and gt_sem.shape[0] == 1:
                gt_sem = gt_sem.squeeze(0)
            elif gt_sem.ndim == 3 and gt_sem.shape[2] == 1:  # Handle HWC format
                gt_sem = gt_sem.squeeze(2)
            
            # Ensure all arrays have the same shape
            if gt_dsm.shape != pred_dsm.shape:
                from skimage.transform import resize
                pred_dsm = resize(pred_dsm, gt_dsm.shape, preserve_range=True)
            
            if gt_sem.shape != gt_dsm.shape:
                from skimage.transform import resize
                gt_sem = resize(gt_sem, gt_dsm.shape, preserve_range=True, order=0)
                gt_sem = gt_sem.astype(int)
            
            # Add batch dimension for compatibility with metrics class
            if gt_dsm.ndim == 2:
                gt_dsm = np.expand_dims(gt_dsm, axis=0)
                pred_dsm = np.expand_dims(pred_dsm, axis=0)
                gt_sem = np.expand_dims(gt_sem, axis=0)
            
            # For IM2HEIGHT evaluation, we only need the ground truth semantic mask
            # No predicted semantic mask is available since IM2HEIGHT only predicts DSM
            pred_sem = gt_sem.copy()  # Not actually used in meaningful way
            
            # Add to metrics (pred_mask parameter is ignored in the updated metrics)
            metrics.add_batch(gt_dsm, pred_dsm, gt_sem.squeeze(0), pred_sem.squeeze(0))
            processed_count += 1
            
            if verbose and (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(data_files)} files...")
                
        except Exception as e:
            if verbose:
                print(f"Error processing {name_without_ext}: {e}")
            error_count += 1
            continue
    
    if processed_count == 0:
        raise ValueError("No files were successfully processed")
    
    # Calculate final metrics
    mse, rmse, rmse_building, high_rise_rmse, mid_rise_rmse, low_rise_rmse, mae, delta1, delta2, delta3, r2 = metrics.calculate_metrics()
    
    results = {
        'processed_files': processed_count,
        'error_files': error_count,
        'total_files': len(data_files),
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2,
        'rmse_building': rmse_building,
        # rmse_matched removed - not applicable for IM2HEIGHT
        'rmse_high_rise': high_rise_rmse,
        'rmse_mid_rise': mid_rise_rmse,
        'rmse_low_rise': low_rise_rmse,
        'delta1': delta1,
        'delta2': delta2,
        'delta3': delta3,
    }
    
    if verbose:
        print(f"\nEvaluation completed:")
        print(f"  Processed: {processed_count}/{len(data_files)} files")
        print(f"  Errors: {error_count}")
    
    return results


def print_results(results, dataset_name):
    """Print formatted evaluation results."""
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS FOR {dataset_name.upper()}")
    print(f"{'='*60}")
    
    print(f"Files processed: {results['processed_files']}/{results['total_files']}")
    if results['error_files'] > 0:
        print(f"Files with errors: {results['error_files']}")
    
    print(f"\nCore Regression Metrics:")
    print(f"  MSE:          {results['mse']:.6f}")
    print(f"  RMSE:         {results['rmse']:.4f}")
    print(f"  MAE:          {results['mae']:.4f}")
    print(f"  R² Score:     {results['r2_score']:.4f}")
    
    print(f"\nBuilding-Specific Metrics:")
    print(f"  RMSE Building: {results['rmse_building']:.4f}")
    # Note: RMSE Matched not applicable for IM2HEIGHT (no predicted building masks)
    
    print(f"\nHeight-Stratified Metrics:")
    print(f"  RMSE Low-rise (1-15m):   {results['rmse_low_rise']:.4f}")
    print(f"  RMSE Mid-rise (15-40m):  {results['rmse_mid_rise']:.4f}")
    print(f"  RMSE High-rise (40m+):   {results['rmse_high_rise']:.4f}")
    
    print(f"\nDelta Accuracy Metrics:")
    print(f"  δ1 (< 1.25):    {results['delta1']:.4f} ({results['delta1']*100:.1f}%)")
    print(f"  δ2 (< 1.56):    {results['delta2']:.4f} ({results['delta2']*100:.1f}%)")
    print(f"  δ3 (< 1.95):    {results['delta3']:.4f} ({results['delta3']*100:.1f}%)")


def save_results_csv(results, dataset_name, output_dir, csv_filename=None):
    """Save results to CSV file in evaluations folder only."""
    import pandas as pd
    
    if csv_filename is None:
        csv_filename = f"{dataset_name}_evaluation_results.csv"
    
    # Flatten results for CSV
    csv_data = {
        'dataset': [dataset_name],
        'processed_files': [results['processed_files']],
        'total_files': [results['total_files']],
        'error_files': [results['error_files']],
        'mse': [results['mse']],
        'rmse': [results['rmse']],
        'mae': [results['mae']],
        'r2_score': [results['r2_score']],
        'rmse_building': [results['rmse_building']],
        # rmse_matched removed - not applicable for IM2HEIGHT
        'rmse_low_rise': [results['rmse_low_rise']],
        'rmse_mid_rise': [results['rmse_mid_rise']],
        'rmse_high_rise': [results['rmse_high_rise']],
        'delta1': [results['delta1']],
        'delta2': [results['delta2']],
        'delta3': [results['delta3']],
    }
    
    df = pd.DataFrame(csv_data)
    
    # Save to evaluations folder only
    evaluations_dir = "evaluations"
    os.makedirs(evaluations_dir, exist_ok=True)
    
    # Save CSV in evaluations folder
    eval_csv_file = os.path.join(evaluations_dir, csv_filename)
    df.to_csv(eval_csv_file, index=False)
    print(f"Results saved to: {eval_csv_file}")
    
    return eval_csv_file


def save_terminal_output(results, dataset_name, csv_filename):
    """Save terminal output to text file in evaluations folder."""
    import datetime
    
    # Create text filename from CSV filename
    text_filename = csv_filename.replace('.csv', '.txt')
    evaluations_dir = "evaluations"
    text_file = os.path.join(evaluations_dir, text_filename)
    
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format the output similar to print_results
    output_lines = [
        f"IM2HEIGHT EVALUATION REPORT",
        f"Generated: {timestamp}",
        f"Dataset: {dataset_name}",
        f"{'='*60}",
        f"EVALUATION RESULTS FOR {dataset_name.upper()}",
        f"{'='*60}",
        f"",
        f"Files processed: {results['processed_files']}/{results['total_files']}",
    ]
    
    if results['error_files'] > 0:
        output_lines.append(f"Files with errors: {results['error_files']}")
    
    output_lines.extend([
        f"",
        f"Core Regression Metrics:",
        f"  MSE:          {results['mse']:.6f}",
        f"  RMSE:         {results['rmse']:.4f}",
        f"  MAE:          {results['mae']:.4f}",
        f"  R² Score:     {results['r2_score']:.4f}",
        f"",
        f"Building-Specific Metrics:",
        f"  RMSE Building: {results['rmse_building']:.4f}",
        f"",
        f"Height-Stratified Metrics:",
        f"  RMSE Low-rise (1-15m):   {results['rmse_low_rise']:.4f}",
        f"  RMSE Mid-rise (15-40m):  {results['rmse_mid_rise']:.4f}",
        f"  RMSE High-rise (40m+):   {results['rmse_high_rise']:.4f}",
        f"",
        f"Delta Accuracy Metrics:",
        f"  δ1 (< 1.25):    {results['delta1']:.4f} ({results['delta1']*100:.1f}%)",
        f"  δ2 (< 1.56):    {results['delta2']:.4f} ({results['delta2']*100:.1f}%)",
        f"  δ3 (< 1.95):    {results['delta3']:.4f} ({results['delta3']*100:.1f}%)",
        f""
    ])
    
    # Write to file
    with open(text_file, 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f"Terminal output saved to: {text_file}")
    return text_file


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='IM2HEIGHT Model Evaluation')
    parser.add_argument('--dataset', required=True, help='Path to ORIGINAL dataset directory (not preprocessed NPY)')
    parser.add_argument('--predictions', required=True, help='Path to predictions directory (e.g., predictions/DFC2023S)')
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'], help='Data split to evaluate')
    parser.add_argument('--output', default='./evaluation_results', help='Output directory or CSV file path for results')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset path does not exist: {args.dataset}")
        sys.exit(1)
    
    if not os.path.exists(args.predictions):
        print(f"Error: Predictions path does not exist: {args.predictions}")
        sys.exit(1)
    
    dataset_name = os.path.basename(args.dataset.rstrip('/'))
    verbose = not args.quiet
    
    # Handle output path - check if it's a CSV file or directory
    if args.output.endswith('.csv'):
        output_dir = os.path.dirname(args.output) or '.'
        csv_filename = os.path.basename(args.output)
    else:
        output_dir = args.output
        csv_filename = f"{dataset_name}_evaluation_results.csv"
    
    if verbose:
        print(f"Dataset: {args.dataset}")
        print(f"Predictions: {args.predictions}")
        print(f"Split: {args.split}")
        print(f"Output: {os.path.join(output_dir, csv_filename)}")
        print()
    
    try:
        # Run evaluation
        results = evaluate_dataset(args.dataset, args.predictions, args.split, verbose)
        
        # Print results
        print_results(results, dataset_name)
        
        # Save results
        eval_csv_file = save_results_csv(results, dataset_name, output_dir, csv_filename)
        
        # Save terminal output to evaluations folder
        save_terminal_output(results, dataset_name, csv_filename)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()