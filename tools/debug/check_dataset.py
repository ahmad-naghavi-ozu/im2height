#!/usr/bin/env python3
import os
import sys

# Add the parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dfc2023_data import DFC2023Dataset

def check_dataset(dataset_path=None):
    """
    Verify dataset integrity and structure
    
    Args:
        dataset_path: Path to dataset (defaults to DFC2023S)
    """
    if dataset_path is None:
        dataset_path = "/home/asfand/Ahmad/datasets/DFC2023S"
    
    print(f"=== Dataset Integrity Check ===")
    print(f"Dataset path: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return
    
    # Check each split
    splits = ['train', 'valid', 'test']
    split_info = {}
    
    for split in splits:
        print(f"\n--- {split.upper()} Split ---")
        try:
            dataset = DFC2023Dataset(dataset_path, split, 'x', 'y')
            split_info[split] = {
                'size': len(dataset),
                'status': '✓ OK'
            }
            print(f"- Size: {len(dataset)} samples")
            
            # Check first sample
            if len(dataset) > 0:
                sample_input, sample_target = dataset[0]
                print(f"- Input shape: {sample_input.shape}")
                print(f"- Target shape: {sample_target.shape}")
                print(f"- Input dtype: {sample_input.dtype}")
                print(f"- Target dtype: {sample_target.dtype}")
                print(f"- Input range: [{sample_input.min():.3f}, {sample_input.max():.3f}]")
                print(f"- Target range: [{sample_target.min():.3f}, {sample_target.max():.3f}]")
                
                # Show first few filenames if available
                if hasattr(dataset, 'input_files') and len(dataset.input_files) > 0:
                    print(f"- Sample files:")
                    for i in range(min(3, len(dataset.input_files))):
                        input_file = os.path.basename(dataset.input_files[i])
                        print(f"  [{i}] {input_file}")
                    if len(dataset.input_files) > 3:
                        print(f"  ... and {len(dataset.input_files) - 3} more")
            else:
                print("- ❌ Empty dataset")
                split_info[split]['status'] = '❌ Empty'
                
        except Exception as e:
            print(f"- ❌ Error loading {split} dataset: {e}")
            split_info[split] = {
                'size': 0,
                'status': f'❌ Error: {str(e)[:50]}...'
            }
    
    # Summary
    print(f"\n=== Summary ===")
    total_samples = sum(info['size'] for info in split_info.values())
    print(f"Total samples: {total_samples}")
    
    for split, info in split_info.items():
        status_icon = "✓" if "OK" in info['status'] else "❌"
        print(f"{status_icon} {split.capitalize()}: {info['size']} samples - {info['status']}")
    
    # Check directory structure
    print(f"\n=== Directory Structure ===")
    for root, dirs, files in os.walk(dataset_path):
        level = root.replace(dataset_path, '').count(os.sep)
        indent = ' ' * 2 * level
        folder_name = os.path.basename(root) or os.path.basename(dataset_path)
        print(f"{indent}{folder_name}/")
        
        # Show file count for each directory
        if files:
            subindent = ' ' * 2 * (level + 1)
            file_count = len(files)
            if file_count <= 5:
                for f in files[:5]:
                    print(f"{subindent}{f}")
            else:
                for f in files[:3]:
                    print(f"{subindent}{f}")
                print(f"{subindent}... and {file_count - 3} more files")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check dataset integrity")
    parser.add_argument("-d", "--dataset", type=str, default=None,
                      help="Path to dataset (defaults to DFC2023S)")
    
    args = parser.parse_args()
    check_dataset(args.dataset)
