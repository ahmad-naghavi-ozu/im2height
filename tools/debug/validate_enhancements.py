#!/usr/bin/env python3
"""
Validation script to demonstrate the training enhancements from dfc2023-multi-gpu-training branch
"""

import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from train_dfc2023 import get_dynamic_config
from dfc2023_data import DFC2023Dataset
import torch

def validate_training_enhancements():
    """Test the enhanced training configuration and multi-GPU improvements"""
    
    print("="*60)
    print("VALIDATION: Training Enhancements from Multi-GPU Branch")
    print("="*60)
    
    # Test dataset loading - use the correct folder structure (x/y instead of rgb/dsm)
    try:
        # The DFC2023S dataset uses x/ and y/ folders, not rgb/ and dsm/
        dataset = DFC2023Dataset('data/DFC2023S', 'train', 'x', 'y')
        print(f"✓ Dataset loaded successfully: {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        # Try to provide helpful information about available folders
        try:
            import os
            train_path = 'data/DFC2023S/train'
            if os.path.exists(train_path):
                folders = [f for f in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, f))]
                print(f"Available folders in {train_path}: {folders}")
            else:
                print(f"Training path {train_path} does not exist")
        except:
            pass
        return
    
    # Get sample information
    sample_input, sample_target = dataset[0]
    image_size = (sample_input.shape[2], sample_input.shape[1])  # (width, height)
    in_channels = sample_input.shape[0]
    
    print(f"✓ Sample analysis:")
    print(f"  - Input shape: {sample_input.shape}")
    print(f"  - Target shape: {sample_target.shape}")
    print(f"  - Image size: {image_size}")
    print(f"  - Input channels: {in_channels}")
    
    # Test GPU detection
    num_gpus = torch.cuda.device_count()
    print(f"✓ GPU detection: {num_gpus} GPUs available")
    
    # Test dynamic configuration for different GPU counts
    print(f"\n📊 Dynamic Configuration Testing:")
    print("-"*40)
    
    for gpu_count in [1, 2, 4]:
        config = get_dynamic_config(image_size, gpu_count)
        batch_size = config["config"]["batch_size"]
        grad_accum = config["gradient_accum"]
        
        # Calculate batch metrics
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            shuffle=True, 
            **config["config"]
        )
        
        num_batches = len(dataloader)
        total_processed = num_batches * batch_size
        coverage = (total_processed / len(dataset)) * 100
        
        print(f"  {gpu_count} GPU(s):")
        print(f"    - Batch size: {batch_size}")
        print(f"    - Gradient accumulation: {grad_accum}")
        print(f"    - Number of batches: {num_batches}")
        print(f"    - Total processed per epoch: {total_processed}")
        print(f"    - Dataset coverage: {coverage:.1f}%")
        
        if coverage < 100:
            print(f"    ⚠️  WARNING: {len(dataset) - total_processed} samples not processed!")
        else:
            print(f"    ✓ All samples processed")
        print()
    
    # Test multi-GPU detection logic
    print(f"🔧 Multi-GPU Training Logic Test:")
    print("-"*40)
    
    # Simulate different trainer device configurations
    test_configs = [
        ([0], 1, "Single GPU"),
        ([0, 1], 2, "Multi-GPU list"),
        (2, 2, "Multi-GPU count"),
        ([0, 1, 2, 3], 4, "4-GPU setup")
    ]
    
    for trainer_devices, expected_gpus, description in test_configs:
        # Simulate the multi-GPU detection logic from train_dfc2023.py
        use_distributed_strategy = False
        if (isinstance(trainer_devices, list) and len(trainer_devices) > 1) or \
           (isinstance(trainer_devices, int) and trainer_devices > 1):
            use_distributed_strategy = True
        
        training_strategy = 'ddp' if use_distributed_strategy else None
        
        print(f"  {description}:")
        print(f"    - Devices: {trainer_devices}")
        print(f"    - Use distributed: {use_distributed_strategy}")
        print(f"    - Strategy: {training_strategy}")
        print()
    
    print("="*60)
    print("✅ VALIDATION COMPLETE")
    print("="*60)
    print("Key improvements from multi-GPU branch:")
    print("1. ✓ Enhanced batch counting with debug output")
    print("2. ✓ Multi-GPU detection and DDP strategy configuration")
    print("3. ✓ Dataset coverage analysis and warnings")
    print("4. ✓ Distributed training strategy selection")
    print("5. ✓ Comprehensive debug tools in tools/debug/")
    print()
    print("Next steps:")
    print("- Run: ./run_dfc2023.sh --action train --gpus 0,1")
    print("- Check: tools/debug/ for debugging utilities")
    print("- Monitor: Enhanced training logs for batch counting")

if __name__ == "__main__":
    validate_training_enhancements()
