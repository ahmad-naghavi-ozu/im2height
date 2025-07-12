#!/usr/bin/env python3
"""Test the improved batch size calculation"""

import sys
sys.path.append('/home/asfand/Ahmad/IM2HEIGHT')

from train import get_dynamic_config

print('=== Improved Batch Size Calculation Test ===')

# Test the key scenarios
test_cases = [
    ((256, 256), 2, 'Original size, 2 GPUs'),
    ((512, 512), 2, 'DFC2019 size, 2 GPUs (should be around 8)'),
    ((500, 500), 2, 'Dublin size, 2 GPUs'),
    ((1024, 1024), 2, 'Large images, 2 GPUs'),
]

for size, gpus, desc in test_cases:
    config = get_dynamic_config(image_size=size, num_gpus=gpus, quiet=True)
    batch_size = config['config']['batch_size']
    grad_accum = config['gradient_accum']
    effective = batch_size * grad_accum
    memory_factor = (size[0] * size[1]) / (256 * 256)
    print(f'{desc}:')
    print(f'  Memory factor: {memory_factor:.1f}x')
    print(f'  Batch size: {batch_size}')
    print(f'  Gradient accumulation: {grad_accum}')
    print(f'  Effective batch: {effective}')
    print()
