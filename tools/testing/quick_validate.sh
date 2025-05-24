#!/bin/bash
# Quick validation of core functionality

echo "=== Quick Im2Height Validation ==="

# Test basic syntax
echo "1. Bash syntax check..."
bash -n run.sh && echo "   ✓ run.sh syntax OK" || echo "   ✗ Syntax error"

# Test basic functionality
echo "2. Testing info action..."
./run.sh --action info --dataset data/DFC2023Amini --quiet > /tmp/info_test.log 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ Info action works"
else
    echo "   ✗ Info action failed"
    cat /tmp/info_test.log
fi

# Test argument validation
echo "3. Testing argument consistency..."
echo "   run.sh uses --gpu_count for train action"
grep -q "gpu_count" run.sh && echo "   ✓ run.sh uses --gpu_count" || echo "   ✗ Missing --gpu_count"

echo "   train.py accepts --gpu_count"
grep -q "gpu_count" train.py && echo "   ✓ train.py has --gpu_count" || echo "   ✗ Missing --gpu_count"

echo "   predict.py uses CUDA_VISIBLE_DEVICES (no --gpu argument needed)"
grep -q "CUDA_VISIBLE_DEVICES" run.sh && echo "   ✓ Predict action uses env var" || echo "   ✗ Missing env var"

echo "4. Testing quiet flag..."
echo "   train.py has --quiet flag"
grep -q "quiet" train.py && echo "   ✓ train.py has --quiet" || echo "   ✗ Missing --quiet"

echo "   predict.py has --quiet flag"
grep -q "quiet" predict.py && echo "   ✓ predict.py has --quiet" || echo "   ✗ Missing --quiet"

echo ""
echo "=== Validation Complete ==="
