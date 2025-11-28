#!/bin/bash

# Create build directory
mkdir -p build
mkdir -p test

# Build the project
cd build
cmake ..
make -j$(nproc)

echo ""
echo "Build complete! Executable: ./sgemm"
echo ""
echo "Usage: ./sgemm <kernel_num>"
echo "  kernel_num: 0 (cuBLAS), 1-7 (custom kernels)"
echo ""
echo "Example: ./sgemm 1  # Test kernel 1 (naive)"
