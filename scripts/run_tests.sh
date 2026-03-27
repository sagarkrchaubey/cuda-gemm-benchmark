#!/bin/bash

set -e

echo "Building CUDA GEMM Benchmark Project..."
make clean
make build

echo ""
echo "Running Correctness Tests..."
./build/test_correctness

echo ""
echo "Running cuBLAS Comparison Tests..."
./build/test_cublas_compare

echo ""
echo "All tests completed!"
