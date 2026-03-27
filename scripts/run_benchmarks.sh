#!/bin/bash

set -e

echo "Building CUDA GEMM Benchmark Project..."
make clean
make build

echo ""
echo "Running benchmarks..."

# Small matrices
echo "Small matrix benchmarks (256x256x256 to 512x512x512)..."
./build/benchmark -m 256 -n 256 -k 256 -i 100 -v -o results/csv/small_256.csv
./build/benchmark -m 512 -n 512 -k 512 -i 100 -v -o results/csv/small_512.csv

# Medium matrices
echo "Medium matrix benchmarks (1024x1024x1024 to 2048x2048x2048)..."
./build/benchmark -m 1024 -n 1024 -k 1024 -i 50 -v -o results/csv/medium_1024.csv
./build/benchmark -m 2048 -n 2048 -k 2048 -i 50 -v -o results/csv/medium_2048.csv

# Large matrices
echo "Large matrix benchmarks (4096x4096x4096)..."
./build/benchmark -m 4096 -n 4096 -k 4096 -i 10 -v -o results/csv/large_4096.csv

echo ""
echo "Benchmarks completed! Results saved to results/csv/"
