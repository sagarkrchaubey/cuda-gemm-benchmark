# cuda-gemm-benchmark

A performance-focused CUDA project that implements and optimizes GEMM (General Matrix Multiplication) kernels from first principles and benchmarks them against NVIDIA cuBLAS using real GPU profiling metrics.

---

## Overview

This project explores GPU performance engineering by building matrix multiplication kernels incrementally, analyzing their efficiency, and comparing them with industry-grade implementations.

The goal is to understand:

- How GPU memory hierarchy affects performance
- How kernel design impacts throughput
- Why highly optimized libraries like cuBLAS outperform custom implementations

---

## Features

- Naive CUDA GEMM implementation
- Optimized CUDA kernels using:
  - Shared memory tiling
  - Coalesced memory access
  - Loop unrolling
- cuBLAS integration for baseline comparison
- Benchmarking with CUDA events
- Profiling with NVIDIA Nsight tools
- Performance analysis using:
  - Execution time
  - GFLOPS
  - Memory behavior
  - GPU utilization

---

## Project Structure

```text
cuda-gemm-benchmark/
├── src/
│   ├── main.cu
│   ├── naive_kernel.cu
│   ├── tiled_kernel.cu
│   ├── optimized_kernel.cu
│   ├── cublas_impl.cu
│   └── kernels.cuh
│
├── include/
│   ├── utils.cuh
│   ├── benchmark.cuh
│   └── matrix_ops.cuh
│
├── benchmark/
│   ├── benchmark.cpp
│   └── benchmark_config.json
│
├── scripts/
│   ├── run_tests.sh
│   └── run_benchmarks.sh
│
├── results/
│   ├── logs/
│   ├── csv/
│   └── plots/
│
├── docs/
│   ├── report.md
│   └── analysis.md
│
├── tests/
│   ├── correctness_tests.cpp
│   └── compare_with_cublas.cpp
│
├── Makefile
└── README.md