#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>

// Naive GEMM kernel
__global__ void naive_gemm(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C);

// Tiled GEMM kernel with shared memory
__global__ void tiled_gemm(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C);

// Optimized GEMM kernel with loop unrolling and coalescing
__global__ void optimized_gemm(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C);

#endif // KERNELS_CUH
