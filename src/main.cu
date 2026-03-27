#include <stdio.h>
#include <cuda_runtime.h>
#include "kernels.cuh"
#include "benchmark.cuh"
#include "utils.cuh"

int main(int argc, char** argv) {
    printf("CUDA GEMM Benchmark\n");
    printf("===================\n\n");

    // Default matrix size
    int M = 1024, N = 1024, K = 1024;
    
    if (argc > 1) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }

    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);

    // Initialize matrices
    float *d_A, *d_B, *d_C;
    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);

    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C, bytesC);

    // TODO: Initialize matrices and run benchmarks

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
