#include <cublas_v2.h>
#include <stdio.h>

// Wrapper for cuBLAS GEMM for benchmarking comparison
void cublas_gemm(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    // cuBLAS uses column-major ordering
    // For row-major matrices, we need to adjust parameters
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                d_B, N,
                d_A, K,
                &beta,
                d_C, N);

    cublasDestroy(handle);
}
