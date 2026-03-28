#include <cublas_v2.h>
#include "../include/utils.cuh"
#include "kernels.cuh"

void launch_cublas(float* d_A, float* d_B, float* d_C, int N) {

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    // Note: column-major internally
    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, N, N,
        &alpha,
        d_B, N,
        d_A, N,
        &beta,
        d_C, N
    );

    cublasDestroy(handle);

    CUDA_CHECK(cudaDeviceSynchronize());
}
