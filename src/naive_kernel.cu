#include "../include/utils.cuh"
#include "kernels.cuh"

// CUDA kernel
__global__ void naive_gemm_kernel(float* A, float* B, float* C, int N) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {

        float sum = 0.0f;

        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

// Launcher (important abstraction)
void launch_naive(float* d_A, float* d_B, float* d_C, int N) {

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);

    naive_gemm_kernel<<<blocks, threads>>>(d_A, d_B, d_C, N);

    CUDA_CHECK(cudaDeviceSynchronize());
}
