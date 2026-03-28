#include "../include/utils.cuh"
#include "kernels.cuh"

#define TILE_SIZE 16

__global__ void tiled_gemm_kernel(float* A, float* B, float* C, int N) {

    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < N / TILE_SIZE; t++) {

        // Load tiles into shared memory
        tileA[threadIdx.y][threadIdx.x] = A[row * N + (t * TILE_SIZE + threadIdx.x)];
        tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    C[row * N + col] = sum;
}

void launch_tiled(float* d_A, float* d_B, float* d_C, int N) {

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(N / TILE_SIZE, N / TILE_SIZE);

    tiled_gemm_kernel<<<blocks, threads>>>(d_A, d_B, d_C, N);

    CUDA_CHECK(cudaDeviceSynchronize());
}
