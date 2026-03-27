#include "kernels.cuh"

#define TILE_SIZE 32

// Tiled GEMM kernel with shared memory optimization
// Uses shared memory to reduce global memory bandwidth
__global__ void tiled_gemm(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float Cvalue = beta * C[row * N + col];

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        As[ty][tx] = (row < M && t * TILE_SIZE + tx < K) ? A[row * K + t * TILE_SIZE + tx] : 0.0f;
        Bs[ty][tx] = (col < N && t * TILE_SIZE + ty < K) ? B[(t * TILE_SIZE + ty) * N + col] : 0.0f;
        __syncthreads();

        // Compute partial result
        for (int k = 0; k < TILE_SIZE; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = alpha * Cvalue;
    }
}
