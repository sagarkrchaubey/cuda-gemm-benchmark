#include "kernels.cuh"

#define TILE_SIZE 32
#define UNROLL 4

// Optimized GEMM kernel with shared memory, loop unrolling, and coalesced memory access
__global__ void optimized_gemm(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
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
        // Load tiles with coalesced memory access
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (col < N && t * TILE_SIZE + ty < K) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        __syncthreads();

        // Compute partial result with loop unrolling
#pragma unroll
        for (int k = 0; k < TILE_SIZE; k += UNROLL) {
            Cvalue += As[ty][k] * Bs[k][tx];
            if (k + 1 < TILE_SIZE) Cvalue += As[ty][k + 1] * Bs[k + 1][tx];
            if (k + 2 < TILE_SIZE) Cvalue += As[ty][k + 2] * Bs[k + 2][tx];
            if (k + 3 < TILE_SIZE) Cvalue += As[ty][k + 3] * Bs[k + 3][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = alpha * Cvalue;
    }
}
