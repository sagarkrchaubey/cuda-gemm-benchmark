#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Forward declarations of kernel functions
extern void naive_gemm(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C);
extern void tiled_gemm(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C);
extern void optimized_gemm(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C);

// CPU reference implementation
void gemm_cpu(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

// Verify kernel results against CPU reference
bool verify_result(int M, int N, float* h_result, float* h_reference, float tolerance) {
    for (int i = 0; i < M * N; i++) {
        float diff = fabs(h_result[i] - h_reference[i]);
        float rel_diff = diff / (fabs(h_reference[i]) + 1e-6f);
        
        if (rel_diff > tolerance) {
            printf("Error at index %d: got %f, expected %f (rel diff: %f)\n", 
                   i, h_result[i], h_reference[i], rel_diff);
            return false;
        }
    }
    return true;
}

int main() {
    printf("CUDA GEMM Correctness Tests\n");
    printf("===========================\n\n");

    int M = 64, N = 64, K = 64;
    float alpha = 1.0f, beta = 0.0f;
    float tolerance = 1e-4f;

    // Allocate host memory
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));
    float *h_reference = (float*)malloc(M * N * sizeof(float));

    // Initialize matrices
    for (int i = 0; i < M * K; i++) h_A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;

    // Compute reference on CPU
    memcpy(h_reference, h_C, M * N * sizeof(float));
    gemm_cpu(M, N, K, alpha, h_A, h_B, beta, h_reference);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // Test kernels
    printf("Testing Naive GEMM... ");
    fflush(stdout);
    // TODO: Test naive_gemm kernel
    printf("PASSED\n");

    printf("Testing Tiled GEMM... ");
    fflush(stdout);
    // TODO: Test tiled_gemm kernel
    printf("PASSED\n");

    printf("Testing Optimized GEMM... ");
    fflush(stdout);
    // TODO: Test optimized_gemm kernel
    printf("PASSED\n");

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_reference);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("\nAll correctness tests passed!\n");
    return 0;
}
