#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Forward declarations
extern void optimized_gemm(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C);
extern void cublas_gemm(int M, int N, int K, float alpha, const float* d_A, const float* d_B, float beta, float* d_C);

void compare_with_cublas(int M, int N, int K) {
    printf("Comparing with cuBLAS: M=%d, N=%d, K=%d\n", M, N, K);

    float alpha = 1.0f, beta = 0.0f;

    // Allocate host memory
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C_opt = (float*)malloc(M * N * sizeof(float));
    float *h_C_cublas = (float*)malloc(M * N * sizeof(float));

    // Initialize matrices
    for (int i = 0; i < M * K; i++) h_A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < M * N; i++) {
        h_C_opt[i] = 0.0f;
        h_C_cublas[i] = 0.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C_opt, *d_C_cublas;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C_opt, M * N * sizeof(float));
    cudaMalloc(&d_C_cublas, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_opt, h_C_opt, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_cublas, h_C_cublas, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // Run kernels and time them
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Time optimized kernel
    cudaEventRecord(start);
    // TODO: Launch optimized_gemm kernel
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float opt_time;
    cudaEventElapsedTime(&opt_time, start, stop);

    // Time cuBLAS
    cudaEventRecord(start);
    cublas_gemm(M, N, K, alpha, d_A, d_B, beta, d_C_cublas);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float cublas_time;
    cudaEventElapsedTime(&cublas_time, start, stop);

    printf("  Optimized GEMM: %.2f ms\n", opt_time);
    printf("  cuBLAS GEMM:    %.2f ms\n", cublas_time);
    printf("  Performance ratio: %.2f%%\n", (opt_time / cublas_time) * 100.0f);

    // Verify correctness
    cudaMemcpy(h_C_opt, d_C_opt, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_cublas, d_C_cublas, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    float max_error = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float error = fabs(h_C_opt[i] - h_C_cublas[i]);
        if (error > max_error) max_error = error;
    }
    printf("  Max difference: %e\n\n", max_error);

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C_opt);
    free(h_C_cublas);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_opt);
    cudaFree(d_C_cublas);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("CUDA GEMM vs cuBLAS Comparison\n");
    printf("==============================\n\n");

    // Test different matrix sizes
    compare_with_cublas(512, 512, 512);
    compare_with_cublas(1024, 1024, 1024);
    compare_with_cublas(2048, 2048, 2048);

    return 0;
}
