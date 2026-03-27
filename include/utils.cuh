#ifndef UTILS_CUH
#define UTILS_CUH

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                    __FILE__, __LINE__, error, cudaGetErrorName(error), #call); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Initialize matrix with random values
void initialize_matrix(float* matrix, int rows, int cols);

// Verify correctness of GEMM result
void verify_gemm_result(int M, int N, int K, float alpha, const float* A, const float* B, float beta, const float* C_result, float tolerance);

// Print matrix (for small matrices)
void print_matrix(const float* matrix, int rows, int cols, const char* name);

#endif // UTILS_CUH
