#ifndef MATRIX_OPS_CUH
#define MATRIX_OPS_CUH

#include <cuda_runtime.h>
#include "utils.cuh"

inline float* allocate_matrix_host(int rows, int cols) {
    return (float*)malloc(rows * cols * sizeof(float));
}

inline float* allocate_matrix_device(int rows, int cols) {
    float* d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, rows * cols * sizeof(float)));
    return d_ptr;
}

inline void deallocate_matrix_host(float* matrix) {
    free(matrix);
}

inline void deallocate_matrix_device(float* matrix) {
    CUDA_CHECK(cudaFree(matrix));
}

inline void copy_host_to_device(const float* h, float* d, int rows, int cols) {
    CUDA_CHECK(cudaMemcpy(d, h, rows * cols * sizeof(float),
                          cudaMemcpyHostToDevice));
}

#endif
