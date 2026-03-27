#ifndef MATRIX_OPS_CUH
#define MATRIX_OPS_CUH

#include <cuda_runtime.h>

// Matrix allocation and deallocation
float* allocate_matrix_device(int rows, int cols);
float* allocate_matrix_host(int rows, int cols);
void deallocate_matrix_device(float* matrix);
void deallocate_matrix_host(float* matrix);

// Data transfer
void copy_host_to_device(const float* h_matrix, float* d_matrix, int rows, int cols);
void copy_device_to_host(const float* d_matrix, float* h_matrix, int rows, int cols);

// Matrix operations
void matrix_multiply_cpu(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C);

#endif // MATRIX_OPS_CUH
