#ifndef KERNELS_CUH
#define KERNELS_CUH

// Naive kernel launcher
void launch_naive(float* d_A, float* d_B, float* d_C, int N);

// Tiled
void launch_tiled(float* d_A, float* d_B, float* d_C, int N);


// cuBLAS wrapper
void launch_cublas(float* d_A, float* d_B, float* d_C, int N);

#endif
