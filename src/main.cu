#include <iostream>
#include "../include/utils.cuh"
#include "../include/matrix_ops.cuh"
#include "../include/benchmark.cuh"
#include "kernels.cuh"

int main() {

    int N = 4096;

    size_t size = N * N * sizeof(float);

    // Allocate host memory
    float* h_A = allocate_matrix_host(N, N);
    float* h_B = allocate_matrix_host(N, N);
    float* h_C = allocate_matrix_host(N, N);

    // Initialize
    initialize_matrix(h_A, N, N);
    initialize_matrix(h_B, N, N);

    // Allocate device memory
    float* d_A = allocate_matrix_device(N, N);
    float* d_B = allocate_matrix_device(N, N);
    float* d_C = allocate_matrix_device(N, N);

    // Copy to GPU
    copy_host_to_device(h_A, d_A, N, N);
    copy_host_to_device(h_B, d_B, N, N);


//This removes cuBLAS initialization overhead
launch_naive(d_A, d_B, d_C, N);
launch_cublas(d_A, d_B, d_C, N);
cudaDeviceSynchronize();


    // Run benchmarks
    BenchmarkResult naive_result = run_benchmark(launch_naive, d_A, d_B, d_C, N);
    print_result("Naive", naive_result);

    BenchmarkResult tiled_result = run_benchmark(launch_tiled, d_A, d_B, d_C, N);
    print_result("Tiled", tiled_result);

    BenchmarkResult cublas_result = run_benchmark(launch_cublas, d_A, d_B, d_C, N);
    print_result("cuBLAS", cublas_result);

    std::cout << "\n--- Speedups ---\n";
    std::cout << "Naive → Tiled: " 
          << naive_result.time_ms / tiled_result.time_ms << "x\n";

    std::cout << "Tiled → cuBLAS: "
          << tiled_result.time_ms / cublas_result.time_ms << "x\n";

    // Cleanup
    deallocate_matrix_device(d_A);
    deallocate_matrix_device(d_B);
    deallocate_matrix_device(d_C);

    deallocate_matrix_host(h_A);
    deallocate_matrix_host(h_B);
    deallocate_matrix_host(h_C);

    return 0;
}
