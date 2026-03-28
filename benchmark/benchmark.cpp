#include <iostream>
#include <cuda_runtime.h>
#include "../include/benchmark.cuh"

BenchmarkResult run_benchmark(
    KernelFunc kernel,
    float* d_A,
    float* d_B,
    float* d_C,
    int N) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

cudaDeviceSynchronize();

cudaEventRecord(start);

kernel(d_A, d_B, d_C, N);

cudaDeviceSynchronize();

cudaEventRecord(stop);
cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    float flops = 2.0f * N * N * N;
    float gflops = (flops / (ms / 1000.0f)) / 1e9;

    BenchmarkResult result;
    result.time_ms = ms;
    result.gflops = gflops;

    return result;
}

void print_result(const char* name, BenchmarkResult result) {
    std::cout << name
              << " | Time: " << result.time_ms
              << " ms | GFLOPS: " << result.gflops
              << std::endl;
}
