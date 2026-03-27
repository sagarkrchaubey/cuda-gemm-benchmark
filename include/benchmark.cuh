#ifndef BENCHMARK_CUH
#define BENCHMARK_CUH

#include <cuda_runtime.h>

// Benchmark result structure
struct BenchmarkResult {
    float execution_time_ms;
    float gflops;
    float memory_bandwidth_gb_s;
    float achieved_occupancy;
};

// Run benchmark and return metrics
BenchmarkResult benchmark_kernel(
    void (*kernel_func)(int, int, int, float, const float*, const float*, float, float*),
    int M, int N, int K,
    float alpha, float* d_A, float* d_B, float beta, float* d_C,
    int num_iterations);

// Print benchmark results
void print_benchmark_results(const char* kernel_name, const BenchmarkResult& result);

// Compare two kernels
void compare_kernels(
    const char* name1, void (*kernel1)(int, int, int, float, const float*, const float*, float, float*),
    const char* name2, void (*kernel2)(int, int, int, float, const float*, const float*, float, float*),
    int M, int N, int K);

#endif // BENCHMARK_CUH
