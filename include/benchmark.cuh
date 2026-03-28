#ifndef BENCHMARK_CUH
#define BENCHMARK_CUH

struct BenchmarkResult {
    float time_ms;
    float gflops;
};

typedef void (*KernelFunc)(float*, float*, float*, int);

BenchmarkResult run_benchmark(
    KernelFunc kernel,
    float* d_A,
    float* d_B,
    float* d_C,
    int N);

void print_result(const char* name, BenchmarkResult result);

#endif
