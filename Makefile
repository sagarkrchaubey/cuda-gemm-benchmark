CUDA_PATH ?= /usr/local/cuda
CUDA_ARCH ?= -arch=sm_50

CC := gcc
NVCC := $(CUDA_PATH)/bin/nvcc
CFLAGS := -Wall -Wextra -O3
NVCCFLAGS := $(CUDA_ARCH) -O3 --ptxas-options=-v
LDFLAGS := -lcublas -lcudart

INCLUDE_DIRS := -I./include -I$(CUDA_PATH)/include
LIB_DIRS := -L$(CUDA_PATH)/lib64

BUILD_DIR := build
SRC_DIR := src
INCLUDE_DIR := include
TEST_DIR := tests
BENCH_DIR := benchmark

# Source files
CUDA_SOURCES := $(SRC_DIR)/main.cu $(SRC_DIR)/naive_kernel.cu $(SRC_DIR)/tiled_kernel.cu $(SRC_DIR)/optimized_kernel.cu $(SRC_DIR)/cublas_impl.cu
CUDA_OBJECTS := $(CUDA_SOURCES:%.cu=$(BUILD_DIR)/%.o)

TEST_SOURCES := $(TEST_DIR)/correctness_tests.cpp
TEST_OBJECTS := $(TEST_SOURCES:%.cpp=$(BUILD_DIR)/%.o) $(CUDA_OBJECTS)

BENCH_SOURCES := $(BENCH_DIR)/benchmark.cpp
BENCH_OBJECTS := $(BENCH_SOURCES:%.cpp=$(BUILD_DIR)/%.o)

# Targets
TARGETS := $(BUILD_DIR)/main $(BUILD_DIR)/test_correctness $(BUILD_DIR)/test_cublas_compare $(BUILD_DIR)/benchmark

.PHONY: all build clean

all: build

build: $(BUILD_DIR) $(TARGETS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/$(SRC_DIR)
	mkdir -p $(BUILD_DIR)/$(TEST_DIR)
	mkdir -p $(BUILD_DIR)/$(BENCH_DIR)

$(BUILD_DIR)/%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_DIRS) -c $< -o $@

$(BUILD_DIR)/%.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDE_DIRS) -c $< -o $@

$(BUILD_DIR)/main: $(CUDA_OBJECTS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LIB_DIRS) $(LDFLAGS)

$(BUILD_DIR)/test_correctness: $(TEST_OBJECTS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LIB_DIRS) $(LDFLAGS)

$(BUILD_DIR)/test_cublas_compare: $(TEST_DIR)/compare_with_cublas.cpp $(CUDA_OBJECTS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LIB_DIRS) $(LDFLAGS)

$(BUILD_DIR)/benchmark: $(BENCH_OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LIB_DIRS) $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

help:
	@echo "CUDA GEMM Benchmark Makefile"
	@echo "============================"
	@echo "Targets:"
	@echo "  make build              - Build all targets"
	@echo "  make clean              - Remove build artifacts"
	@echo "  make help               - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  CUDA_PATH               - Path to CUDA installation (default: /usr/local/cuda)"
	@echo "  CUDA_ARCH               - CUDA architecture (default: -arch=sm_50)"
