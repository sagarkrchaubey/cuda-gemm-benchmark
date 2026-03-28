# ==============================
# Project Structure
# ==============================

SRC_DIR = src
BENCH_DIR = benchmark
BIN_DIR = bin

TARGET = $(BIN_DIR)/benchmark

# ==============================
# Environment (Cluster-specific)
# ==============================

GPU_ENV = module purge && \
          . /home/apps/spack/share/spack/setup-env.sh && \
          spack load gcc@8.5.0 && \
          module load cuda/11.0

# ==============================
# Compiler
# ==============================

NVCC = nvcc

# ==============================
# Flags
# ==============================

CUDA_FLAGS = -O3 -arch=sm_70
LIBS = -lcublas

# ==============================
# Source Files
# ==============================

SRC = \
$(SRC_DIR)/main.cu \
$(SRC_DIR)/naive_kernel.cu \
$(SRC_DIR)/cublas_impl.cu \
$(SRC_DIR)/tiled_kernel.cu \
$(BENCH_DIR)/benchmark.cpp

# ==============================
# Targets
# ==============================

all: dir build

dir:
	@mkdir -p $(BIN_DIR)
	@mkdir -p results/logs

build:
	@echo "Building CUDA GEMM Benchmark..."
	$(NVCC) $(CUDA_FLAGS) $(SRC) -o $(TARGET) $(LIBS)

run:
	@$(GPU_ENV) && ./$(TARGET)

clean:
	rm -rf $(BIN_DIR)

