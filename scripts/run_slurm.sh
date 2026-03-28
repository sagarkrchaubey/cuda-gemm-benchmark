#!/bin/bash
#SBATCH --job-name=cuda_gemm
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=results/logs/output_%j.log
#SBATCH --error=results/logs/error_%j.err

echo "============================================"
echo "CUDA GEMM Benchmark Job Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "============================================"

# ==============================
# Environment Setup
# ==============================

module purge

module load cuda/11.0

echo "=== Environment Info ==="
gcc --version
nvcc --version
nvidia-smi
echo "========================"

# ==============================
# Move to project directory
# ==============================

cd $SLURM_SUBMIT_DIR

# ==============================
# Ensure directories exist
# ==============================

mkdir -p bin
mkdir -p results/logs

# ==============================
# Build (important for cluster)
# ==============================

echo "Building project..."
make clean
make

# ==============================
# Run Benchmark
# ==============================

echo "Running benchmark..."

./bin/benchmark

echo "============================================"
echo "Job Finished"
echo "============================================"
