#!/usr/bin/env bash
# =============================================================================
# scripts/benchmark.sh
#
# Builds all CPU implementations and runs the unified benchmark.
# Results are saved as CSV in results/benchmarks/.
#
# Usage:
#   cd matrix-multiplication-optimization
#   bash scripts/benchmark.sh [--quick]
#
# Flags:
#   --quick   only test sizes up to 512 (faster for development)
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "========================================================"
echo "  Matrix Multiplication Benchmark"
echo "  $(date)"
echo "========================================================"

mkdir -p results/benchmarks

# ── Compiler flags
CXX="g++"
CXXFLAGS="-O2 -std=c++17 -mavx2 -mfma -fopenmp -I include"
MPICXX="mpicxx"
MPICXXFLAGS="-O2 -std=c++17 -mavx2 -mfma -fopenmp -I include"

# ── 1. Build everything
echo ""
echo "► Building all implementations..."

$CXX $CXXFLAGS -o bin/naive        src/naive.cpp      2>&1
$CXX $CXXFLAGS -o bin/reordered    src/reordered.cpp  2>&1
$CXX $CXXFLAGS -o bin/blocked      src/blocked.cpp    2>&1
$CXX $CXXFLAGS -o bin/openmp_matmul src/openmp.cpp    2>&1
$CXX $CXXFLAGS -o bin/avx_matmul   src/avx.cpp        2>&1
$CXX $CXXFLAGS -o bin/benchmark    benchmark/benchmark.cpp 2>&1
$MPICXX $MPICXXFLAGS -o bin/mpi_matmul    src/mpi.cpp     2>&1
$MPICXX $MPICXXFLAGS -o bin/hybrid_matmul src/hybrid.cpp  2>&1

# CUDA (optional – skip if nvcc not found)
if command -v nvcc &>/dev/null; then
    nvcc -O2 -arch=sm_75 -std=c++17 -I include \
         src/cuda.cu -o bin/cuda_matmul 2>&1 && \
         echo "    nvcc: cuda_matmul built" || \
         echo "    nvcc: build failed (skipping)"
fi

echo "    All builds complete."

# ── 2. Unified CPU benchmark
echo ""
echo "► Running unified CPU benchmark..."
QUICK="${1:-}"
./bin/benchmark $QUICK | tee results/benchmarks/cpu_results.csv
echo ""
echo "  CSV saved to results/benchmarks/cpu_results.csv"

# ── 3. Individual runs (standalone executables)
echo ""
echo "► Standalone runs (N=1024)..."

echo ""
echo "--- Naive ---"
./bin/naive 1024

echo ""
echo "--- Loop Reordered ---"
./bin/reordered 1024

echo ""
echo "--- Cache Blocked ---"
./bin/blocked 1024

echo ""
echo "--- OpenMP (thread scaling) ---"
./bin/openmp_matmul 1024

echo ""
echo "--- AVX SIMD ---"
./bin/avx_matmul 1024

# ── 4. MPI scaling
echo ""
echo "► MPI scaling sweep..."
for NP in 1 2 4 8; do
    echo ""
    echo "--- MPI np=$NP ---"
    mpirun -np $NP --oversubscribe ./bin/mpi_matmul 1024
done

# ── 5. CUDA (if built)
if [ -f bin/cuda_matmul ]; then
    echo ""
    echo "► CUDA benchmark..."
    for SZ in 512 1024 2048 4096; do
        echo ""
        echo "--- CUDA N=$SZ ---"
        ./bin/cuda_matmul $SZ
    done
fi

echo ""
echo "========================================================"
echo "  Benchmark complete. Results in results/benchmarks/"
echo "========================================================"
