# MatMul-Bench: Dense Matrix Multiplication Optimization & Performance Study

A high-performance C++ matrix multiplication benchmark and optimization study. This repository tracks dense matrix multiplication performance starting from a naive O(N^3) baseline up through cache-blocking, OpenMP multithreading, AVX2 SIMD intrinsics, NVIDIA CUDA GPU acceleration, MPI distributed memory, and hybrid MPI+OpenMP parallelism.

## Quick Start

### Build All Executables
```bash
make all
```

### Run Benchmarks & Analysis Scripts
```bash
# Run unified CPU benchmark (CSV output)
make benchmark

# Run Linux perf hardware counter analysis
make perf

# Run Valgrind Cachegrind simulation
make cachegrind
```

### Individual Target Execution
```bash
N=1024 make run-naive
N=1024 make run-reordered
N=1024 make run-blocked
N=1024 make run-openmp
N=1024 make run-avx
N=1024 make run-mpi
N=2048 make run-cuda
```

## Summary Performance Benchmark (N = 1024)

Tested on Intel Core i5-9300H CPU @ 2.40GHz (4 physical cores) & NVIDIA GeForce GTX 1650 GPU (896 CUDA cores):

| Implementation | Source File | Time (ms) | Throughput (GFLOP/s) | Speedup vs Naive | Key Optimization Mechanism |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive (`i-j-k`)** | [`src/naive.cpp`](file:///mnt/SSD_512_win/Linux/College-Backup/Projects-Aug/matrix-multiplication-optimization/src/naive.cpp) | 2203.24 ms | 0.97 GFLOP/s | 1.00x | Textbook baseline. |
| **Loop Reordered (`i-k-j`)** | [`src/reordered.cpp`](file:///mnt/SSD_512_win/Linux/College-Backup/Projects-Aug/matrix-multiplication-optimization/src/reordered.cpp) | 543.00 ms | 3.96 GFLOP/s | **4.05x** | Spatial cache-line locality on rows of B and C. |
| **Cache Blocked (`256x256`)** | [`src/blocked.cpp`](file:///mnt/SSD_512_win/Linux/College-Backup/Projects-Aug/matrix-multiplication-optimization/src/blocked.cpp) | 674.50 ms | 3.18 GFLOP/s | **3.27x** | Working sub-tiles fit inside L2/L3 cache. |
| **OpenMP (4 Cores)** | [`src/openmp.cpp`](file:///mnt/SSD_512_win/Linux/College-Backup/Projects-Aug/matrix-multiplication-optimization/src/openmp.cpp) | 232.90 ms | 9.22 GFLOP/s | **9.46x** | Multithreaded parallel loop slicing. |
| **AVX2 SIMD (`i-k-j`)** | [`src/avx.cpp`](file:///mnt/SSD_512_win/Linux/College-Backup/Projects-Aug/matrix-multiplication-optimization/src/avx.cpp) | 114.32 ms | 18.79 GFLOP/s | **19.27x** | 256-bit SIMD registers + FMA3 (8 floats/cycle). |
| **CUDA Naive** | [`src/cuda.cu`](file:///mnt/SSD_512_win/Linux/College-Backup/Projects-Aug/matrix-multiplication-optimization/src/cuda.cu) | 7.49 ms | 286.85 GFLOP/s | **294.16x** | 896 CUDA cores parallel execution. |
| **CUDA Tiled (Shared Mem)**| [`src/cuda.cu`](file:///mnt/SSD_512_win/Linux/College-Backup/Projects-Aug/matrix-multiplication-optimization/src/cuda.cu) | **5.84 ms** | **367.92 GFLOP/s** | **377.27x** | On-chip Shared Memory SRAM tiling (32x32). |
| **MPI (4 Processes)** | [`src/mpi.cpp`](file:///mnt/SSD_512_win/Linux/College-Backup/Projects-Aug/matrix-multiplication-optimization/src/mpi.cpp) | 354.08 ms | 6.20 GFLOP/s | **6.22x** | Row-sliced message passing distribution. |
| **Hybrid MPI + OpenMP** | [`src/hybrid.cpp`](file:///mnt/SSD_512_win/Linux/College-Backup/Projects-Aug/matrix-multiplication-optimization/src/hybrid.cpp) | 340.95 ms | 6.30 GFLOP/s | **6.46x** | Inter-node MPI + intra-node OpenMP threads. |

## Repository Structure

```text
matrix-multiplication-optimization/
├── include/
│   └── matrix_utils.hpp     # Aligned memory allocation (posix_memalign), timers, GFLOP/s
├── src/
│   ├── naive.cpp            # Baseline i-j-k matrix multiply
│   ├── reordered.cpp        # Loop order permutations (i-j-k, i-k-j, j-k-i)
│   ├── blocked.cpp          # Cache-tiled sub-block matrix multiply
│   ├── openmp.cpp           # Shared-memory OpenMP multithreading
│   ├── avx.cpp              # 256-bit AVX2 SIMD intrinsics + FMA3
│   ├── cuda.cu              # CUDA naive & shared memory tiled kernels
│   ├── mpi.cpp              # Distributed memory MPI row distribution
│   └── hybrid.cpp           # MPI + OpenMP hybrid parallelism
├── benchmark/
│   └── benchmark.cpp        # Unified CSV benchmark driver
├── scripts/
│   ├── benchmark.sh         # Full automated benchmark suite
│   ├── perf.sh              # Linux perf hardware counter script
│   └── cachegrind.sh        # Valgrind Cachegrind simulation script
├── ANALYSIS_REPORT.md       # Comprehensive empirical performance report
├── notes.md                 # Technical reference & profiling guide
├── Makefile                 # Build system configuration
└── README.md
```

## Hardware Profiling & Verification

### 1. Linux `perf` Hardware Counters (`perf stat`)
- **IPC (Instructions Per Cycle)**: Increases from **1.10 IPC** (Naive) to **3.48 IPC** (AVX SIMD).
- **Cache Requests**: AVX SIMD reduces L3 cache memory bus requests from **1.517 Billion down to 166 Million** (an 89% reduction in cache requests).

### 2. Valgrind Cachegrind (`cg_annotate --show=Dr,D1mr,DLmr`)
- **L1 Data Cache Misses**: Loop reordering (`i-k-j`) drops L1 data read misses from **67.4 Million down to 4.2 Million** (N=256), reducing L1 miss rate from **50.21% to 3.14%**.

## Documentation
- **Full Empirical Benchmark Report**: [`ANALYSIS_REPORT.md`](file:///mnt/SSD_512_win/Linux/College-Backup/Projects-Aug/matrix-multiplication-optimization/ANALYSIS_REPORT.md)
- **Developer & Profiling Guide**: [`notes.md`](file:///mnt/SSD_512_win/Linux/College-Backup/Projects-Aug/matrix-multiplication-optimization/notes.md)
