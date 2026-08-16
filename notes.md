# Matrix Multiplication Optimization & Engineering Notes

## Source Code & Optimization Reference

### 1. `include/matrix_utils.hpp` - Foundation & Infrastructure
- **Row-Major Flattened Storage**: 2D matrices are stored in flat 1D memory (`data[i * N + j]`). Avoids double-pointer indirection (`float**`) and fragmented memory allocations.
- **64-Byte Aligned Allocation (`posix_memalign`)**:
  - `posix_memalign(&ptr, 64, bytes)` ensures matrix memory starts at exact 64-byte CPU cache line boundaries.
  - Required for AVX 256-bit vector load/store instructions (`_mm256_load_ps`) to prevent unaligned memory penalties.
- **Relative Verification (`matrices_close`)**: Floating-point additions are non-associative. Relative tolerance scaling (`diff > tol * mag`) verifies correctness across different execution orders.
- **GFLOP/s Formula**: For N x N dense matrix multiplication (2*N^3 operations):
  Throughput = (2 * N^3) / (Time_in_sec * 10^9)

### 2. `src/naive.cpp` - Baseline `i-j-k` Implementation
- **Loop Order**: `i` (rows) -> `j` (cols) -> `k` (reduction).
- **Core Cause of Poor Performance**: As k increments, B[k][j] jumps by N floats (4N bytes). At N=1024, every step jumps 4096 bytes, missing the 64-byte cache line and forcing continuous main memory DRAM fetches.

### 3. `src/reordered.cpp` - Loop Permutations (`i-k-j`)
- **Key Insight**: Changing loop order to `i-k-j` keeps A[i][k] fixed in a CPU register while sweeping row k of B and row i of C sequentially.
- **Spatial Locality**: Inner `j` loop accesses contiguous memory addresses, enabling CPU hardware prefetchers and multi-word cache line hits.
- **Result**: **~4x speedup** over naive without modifying arithmetic operations (2*N^3 FLOPs).

### 4. `src/blocked.cpp` - Cache Blocking (Tiling)
- **Key Insight**: Divides N x N matrices into small B x B tiles (64x64 or 256x256).
- **Temporal Locality**: Ensures sub-tiles fit inside fast L1/L2/L3 cache memory before eviction.
- **Trade-off Rule**:
  - At N=1024 (4 MB), matrix B fits in 8 MB L3 cache; unblocked `i-k-j` has lower loop counter overhead.
  - At N=2048 (16 MB), matrix B exceeds L3 cache; **blocking provides a ~1.88x speedup** over unblocked AVX.

### 5. `src/openmp.cpp` - Multithreaded Parallelism
- **Core Directives**:
  - `#pragma omp parallel for schedule(static)`: Spawns threads and assigns row chunks of A and C to separate cores.
  - `collapse(2)`: Flattens outer `i` and `j` loops.
- **Key Learnings**:
  - Peak efficiency achieved at **physical core count (4 cores)**.
  - Hyper-Threading (8 threads) degrades performance due to shared execution pipeline & L1/L2 cache resource contention.

### 6. `src/avx.cpp` - AVX2 SIMD Vectorization
- **Header**: `<immintrin.h>`
- **Intrinsics Used**:
  - `__m256`: 256-bit register holding **8 single-precision floats** (32 bits each).
  - `_mm256_set1_ps(val)`: Broadcasts 1 scalar float across all 8 lanes of a SIMD register.
  - `_mm256_loadu_ps(addr)`: Loads 8 contiguous floats from memory into a vector register.
  - `_mm256_fmadd_ps(a, b, c)`: **Fused Multiply-Add** (a * b + c) executing 8 multiplies and 8 additions in **1 CPU clock cycle**.
  - `_mm256_storeu_ps(addr, vec)`: Writes 8 floats from register back to memory.
- **Mathematical Shift**: Changes mental model from "Dot Product of Row x Column" to "Linear Combination of Rows (C[i] += A[i][k] * B[k])".

### 7. `src/cuda.cu` - GPU Acceleration (NVIDIA CUDA)
- **Architecture**: 896 CUDA cores on NVIDIA GTX 1650.
- **Shared Memory Tiling (`__shared__`)**: Loads 32x32 sub-tiles into fast on-chip Shared Memory SRAM (1-2 cycles latency), reusing each element 32 times and bypassing 200+ cycle Global VRAM DRAM latencies.
- **Key Primitives**: `__global__`, `dim3 grid/block`, `__syncthreads()`, `cudaMemcpy(HostToDevice/DeviceToHost)`.

### 8. `src/mpi.cpp` - Distributed Memory Parallelism
- **Communication Primitives**:
  - `MPI_Init()`, `MPI_Comm_rank()`, `MPI_Comm_size()`, `MPI_Finalize()`.
  - `MPI_Scatter`: Partitions matrix A into row-slices across processes.
  - `MPI_Bcast`: Replicates matrix B across processes.
  - `MPI_Gather`: Reassembles final matrix C at Rank 0.

### 9. `src/hybrid.cpp` - Hybrid MPI + OpenMP
- **Concept**: 1 MPI rank per cluster node -> OpenMP multithreading across node cores.
- **Thread Safety**: `MPI_Init_thread(..., MPI_THREAD_FUNNELED, ...)` ensures only the master thread makes MPI network calls.

## Profiling Tools Reference

### 1. Linux `perf` (Hardware Performance Counters)

`perf` reads hardware performance counters built directly inside the CPU with near-zero overhead (~1-2%).

#### Command 1: `perf stat` (Summary Counter Metrics)
```bash
perf stat -e cycles,instructions,cache-references,cache-misses -r 3 -- <EXECUTABLE> <ARGS>
```
- **Derived Metrics**:
  - IPC = instructions / cycles (High IPC > 2.0 = efficient execution; Low IPC < 1.0 = stalling).
  - Cache Miss Rate = (cache-misses / cache-references) * 100%.

#### Environment Variables with `perf`:
```bash
OMP_NUM_THREADS=2 perf stat -e cycles,instructions,cache-references,cache-misses -r 3 -- mpirun -np 4 --oversubscribe ./bin/hybrid_matmul 1024
```

#### Command 2: `perf record` & `perf report` (CPU Sampling Profiler)
```bash
perf record -g -- <EXECUTABLE> <ARGS>
perf report
```

#### Command 3: `perf annotate` (Assembly Line-by-Line Overhead)
- **`vmovss` / `mov`**: High % indicates memory latency stalls.
- **`vfmadd` / `fmul`**: High % indicates compute arithmetic execution.
- **`ss`**: Scalar Single precision (1 element).
- **`ps`**: Packed Single precision (8 SIMD elements).

### 2. Valgrind Cachegrind (Software Cache Simulation)

Cachegrind simulates a CPU cache in software and annotates exact source code line numbers with cache hit/miss counts. (Runs ~30x slower; use small problem sizes like N=256).

#### Workflow Steps:
1. **Compile with Debug Info (`-g`)**:
   ```bash
   g++ -O2 -g -std=c++17 -I include src/naive.cpp -o bin/naive_cg
   ```
2. **Run Cachegrind Simulation**:
   ```bash
   valgrind --tool=cachegrind --cache-sim=yes --cachegrind-out-file=naive.cg ./bin/naive_cg 256
   ```
3. **View Annotated Source Report**:
   ```bash
   cg_annotate --show=Dr,D1mr,DLmr naive.cg
   ```

#### Column Definitions:
- **`Dr` (Data Reads)**: Total memory read instructions executed (`A[i][k]`, `B[k][j]`).
- **`D1mr` (L1 Data Read Misses)**: Reads that missed L1 Data Cache.
- **`DLmr` (LLC Data Read Misses)**: Reads that missed Last-Level (L3) Cache and accessed RAM.

### 3. NVIDIA `nvprof` & `ncu` (CUDA GPU Hardware Profiler)

NVIDIA GPU profiling tools intercept GPU kernels, CUDA API calls, and PCIe memory transfers.

#### Basic GPU Timing (`nvprof`):
```bash
nvprof ./bin/cuda_matmul 1024
```
- **`GPU activities`**:
  - `matmul_naive_kernel`: Kernel execution time on Global VRAM DRAM.
  - `matmul_tiled_kernel`: Kernel execution time on Shared Memory SRAM.
  - `[CUDA memcpy HtoD]` / `DtoH`: Host <-> Device PCIe bus transfer times.

#### Advanced GPU Hardware Counter Metrics (`ncu` - Nsight Compute):
```bash
sudo ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active --kernel-name matmul_tiled_kernel ./bin/cuda_matmul 1024
```
- **`sm__warps_active`**: GPU Occupancy % (ratio of active 32-thread warps per SM vs peak capacity).
- **`sm__throughput`**: Percentage of peak hardware SM compute throughput achieved.
