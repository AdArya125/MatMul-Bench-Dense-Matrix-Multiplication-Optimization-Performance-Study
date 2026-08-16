# Matrix Multiplication Optimization & Performance Study

**System Hardware Specs:**
- **CPU:** Intel Core i5-9300H @ 2.40GHz (4 physical cores, 8 logical threads)
- **CPU Cache:** L1d: 128 KB, L2: 1 MB, L3: 8 MB (shared)
- **RAM:** DDR4
- **GPU:** NVIDIA GeForce GTX 1650 (4 GB VRAM, 16 SMs / 896 CUDA cores)

## Master Performance Comparison Table (N = 1024)

| Checkpoint | Implementation | Execution Time | Throughput | Speedup vs Naive Baseline | Key Bottleneck / Optimization |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CP 1** | **Naive (`i-j-k`)** | 2203.24 ms | 0.97 GFLOP/s | 1.00x | Stride-N memory access on B causes severe L1/L2 cache misses. |
| **CP 2** | **Loop Reordering (`i-k-j`)** | 543.00 ms | 3.96 GFLOP/s | **4.05x** | Spatial locality: inner `j` loop accesses B and C sequentially. |
| **CP 2** | *Loop Reordering (`j-k-i`)* | 10300.00 ms | 0.21 GFLOP/s | *0.21x (Slowdown)* | Stride-N accesses on both A and C. |
| **CP 3** | **Cache Blocked (`256x256`)** | 674.50 ms | 3.18 GFLOP/s | **3.27x** | Keeps working sub-tiles in L2/L3 cache. |
| **CP 4** | **OpenMP (4 physical cores)** | 232.90 ms | 9.22 GFLOP/s | **9.46x** | Multi-threaded parallelism on outer `i` loop. |
| **CP 4** | *OpenMP (8 hyper-threads)* | 295.78 ms | 7.26 GFLOP/s | *7.45x* | HT thrashing & shared L3 memory bus saturation. |
| **CP 5** | **AVX2 SIMD (`i-k-j`)** | 114.32 ms | 18.79 GFLOP/s | **19.27x** | 256-bit SIMD registers + FMA3 (8 floats processed per cycle). |
| **CP 6** | **CUDA Naive (Global Mem)** | 7.49 ms | 286.85 GFLOP/s | **294.16x** | 896 CUDA cores parallel execution. |
| **CP 6** | **CUDA Tiled (Shared Mem)** | **5.84 ms** | **367.92 GFLOP/s** | **377.27x** | On-chip Shared Memory SRAM tiling (32x32). |
| **CP 7** | **MPI (4 processes)** | 354.08 ms | 6.20 GFLOP/s | **6.22x** | Row-sliced distributed memory distribution. |
| **CP 8** | **Hybrid MPI + OpenMP (4p x 2t)** | 340.95 ms | 6.30 GFLOP/s | **6.46x** | Combined inter-node MPI messaging + intra-node OpenMP threads. |

## Detailed Checkpoint Breakdown & Analysis

### Checkpoint 1: Naive `i-j-k` Baseline
- **Formula:** C[i][j] = sum_k (A[i][k] * B[k][j])
- **Issue:** As k increments, B[k][j] jumps by N floats (4*N bytes). For N=1024, every step jumps 4096 bytes, blowing past the 64-byte cache line boundary and forcing main memory DRAM fetches on every single operation.

### Checkpoint 2: Loop Reordering (`i-k-j`)
- **Key Insight:** Simply changing loop nesting from `i-j-k` to `i-k-j` keeps A[i][k] fixed in a CPU register while sweeping row k of B and row i of C sequentially.
- **Impact:** **4.05x speedup** over naive without altering arithmetic operations (2*N^3 FLOPs). Demonstrates that memory layout interaction dominates algorithm performance.

### Checkpoint 3: Cache Blocking (Tiling)
- **Key Insight:** Divides N x N matrices into sub-blocks (B x B) to ensure sub-tiles fit inside fast L1/L2 cache memory.
- **Trade-off Analysis (N=1024 vs N=2048):**
  - At N=1024 (4 MB matrix), B fits in the 8 MB L3 cache, so unblocked `avx i-k-j` (114 ms) beats `avx+blocked` (142 ms) due to lower loop overhead.
  - At N=2048 (16 MB matrix), B exceeds the 8 MB L3 cache. Unblocked AVX thrashes (2831 ms / 6.07 GFLOP/s), whereas **`avx+blocked` achieves 1506 ms / 11.41 GFLOP/s (1.88x faster than unblocked AVX)**.

### Checkpoint 4: OpenMP Multithreading
- **Key Insight:** Distributes outer `i` loop iterations across CPU threads.
- **Findings:**
  - Optimal performance achieved at **4 physical cores** (232.9 ms / 9.22 GFLOP/s).
  - Hyper-Threading (8 threads) causes performance degradation (295.7 ms) due to pipeline and L1/L2 cache resource contention.
  - `collapse(2)` with `i-j-k` order suffers a severe slowdown (1160 ms) by re-introducing stride-N cache miss patterns.

### Checkpoint 5: AVX / AVX2 SIMD Vectorization
- **Key Insight:** Replaces scalar operations with 256-bit `__m256` vector registers and Fused Multiply-Add (`_mm256_fmadd_ps`), executing 8 float multiplies and 8 additions in a single CPU cycle.
- **Impact:** **5.68x speedup over scalar `i-k-j`** and **19.27x speedup over naive baseline**.

### Checkpoint 6: CUDA GPU Acceleration
- **Key Insight:** Offloads matrix computation to 896 CUDA cores on NVIDIA GTX 1650.
- **Shared Memory Tiling:** `gpu-tiled` loads 32x32 sub-blocks into fast on-chip Shared Memory SRAM (1-2 cycles latency), reusing each loaded element 32 times and reducing global VRAM bandwidth stalls.
- **Impact:** **19.5x speedup over best CPU code (AVX)** and **377.27x speedup over naive C++**.
- **PCIe Transfer Cost (N=512 vs N=4096):**
  - At N=512, transfers (2.06 ms) take 2.7x longer than computation (0.75 ms).
  - At N=4096, computation (341.2 ms) dominates transfers (28.4 ms), proving GPUs excel on large compute-dense workloads.

### Checkpoint 7: MPI Distributed Parallelism
- **Key Insight:** Uses `MPI_Scatter` to partition matrix A into row slices and `MPI_Bcast` to replicate matrix B across processes.
- **Communication Cost:** `MPI_Bcast` of B requires 4 MB of data per process. At 8 processes on a single node, memory replication increases L3 cache pressure and IPC overhead (7.49 ms Bcast latency).
- **OpenMP vs. MPI Tradeoff:** OpenMP is faster on a single shared-memory machine (232.9 ms vs 354.08 ms) because threads share a single copy of B, whereas MPI is designed for scaling across separate physical cluster nodes.

### Checkpoint 8: Hybrid MPI + OpenMP
- **Key Insight:** Combines MPI_THREAD_FUNNELED inter-node communication with intra-node OpenMP multithreading.
- **HPC Value:** Reduces memory replication of B on multi-core cluster nodes (e.g. 1 Bcast per node instead of 1 Bcast per core), solving the memory footprint explosion of pure MPI on modern high-core-count clusters.

## Profiling Verification & Hardware Empirical Data

### 1. Full Linux `perf` Hardware Counter Profile (N=1024)

`perf stat` readings on the physical CPU Performance Monitoring Unit (PMU):

| Implementation | CPU Cycles | Instructions | IPC (Instructions/Cycle) | Cache References | Cache Misses | Total Elapsed Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`naive`** | 30.18 Billion | 30.30 Billion | **1.00** | 7.49 Billion | 374 Million (5.00%) | 9.879 s |
| **`reordered`** | 179.36 Billion *| 99.55 Billion *| **0.56** | 29.06 Billion | 2.71 Billion (9.35%)| 60.734 s |
| **`blocked`** | 44.28 Billion | 142.87 Billion | **3.23** | 3.27 Billion | 170 Million (5.20%) | 15.746 s |
| **`openmp_matmul`** | 140.51 Billion | 203.07 Billion | **1.45** | 12.20 Billion | 1.51 Billion (12.44%)| 20.677 s |
| **`avx_matmul`** | **15.82 Billion** | **45.56 Billion** | **2.88** | **1.66 Billion** | **399 Million (24.07%)**| **5.523 s** |
| **`mpi_matmul` (4p)** | **7.10 Billion** | **11.62 Billion** | **1.64** | **0.209 Billion** | **117 Million (55.92%)**| **1.478 s** |
| **`hybrid_matmul` (4p x 2t)**| **7.90 Billion** | **11.08 Billion** | **1.40** | **0.203 Billion** | **109 Million (53.48%)**| **1.261 s** |

- **IPC Analysis**: AVX SIMD and Cache Blocking achieve high execution density (**2.88 to 3.23 IPC**), keeping CPU pipelines active.
- **Cache Traffic Reduction**: AVX SIMD reduces L3 cache memory bus requests from **7.49 Billion down to 1.66 Billion** (an 78% reduction in memory requests) because 32-byte vector loads touch the memory subsystem far less frequently.

### 2. Valgrind Cachegrind Software Cache Simulation (N=256)

`cg_annotate --show=Dr,D1mr,DLmr` simulated cache event results:

| Function | `Dr` (Data Reads) | `D1mr` (L1 Data Read Misses) | **L1 Miss Rate** (D1mr/Dr) | `DLmr` (L3 Misses) |
| :--- | :--- | :--- | :--- | :--- |
| **`matmul_naive` (`i-j-k`)** | 134,217,748 | **67,399,688** | **50.21% Miss Rate!** | 0 |
| **`matmul_ikj` (`i-k-j`)** | 134,479,892 | **4,227,080** | **3.14% Miss Rate!** | 0 |

- **Empirical Cause of Speedup**: Loop reordering reduces L1 data cache misses from **67.4 Million down to 4.2 Million** (a **16x reduction in L1 cache misses**), dropping the L1 miss rate from **50.2% to 3.1%**.

### 3. NVIDIA `nvprof` GPU Profiling (N=1024)

`nvprof ./bin/cuda_matmul 1024` activity breakdown:

| GPU Activity / API Call | Total Time | Calls | Avg Time / Call | % of GPU Activity Time |
| :--- | :--- | :--- | :--- | :--- |
| **`matmul_naive_kernel` (Global Mem)** | 31.21 ms | 4 | 7.80 ms | 53.35% |
| **`matmul_tiled_kernel` (Shared Mem)**| **25.24 ms** | **4** | **6.31 ms** | **43.15%** |
| **`[CUDA memcpy HtoD]` (PCIe Input)** | 1.39 ms | 2 | 0.69 ms | 2.37% |
| **`[CUDA memcpy DtoH]` (PCIe Output)**| 0.63 ms | 1 | 0.63 ms | 1.07% |

- **Shared Memory SRAM Tiling Impact**: `matmul_tiled_kernel` saves **6.0 ms of GPU DRAM memory stall time** by caching sub-blocks inside 1-2 cycle Shared Memory SRAM.
- **PCIe Transfer Overhead**: Host-to-Device and Device-to-Host memory copies take **2.01 ms total**, representing only ~25% of GPU execution time at N=1024 (and less than 5% at N=4096).
