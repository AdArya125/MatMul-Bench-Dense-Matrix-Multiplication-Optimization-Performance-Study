#!/usr/bin/env bash
# =============================================================================
# scripts/run_perf_sweep.sh
#
# Runs `perf stat -r 3` across a complete sweep of matrix sizes (100 to 10007)
# for ALL kernels without any cutoffs.
# Logs hardware counters (cycles, instructions, IPC, cache refs, cache misses)
# directly to results/benchmarks/perf_sweep_results.txt
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p results/benchmarks bin

LOG_FILE="results/benchmarks/perf_sweep_results.txt"

echo "========================================================" | tee "$LOG_FILE"
echo "  MatMul-Bench: Full Uninhibited perf Sweep (100 to 10007)" | tee -a "$LOG_FILE"
echo "  Start Time: $(date)" | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "► Compiling perf_sweep_runner.cpp..." | tee -a "$LOG_FILE"

g++ -O2 -mavx2 -mfma -fopenmp -std=c++17 -I include \
    benchmark/perf_sweep_runner.cpp -o bin/perf_sweep_runner 2>&1 | tee -a "$LOG_FILE"

echo "  Compilation complete." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

SIZES=(100 127 256 383 512 701 1024 1337 2048 3001 4096 6143 8192 10007)
EVENTS="cycles,instructions,cache-references,cache-misses"

for N in "${SIZES[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "==========================================================" | tee -a "$LOG_FILE"
    echo "  SWEEP SIZE N = $N" | tee -a "$LOG_FILE"
    echo "==========================================================" | tee -a "$LOG_FILE"

    # 1. Naive (run up to N=2048 to prevent multi-hour hangs on 10k naive)
    if [ "$N" -le 2048 ]; then
        echo "" | tee -a "$LOG_FILE"
        echo "► Kernel: naive (N=$N)" | tee -a "$LOG_FILE"
        perf stat -e "$EVENTS" -r 3 -- ./bin/perf_sweep_runner naive "$N" 2>&1 | tee -a "$LOG_FILE"
    fi

    # 2. Reordered i-k-j (ALL SIZES)
    echo "" | tee -a "$LOG_FILE"
    echo "► Kernel: reordered_ikj (N=$N)" | tee -a "$LOG_FILE"
    perf stat -e "$EVENTS" -r 3 -- ./bin/perf_sweep_runner reordered_ikj "$N" 2>&1 | tee -a "$LOG_FILE"

    # 3. Blocked-64 (ALL SIZES)
    echo "" | tee -a "$LOG_FILE"
    echo "► Kernel: blocked_64 (N=$N)" | tee -a "$LOG_FILE"
    perf stat -e "$EVENTS" -r 3 -- ./bin/perf_sweep_runner blocked_64 "$N" 2>&1 | tee -a "$LOG_FILE"

    # 4. OpenMP 4c (ALL SIZES)
    echo "" | tee -a "$LOG_FILE"
    echo "► Kernel: openmp_4c (N=$N)" | tee -a "$LOG_FILE"
    perf stat -e "$EVENTS" -r 3 -- ./bin/perf_sweep_runner openmp_4c "$N" 2>&1 | tee -a "$LOG_FILE"

    # 5. AVX Blocked (ALL SIZES)
    echo "" | tee -a "$LOG_FILE"
    echo "► Kernel: avx_blocked (N=$N)" | tee -a "$LOG_FILE"
    perf stat -e "$EVENTS" -r 3 -- ./bin/perf_sweep_runner avx_blocked "$N" 2>&1 | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"
echo "  perf Full Sweep Complete! Results saved to $LOG_FILE" | tee -a "$LOG_FILE"
echo "  End Time: $(date)" | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"
