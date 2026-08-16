#!/usr/bin/env bash
# =============================================================================
# scripts/run_sweep.sh
#
# Compiles benchmark_sweep.cpp and runs the size-sweep benchmark.
# Logs both progress and CSV results to results/benchmarks/sweep_log.txt
# and results/benchmarks/sweep_results.csv so no data is lost if terminal disconnects.
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p results/benchmarks bin

LOG_FILE="results/benchmarks/sweep_log.txt"
CSV_FILE="results/benchmarks/sweep_results.csv"

echo "========================================================" | tee "$LOG_FILE"
echo "  MatMul-Bench: Matrix Size Sweep Benchmark (100 to 10k)" | tee -a "$LOG_FILE"
echo "  Start Time: $(date)" | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "► Compiling benchmark_sweep.cpp..." | tee -a "$LOG_FILE"

g++ -O2 -mavx2 -mfma -fopenmp -std=c++17 -I include \
    benchmark/benchmark_sweep.cpp -o bin/benchmark_sweep 2>&1 | tee -a "$LOG_FILE"

echo "  Compilation complete." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "► Running size sweep benchmark..." | tee -a "$LOG_FILE"
echo "  Results will be saved to: $CSV_FILE" | tee -a "$LOG_FILE"
echo "  Progress logged to:       $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run executable and capture CSV output directly to file while showing real-time lines
./bin/benchmark_sweep | tee "$CSV_FILE"

echo "" | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"
echo "  Sweep Complete! Output written to $CSV_FILE" | tee -a "$LOG_FILE"
echo "  End Time: $(date)" | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"
