#!/usr/bin/env bash
# =============================================================================
# scripts/perf.sh
#
# Uses Linux `perf` for hardware performance-counter analysis.
#
# What is perf?
# ─────────────
# `perf` is a Linux kernel profiling tool that reads CPU hardware performance
# counters (PMU – Performance Monitoring Unit). Unlike Cachegrind (simulation),
# perf reads ACTUAL hardware events with very low overhead (~1–5%).
#
# Key perf subcommands:
#
#   perf stat ./program
#     → Runs program and prints summary of hardware counters at the end.
#     → Most useful for quick sanity checks.
#
#   perf record -g ./program
#     → Samples program execution at regular intervals (default: ~1000 Hz).
#     → Records call-graph (-g) for flamegraph generation.
#
#   perf report
#     → Interactive TUI to browse the recorded profile.
#
#   perf annotate
#     → Shows assembly/source mixed with per-instruction sample counts.
#
# Hardware events (for `perf stat -e`):
#
#   cycles           : CPU clock cycles consumed
#   instructions     : Instructions retired (completed)
#   cache-references : Last-level cache accesses
#   cache-misses     : Last-level cache misses (goes to RAM)
#   branch-misses    : Branch predictor misses (pipeline flush)
#   fp_arith_inst_retired.256b_packed_single : AVX 256-bit FP operations (Intel)
#
# IPC = instructions / cycles
#   IPC > 2 : CPU is well-utilized (superscalar execution)
#   IPC < 1 : Memory-bound or branch-misprediction stalls
#
# LLC miss rate = cache-misses / cache-references
#   < 5%    : Good cache behavior
#   > 30%   : Severe memory pressure
#
# Usage:
#   cd matrix-multiplication-optimization
#   bash scripts/perf.sh [N]   (default N=1024)
#
# Requirements:
#   perf_event_paranoid should be ≤ 1 for hardware counters:
#     sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

N="${1:-1024}"
OUTDIR="results/perf"
mkdir -p "$OUTDIR"

# ── Build with optimisations but debug symbols (for perf annotate)
echo "Building with -O2 -g..."
g++ -O2 -g -std=c++17 -I include -o bin/naive_perf     src/naive.cpp
g++ -O2 -g -std=c++17 -mavx2 -mfma -I include -o bin/ikj_perf src/reordered.cpp
g++ -O2 -g -std=c++17 -mavx2 -mfma -I include -o bin/avx_perf src/avx.cpp

# Events to measure
EVENTS="cycles,instructions,cache-references,cache-misses,branch-misses"

echo ""
echo "========================================================"
echo "  Linux perf Performance Counter Analysis  N=$N"
echo "  $(date)"
echo "========================================================"

perf_stat() {
    local binary="$1"
    local label="$2"
    local outfile="$OUTDIR/${label}_N${N}_perf.txt"

    echo ""
    echo "► $label (N=$N)"
    echo "----------------------------------------------------------"

    # perf stat flags:
    #   -e events    : comma-separated list of events to measure
    #   -r 3         : repeat 3 times and average
    #   --           : separator between perf flags and the program
    perf stat -e "$EVENTS" -r 3 -- "$binary" "$N" 2>&1 \
        | tee "$outfile"

    echo ""
    echo "  Output saved to: $outfile"
}

# Check paranoid level
PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "?")
if [ "$PARANOID" -gt 1 ] 2>/dev/null; then
    echo ""
    echo "WARNING: perf_event_paranoid=$PARANOID"
    echo "  Hardware counters may not work. Run:"
    echo "    sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'"
    echo "  Falling back to software events only."
    EVENTS="task-clock,context-switches,page-faults,instructions,cycles"
fi

perf_stat ./bin/naive_perf   "naive_ijk"
perf_stat ./bin/ikj_perf     "reordered_ikj"
perf_stat ./bin/avx_perf     "avx_simd"

# ── Optional: record flamegraph data for one run
echo ""
echo "► Recording perf profile for avx_simd (for flamegraph)..."
PERF_DATA="$OUTDIR/avx_N${N}.perf.data"
perf record -g -o "$PERF_DATA" -- ./bin/avx_perf "$N" 2>&1

echo ""
echo "  Profile written to: $PERF_DATA"
echo "  View with: perf report -i $PERF_DATA"
echo "  Or generate flamegraph:"
echo "    perf script -i $PERF_DATA | stackcollapse-perf.pl | flamegraph.pl > flame.svg"

echo ""
echo "========================================================"
echo "  perf analysis complete. Results in $OUTDIR/"
echo ""
echo "  Metrics to compare across implementations:"
echo "    IPC = instructions / cycles    (higher = better utilization)"
echo "    LLC miss rate = cache-misses / cache-references  (lower = better)"
echo "========================================================"
