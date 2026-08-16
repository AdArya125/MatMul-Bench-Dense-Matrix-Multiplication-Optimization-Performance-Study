#!/usr/bin/env bash
# =============================================================================
# scripts/cachegrind.sh
#
# Runs Valgrind Cachegrind on the naive, reordered, and blocked implementations
# and generates annotated reports for comparison.
#
# What is Cachegrind?
# ────────────────────
# Cachegrind is a Valgrind tool that SIMULATES the CPU's cache hierarchy and
# counts every memory access categorised by cache level:
#
#   Ir  : total instruction references (= instructions executed)
#   I1mr: L1 instruction cache misses
#   ILmr: Last-level cache (LLC) instruction misses
#
#   Dr  : total data reads
#   D1mr: L1 data cache read misses
#   DLmr: LLC data read misses
#
#   Dw  : total data writes
#   D1mw: L1 data cache write misses
#   DLmw: LLC data write misses
#
# Key metrics:
#   L1 miss rate (data read) = D1mr / Dr   (should be small, e.g. <5%)
#   LLC miss rate (data)     = DLmr / Dr   (very expensive – goes to RAM)
#
# Important: Cachegrind is a SIMULATOR, not a hardware counter. It uses a
# simple LRU replacement model and fixed cache parameters. Actual CPU cache
# behavior can differ, but the relative ordering of algorithms is accurate.
#
# Usage:
#   cd matrix-multiplication-optimization
#   bash scripts/cachegrind.sh [N]   (default N=256 for manageable runtime)
#
# Cachegrind makes code run ~20–50× slower, so keep N small (256–512).
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

N="${1:-256}"
OUTDIR="results/cachegrind"
mkdir -p "$OUTDIR"

# Cache parameters matching our i5-9300H:
#   L1d: 32 KB, 8-way, 64-byte cache lines
#   L2 : 256 KB (per core), 4-way, 64-byte lines
#   L3 : 8 MB (shared), 16-way, 64-byte lines
#
# Cachegrind flags:
#   --tool=cachegrind            use the cachegrind tool
#   --I1=size,assoc,line_size    L1 instruction cache parameters
#   --D1=size,assoc,line_size    L1 data cache parameters
#   --LL=size,assoc,line_size    last-level cache parameters
#   --cachegrind-out-file=FILE   write raw output to FILE
#   --branch-sim=no              skip branch prediction simulation (faster)

CACHEGRIND_FLAGS="
  --tool=cachegrind
  --branch-sim=no
  --D1=32768,8,64
  --LL=8388608,16,64
"

echo "========================================================"
echo "  Cachegrind Cache Analysis  N=$N"
echo "  $(date)"
echo "========================================================"

run_cg() {
    local binary="$1"
    local label="$2"
    local outfile="$OUTDIR/${label}_N${N}.cg"

    echo ""
    echo "► Running: $label (N=$N)"
    echo "  Output → $outfile"

    valgrind $CACHEGRIND_FLAGS \
        --cachegrind-out-file="$outfile" \
        "$binary" "$N" 2>&1 | tail -20

    echo ""
    echo "  ── cg_annotate summary for $label:"
    # cg_annotate parses the raw .cg file and produces human-readable output
    # --auto=yes: automatically annotate source files found in the binary
    cg_annotate "$outfile" --auto=yes 2>/dev/null | head -80 \
        | tee "$OUTDIR/${label}_N${N}_summary.txt"
}

# Build the binaries if not present
if [ ! -f bin/naive ]; then
    echo "Building binaries first..."
    g++ -O0 -g -std=c++17 -I include -o bin/naive_cg     src/naive.cpp
    g++ -O0 -g -std=c++17 -I include -o bin/reordered_cg src/reordered.cpp
    g++ -O0 -g -std=c++17 -I include -o bin/blocked_cg   src/blocked.cpp
else
    # Rebuild with debug symbols for annotation
    g++ -O0 -g -std=c++17 -I include -o bin/naive_cg     src/naive.cpp
    g++ -O0 -g -std=c++17 -I include -o bin/reordered_cg src/reordered.cpp
    g++ -O0 -g -std=c++17 -I include -o bin/blocked_cg   src/blocked.cpp
fi

# ── NOTE: We use -O0 (no optimisation) for Cachegrind so the loop
#    structure is preserved and the source annotation is meaningful.
#    With -O2, the compiler can reorder/fuse loops, making the
#    annotation hard to interpret. The cache behavior at -O0 still
#    demonstrates the algorithmic difference between loop orders.

run_cg ./bin/naive_cg     "naive"
run_cg ./bin/reordered_cg "reordered"   # will run ALL three variants
run_cg ./bin/blocked_cg   "blocked"

echo ""
echo "========================================================"
echo "  Cachegrind analysis complete."
echo "  Summary files in: $OUTDIR/"
echo ""
echo "  To compare results side by side:"
echo "    cg_annotate $OUTDIR/naive_N${N}.cg"
echo "    cg_annotate $OUTDIR/reordered_N${N}.cg"
echo "    cg_annotate $OUTDIR/blocked_N${N}.cg"
echo ""
echo "  Key metric to compare: DLmr (LLC data read misses)"
echo "  Lower = fewer main-memory accesses = better cache locality"
echo "========================================================"
