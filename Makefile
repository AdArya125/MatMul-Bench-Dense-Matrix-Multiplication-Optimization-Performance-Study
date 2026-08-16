# =============================================================================
# Makefile — Matrix Multiplication Optimization Project
# =============================================================================
# Usage:
#   make              → build everything
#   make naive        → build only the naive binary
#   make run-naive    → build and run naive (N=512)
#   make benchmark    → run the unified benchmark
#   make cachegrind   → run Cachegrind analysis (N=256)
#   make perf         → run perf analysis (N=1024)
#   make clean        → remove all binaries
#
# N=512 make run-naive   → override matrix size

N       ?= 512
CXX     := g++
MPICXX  := mpicxx
NVCC    := nvcc
INCLUDE := include

CXXFLAGS     := -O2 -std=c++17 -I$(INCLUDE)
SIMD_FLAGS   := -mavx2 -mfma
OMP_FLAGS    := -fopenmp
CUDA_ARCH    := -arch=sm_75
DEBUG_FLAGS  := -O0 -g -std=c++17 -I$(INCLUDE)

BINDIR  := bin
SRCDIR  := src
BCHDIR  := benchmark

$(shell mkdir -p $(BINDIR))

# ─── CPU targets ──────────────────────────────────────────────────────────────

.PHONY: naive reordered blocked openmp avx mpi hybrid benchmark-bin

naive:
	$(CXX) $(CXXFLAGS) -o $(BINDIR)/naive $(SRCDIR)/naive.cpp
	@echo "Built: $(BINDIR)/naive"

reordered:
	$(CXX) $(CXXFLAGS) $(SIMD_FLAGS) -o $(BINDIR)/reordered $(SRCDIR)/reordered.cpp
	@echo "Built: $(BINDIR)/reordered"

blocked:
	$(CXX) $(CXXFLAGS) $(SIMD_FLAGS) -o $(BINDIR)/blocked $(SRCDIR)/blocked.cpp
	@echo "Built: $(BINDIR)/blocked"

openmp:
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) -o $(BINDIR)/openmp_matmul $(SRCDIR)/openmp.cpp
	@echo "Built: $(BINDIR)/openmp_matmul"

avx:
	$(CXX) $(CXXFLAGS) $(SIMD_FLAGS) -o $(BINDIR)/avx_matmul $(SRCDIR)/avx.cpp
	@echo "Built: $(BINDIR)/avx_matmul"

mpi:
	$(MPICXX) $(CXXFLAGS) $(SIMD_FLAGS) -o $(BINDIR)/mpi_matmul $(SRCDIR)/mpi.cpp
	@echo "Built: $(BINDIR)/mpi_matmul"

hybrid:
	$(MPICXX) $(CXXFLAGS) $(SIMD_FLAGS) $(OMP_FLAGS) \
	    -o $(BINDIR)/hybrid_matmul $(SRCDIR)/hybrid.cpp
	@echo "Built: $(BINDIR)/hybrid_matmul"

benchmark-bin:
	$(CXX) $(CXXFLAGS) $(SIMD_FLAGS) $(OMP_FLAGS) \
	    -o $(BINDIR)/benchmark $(BCHDIR)/benchmark.cpp
	@echo "Built: $(BINDIR)/benchmark"

# ─── CUDA ────────────────────────────────────────────────────────────────────

cuda:
	$(NVCC) -O2 $(CUDA_ARCH) -std=c++17 -I$(INCLUDE) \
	    $(SRCDIR)/cuda.cu -o $(BINDIR)/cuda_matmul
	@echo "Built: $(BINDIR)/cuda_matmul"

# ─── Build all ───────────────────────────────────────────────────────────────

all: naive reordered blocked openmp avx mpi hybrid benchmark-bin cuda
	@echo ""
	@echo "All targets built."

# ─── Run targets ─────────────────────────────────────────────────────────────

run-naive: naive
	./$(BINDIR)/naive $(N)

run-reordered: reordered
	./$(BINDIR)/reordered $(N)

run-blocked: blocked
	./$(BINDIR)/blocked $(N)

run-openmp: openmp
	./$(BINDIR)/openmp_matmul $(N)

run-avx: avx
	./$(BINDIR)/avx_matmul $(N)

run-mpi: mpi
	mpirun -np 4 --oversubscribe ./$(BINDIR)/mpi_matmul $(N)

run-cuda: cuda
	./$(BINDIR)/cuda_matmul $(N)

# ─── Analysis ────────────────────────────────────────────────────────────────

benchmark: benchmark-bin
	mkdir -p results/benchmarks
	./$(BINDIR)/benchmark | tee results/benchmarks/cpu_results.csv

cachegrind:
	bash scripts/cachegrind.sh $(N)

perf:
	bash scripts/perf.sh $(N)

# ─── Misc ────────────────────────────────────────────────────────────────────

clean:
	rm -f $(BINDIR)/*
	@echo "Cleaned bin/"

info:
	@echo "Compiler : $(shell $(CXX) --version | head -1)"
	@echo "MPI      : $(shell $(MPICXX) --version | head -1)"
	@echo "NVCC     : $(shell $(NVCC) --version | head -1)"
	@echo "CPU      : $(shell lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
	@echo "Cores    : $(shell nproc)"
	@echo "AVX2     : $(shell grep -o avx2 /proc/cpuinfo | head -1)"
