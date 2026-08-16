/**
 * benchmark_sweep.cpp  —  Matrix size sweep benchmark.
 *
 * Evaluates performance across a diverse range of matrix sizes (100 to 10000),
 * including power-of-two, non-divisible, odd, and prime dimensions to capture
 * memory alignment, cache line boundary, and padding effects.
 *
 * Build:
 *   g++ -O2 -mavx2 -mfma -fopenmp -std=c++17 -I../include benchmark_sweep.cpp -o benchmark_sweep
 *
 * Run:
 *   ./benchmark_sweep > ../results/benchmarks/sweep_results.csv
 */

#include <immintrin.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <functional>
#include <string>
#include <algorithm>
#include "matrix_utils.hpp"

// ─── Kernel Implementations ──────────────────────────────────────────────────

// 1. Baseline i-j-k
void matmul_naive(Matrix C, const Matrix A, const Matrix B, int N) {
    zero_matrix(C, N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            scalar_t sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

// 2. Reordered i-k-j
void matmul_ikj(Matrix C, const Matrix A, const Matrix B, int N) {
    zero_matrix(C, N);
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            scalar_t a = A[i * N + k];
            for (int j = 0; j < N; ++j)
                C[i * N + j] += a * B[k * N + j];
        }
    }
}

// 3. Cache Blocked (block=64)
void matmul_blocked64(Matrix C, const Matrix A, const Matrix B, int N) {
    const int BLK = 64;
    zero_matrix(C, N);
    for (int ii = 0; ii < N; ii += BLK) {
        for (int kk = 0; kk < N; kk += BLK) {
            for (int jj = 0; jj < N; jj += BLK) {
                int ie = std::min(ii + BLK, N);
                int ke = std::min(kk + BLK, N);
                int je = std::min(jj + BLK, N);
                for (int i = ii; i < ie; ++i) {
                    for (int k = kk; k < ke; ++k) {
                        scalar_t a = A[i * N + k];
                        for (int j = jj; j < je; ++j)
                            C[i * N + j] += a * B[k * N + j];
                    }
                }
            }
        }
    }
}

// 4. OpenMP (static, max physical threads)
void matmul_omp(Matrix C, const Matrix A, const Matrix B, int N) {
    zero_matrix(C, N);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            scalar_t a = A[i * N + k];
            for (int j = 0; j < N; ++j)
                C[i * N + j] += a * B[k * N + j];
        }
    }
}

// 5. AVX2 SIMD + Blocking (handles non-divisible N via scalar tail)
void matmul_avx_blk(Matrix C, const Matrix A, const Matrix B, int N) {
    const int BLK = 64;
    zero_matrix(C, N);
    for (int ii = 0; ii < N; ii += BLK) {
        for (int kk = 0; kk < N; kk += BLK) {
            for (int jj = 0; jj < N; jj += BLK) {
                int ie = std::min(ii + BLK, N);
                int ke = std::min(kk + BLK, N);
                int je = std::min(jj + BLK, N);
                for (int i = ii; i < ie; ++i) {
                    for (int k = kk; k < ke; ++k) {
                        __m256 aik = _mm256_set1_ps(A[i * N + k]);
                        int j = jj;
                        for (; j + 8 <= je; j += 8) {
                            __m256 c = _mm256_loadu_ps(&C[i * N + j]);
                            __m256 b = _mm256_loadu_ps(&B[k * N + j]);
                            _mm256_storeu_ps(&C[i * N + j], _mm256_fmadd_ps(aik, b, c));
                        }
                        scalar_t as = A[i * N + k];
                        for (; j < je; ++j)
                            C[i * N + j] += as * B[k * N + j];
                    }
                }
            }
        }
    }
}

// ─── Benchmark Runner ────────────────────────────────────────────────────────

struct Kernel {
    std::string name;
    std::function<void(Matrix, const Matrix, const Matrix, int)> fn;
    int max_safe_N; // Cutoff size to prevent extreme runtimes (e.g. naive at N=10000 takes hours)
};

double bench_one(const Kernel& k, int N, int reps = 2) {
    Matrix A = alloc_matrix(N);
    Matrix B = alloc_matrix(N);
    Matrix C = alloc_matrix(N);

    fill_random(A, N, 42);
    fill_random(B, N, 99);

    // Warm-up run
    k.fn(C, A, B, N);

    double best = 1e18;
    for (int r = 0; r < reps; ++r) {
        Timer t;
        t.start();
        k.fn(C, A, B, N);
        best = std::min(best, t.elapsed_ms());
    }

    free_matrix(A);
    free_matrix(B);
    free_matrix(C);
    return best;
}

int main() {
    // Force OpenMP to use 4 physical cores to avoid HT noise
    omp_set_num_threads(4);

    // Diverse test sizes: small, medium, large; clean, odd, prime, non-SIMD divisible
    std::vector<int> sizes = {
        100,    // Small round non-power of 2
        127,    // Prime size (odd, misaligned SIMD tail)
        256,    // Clean power of 2
        383,    // Prime size
        512,    // Power of 2
        701,    // Prime size
        1024,   // Power of 2 (L3 cache boundary)
        1337,   // Odd non-power of 2
        2048,   // Large power of 2 (exceeds L3 cache)
        3001,   // Large prime size
        4096,   // Huge power of 2
        6143,   // Odd large size
        8192,   // Massive power of 2
        10007   // Massive prime size (~10k)
    };

    std::vector<Kernel> kernels = {
        {"naive",        matmul_naive,    1024},  // Cutoff at 1024 (O(N^3) naive at N=2048 takes ~20s per run)
        {"reordered_ikj",matmul_ikj,      4096},  // Cutoff at 4096
        {"blocked_64",   matmul_blocked64,4096},  // Cutoff at 4096
        {"openmp_4c",    matmul_omp,      8192},  // Cutoff at 8192
        {"avx_blocked",  matmul_avx_blk, 10007}   // Runs all up to 10007
    };

    std::cout << "kernel,N,time_ms,gflops\n";
    std::cout.flush();

    for (int N : sizes) {
        for (const auto& k : kernels) {
            if (N > k.max_safe_N) continue;

            double ms = bench_one(k, N);
            double gf = compute_gflops(N, ms);

            std::cout << k.name << "," << N << ","
                      << std::fixed << std::setprecision(3) << ms << ","
                      << std::fixed << std::setprecision(3) << gf << "\n";
            std::cout.flush(); // Ensure output is immediately written to log file
        }
    }

    return 0;
}
