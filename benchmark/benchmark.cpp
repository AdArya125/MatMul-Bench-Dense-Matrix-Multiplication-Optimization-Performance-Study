// benchmark.cpp - Unified benchmark runner.
// Build: g++ -O2 -mavx2 -mfma -fopenmp -std=c++17 -I../include benchmark.cpp -o benchmark

#include <immintrin.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <functional>
#include <string>
#include <algorithm>
#include "matrix_utils.hpp"

void matmul_naive(Matrix C, const Matrix A, const Matrix B, int N) {
    zero_matrix(C, N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            scalar_t s = 0;
            for (int k = 0; k < N; ++k)
                s += A[i * N + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

void matmul_ikj(Matrix C, const Matrix A, const Matrix B, int N) {
    zero_matrix(C, N);
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k) {
            scalar_t a = A[i * N + k];
            for (int j = 0; j < N; ++j)
                C[i * N + j] += a * B[k * N + j];
        }
}

void matmul_blocked64(Matrix C, const Matrix A, const Matrix B, int N) {
    const int BLK = 64;
    zero_matrix(C, N);
    for (int ii = 0; ii < N; ii += BLK)
        for (int kk = 0; kk < N; kk += BLK)
            for (int jj = 0; jj < N; jj += BLK) {
                int ie = std::min(ii + BLK, N), ke = std::min(kk + BLK, N), je = std::min(jj + BLK, N);
                for (int i = ii; i < ie; ++i)
                    for (int k = kk; k < ke; ++k) {
                        scalar_t a = A[i * N + k];
                        for (int j = jj; j < je; ++j)
                            C[i * N + j] += a * B[k * N + j];
                    }
            }
}

void matmul_omp(Matrix C, const Matrix A, const Matrix B, int N) {
    zero_matrix(C, N);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k) {
            scalar_t a = A[i * N + k];
            for (int j = 0; j < N; ++j)
                C[i * N + j] += a * B[k * N + j];
        }
}

void matmul_avx_blk(Matrix C, const Matrix A, const Matrix B, int N) {
    const int BLK = 64;
    zero_matrix(C, N);
    for (int ii = 0; ii < N; ii += BLK)
        for (int kk = 0; kk < N; kk += BLK)
            for (int jj = 0; jj < N; jj += BLK) {
                int ie = std::min(ii + BLK, N), ke = std::min(kk + BLK, N), je = std::min(jj + BLK, N);
                for (int i = ii; i < ie; ++i)
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

struct Kernel {
    std::string name;
    std::function<void(Matrix, const Matrix, const Matrix, int)> fn;
};

double bench_one(const Kernel& k, int N, int reps = 3) {
    Matrix A = alloc_matrix(N), B = alloc_matrix(N), C = alloc_matrix(N);
    fill_random(A, N, 42);
    fill_random(B, N, 99);
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

int main(int argc, char* argv[]) {
    std::vector<int> sizes = {64, 128, 256, 512, 1024, 2048};
    if (argc > 1 && std::string(argv[1]) == "--quick")
        sizes = {64, 128, 256, 512};

    std::vector<Kernel> kernels = {
        {"naive", matmul_naive},
        {"ikj", matmul_ikj},
        {"blocked-64", matmul_blocked64},
        {"omp-static", matmul_omp},
        {"avx-blocked", matmul_avx_blk},
    };

    std::cout << "kernel,N,time_ms,gflops\n";

    for (int N : sizes) {
        for (const auto& k : kernels) {
            if (k.name == "naive" && N >= 2048) {
                std::cout << k.name << "," << N << ",N/A,N/A\n";
                continue;
            }
            double ms = bench_one(k, N);
            double gf = compute_gflops(N, ms);
            std::cout << k.name << "," << N << ","
                      << std::fixed << std::setprecision(3) << ms << ","
                      << std::fixed << std::setprecision(3) << gf << "\n";
            std::cout.flush();
        }
    }
    return 0;
}
