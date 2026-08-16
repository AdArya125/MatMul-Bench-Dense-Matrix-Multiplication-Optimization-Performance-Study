/**
 * perf_sweep_runner.cpp  —  Single-run driver for perf stat analysis.
 *
 * Runs a specific kernel at a specific size N once so perf stat can measure
 * hardware counters cleanly without averaging across multiple kernels.
 *
 * Usage:
 *   ./perf_sweep_runner <kernel_name> <N>
 */

#include <immintrin.h>
#include <omp.h>
#include <iostream>
#include <string>
#include <algorithm>
#include "matrix_utils.hpp"

void matmul_naive(Matrix C, const Matrix A, const Matrix B, int N) {
    zero_matrix(C, N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            scalar_t sum = 0.0f;
            for (int k = 0; k < N; ++k) sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

void matmul_ikj(Matrix C, const Matrix A, const Matrix B, int N) {
    zero_matrix(C, N);
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k) {
            scalar_t a = A[i * N + k];
            for (int j = 0; j < N; ++j) C[i * N + j] += a * B[k * N + j];
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
                        for (int j = jj; j < je; ++j) C[i * N + j] += a * B[k * N + j];
                    }
            }
}

void matmul_omp(Matrix C, const Matrix A, const Matrix B, int N) {
    zero_matrix(C, N);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k) {
            scalar_t a = A[i * N + k];
            for (int j = 0; j < N; ++j) C[i * N + j] += a * B[k * N + j];
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
                        for (; j < je; ++j) C[i * N + j] += as * B[k * N + j];
                    }
            }
}

int main(int argc, char* argv[]) {
    if (argc < 3) return 1;

    std::string kernel = argv[1];
    int N = std::stoi(argv[2]);

    omp_set_num_threads(4);

    Matrix A = alloc_matrix(N);
    Matrix B = alloc_matrix(N);
    Matrix C = alloc_matrix(N);

    fill_random(A, N, 42);
    fill_random(B, N, 99);

    if (kernel == "naive") matmul_naive(C, A, B, N);
    else if (kernel == "reordered_ikj") matmul_ikj(C, A, B, N);
    else if (kernel == "blocked_64") matmul_blocked64(C, A, B, N);
    else if (kernel == "openmp_4c") matmul_omp(C, A, B, N);
    else if (kernel == "avx_blocked") matmul_avx_blk(C, A, B, N);

    free_matrix(A);
    free_matrix(B);
    free_matrix(C);
    return 0;
}
