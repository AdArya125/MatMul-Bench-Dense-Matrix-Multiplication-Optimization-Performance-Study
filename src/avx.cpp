// avx.cpp - Vectorised matrix multiplication using AVX2 intrinsics.
// Build: g++ -O2 -mavx2 -mfma -std=c++17 -I../include avx.cpp -o avx_matmul

#include <immintrin.h>
#include <iostream>
#include "matrix_utils.hpp"

// Scalar i-k-j baseline
void matmul_scalar_ikj(Matrix C, const Matrix A, const Matrix B, int N) {
    zero_matrix(C, N);
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k) {
            scalar_t a = A[i * N + k];
            for (int j = 0; j < N; ++j)
                C[i * N + j] += a * B[k * N + j];
        }
}

// AVX2 vectorised kernel processing 8 floats at a time
void matmul_avx(Matrix C, const Matrix A, const Matrix B, int N) {
    zero_matrix(C, N);
    int j_simd_end = (N / 8) * 8;

    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            __m256 a_ik = _mm256_set1_ps(A[i * N + k]);

            for (int j = 0; j < j_simd_end; j += 8) {
                __m256 c_vec = _mm256_loadu_ps(&C[i * N + j]);
                __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                c_vec = _mm256_fmadd_ps(a_ik, b_vec, c_vec);
                _mm256_storeu_ps(&C[i * N + j], c_vec);
            }

            // Remainder scalar loop for non-multiple of 8 sizes
            scalar_t a_scalar = A[i * N + k];
            for (int j = j_simd_end; j < N; ++j)
                C[i * N + j] += a_scalar * B[k * N + j];
        }
    }
}

// Combined cache blocking + AVX SIMD kernel
void matmul_avx_blocked(Matrix C, const Matrix A, const Matrix B, int N, int block) {
    zero_matrix(C, N);

    for (int ii = 0; ii < N; ii += block) {
        for (int kk = 0; kk < N; kk += block) {
            for (int jj = 0; jj < N; jj += block) {
                int i_end = std::min(ii + block, N);
                int k_end = std::min(kk + block, N);
                int j_end = std::min(jj + block, N);

                for (int i = ii; i < i_end; ++i) {
                    for (int k = kk; k < k_end; ++k) {
                        __m256 a_ik = _mm256_set1_ps(A[i * N + k]);
                        int j = jj;

                        for (; j + 8 <= j_end; j += 8) {
                            __m256 c_vec = _mm256_loadu_ps(&C[i * N + j]);
                            __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                            c_vec = _mm256_fmadd_ps(a_ik, b_vec, c_vec);
                            _mm256_storeu_ps(&C[i * N + j], c_vec);
                        }
                        scalar_t a_s = A[i * N + k];
                        for (; j < j_end; ++j)
                            C[i * N + j] += a_s * B[k * N + j];
                    }
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? std::stoi(argv[1]) : 1024;

    std::cout << "AVX/AVX2 SIMD matmul N=" << N << "\n\n";

    Matrix A = alloc_matrix(N);
    Matrix B = alloc_matrix(N);
    Matrix C = alloc_matrix(N);
    Matrix Cref = alloc_matrix(N);

    fill_random(A, N, 42);
    fill_random(B, N, 99);

    matmul_scalar_ikj(Cref, A, B, N);

    auto bench = [&](auto fn, const std::string& label) {
        fn(C, A, B, N);
        double best = 1e18;
        for (int r = 0; r < 3; ++r) {
            Timer t; t.start();
            fn(C, A, B, N);
            best = std::min(best, t.elapsed_ms());
        }
        bool ok = matrices_close(C, Cref, N, 1e-2f);
        std::cout << std::left << std::setw(20) << label
                  << std::right << std::setw(10) << std::fixed << std::setprecision(3) << best << " ms | "
                  << std::setprecision(2) << compute_gflops(N, best) << " GFLOP/s | "
                  << (ok ? "OK" : "FAIL") << "\n";
        return best;
    };

    double t_scalar = bench(matmul_scalar_ikj, "scalar i-k-j");
    double t_avx = bench(matmul_avx, "avx i-k-j");

    auto avx_blocked64 = [](Matrix C2, const Matrix A2, const Matrix B2, int N2) {
        matmul_avx_blocked(C2, A2, B2, N2, 64);
    };
    double t_avx_blk = bench(avx_blocked64, "avx+blocked-64");

    std::cout << "\nSpeedup AVX over scalar:       " << std::fixed << std::setprecision(2) << t_scalar / t_avx << "x\n";
    std::cout << "Speedup AVX+block over scalar: " << t_scalar / t_avx_blk << "x\n";

    free_matrix(A);
    free_matrix(B);
    free_matrix(C);
    free_matrix(Cref);
    return 0;
}
