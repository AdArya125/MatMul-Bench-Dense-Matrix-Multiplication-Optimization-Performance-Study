// test_blocked_ijk.cpp - Experimenting with i-j-k inside cache blocks.
// Build: g++ -O2 -std=c++17 -I../include test_blocked_ijk.cpp -o test_blocked_ijk

#include <iostream>
#include <algorithm>
#include "matrix_utils.hpp"

// Blocked with i-k-j inner order (our standard blocked kernel)
void matmul_blocked_ikj(Matrix C, const Matrix A, const Matrix B, int N, int block) {
    zero_matrix(C, N);
    for (int ii = 0; ii < N; ii += block) {
        for (int kk = 0; kk < N; kk += block) {
            for (int jj = 0; jj < N; jj += block) {
                int i_end = std::min(ii + block, N);
                int k_end = std::min(kk + block, N);
                int j_end = std::min(jj + block, N);

                for (int i = ii; i < i_end; ++i) {
                    for (int k = kk; k < k_end; ++k) {
                        scalar_t a_ik = A[i * N + k];
                        for (int j = jj; j < j_end; ++j) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

// Blocked with i-j-k inner order (testing i-j-k inside tiles)
void matmul_blocked_ijk(Matrix C, const Matrix A, const Matrix B, int N, int block) {
    zero_matrix(C, N);
    for (int ii = 0; ii < N; ii += block) {
        for (int jj = 0; jj < N; jj += block) {
            for (int kk = 0; kk < N; kk += block) {
                int i_end = std::min(ii + block, N);
                int j_end = std::min(jj + block, N);
                int k_end = std::min(kk + block, N);

                for (int i = ii; i < i_end; ++i) {
                    for (int j = jj; j < j_end; ++j) {
                        scalar_t sum = 0.0f;
                        for (int k = kk; k < k_end; ++k) {
                            sum += A[i * N + k] * B[k * N + j];
                        }
                        C[i * N + j] += sum;
                    }
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? std::stoi(argv[1]) : 1024;
    int blk = 64;

    std::cout << "Experiment: Comparing Blocked i-k-j vs Blocked i-j-k (N=" << N << ", Block=" << blk << ")\n\n";

    Matrix A = alloc_matrix(N);
    Matrix B = alloc_matrix(N);
    Matrix C = alloc_matrix(N);

    fill_random(A, N, 42);
    fill_random(B, N, 99);

    auto bench = [&](auto fn, std::string label) {
        fn(C, A, B, N, blk);
        double best = 1e18;
        for (int r = 0; r < 3; ++r) {
            Timer t; t.start();
            fn(C, A, B, N, blk);
            best = std::min(best, t.elapsed_ms());
        }
        std::cout << std::left << std::setw(22) << label
                  << std::right << std::setw(10) << std::fixed << std::setprecision(3) << best << " ms | "
                  << std::setprecision(2) << compute_gflops(N, best) << " GFLOP/s\n";
        return best;
    };

    double t_ikj = bench(matmul_blocked_ikj, "Blocked i-k-j");
    double t_ijk = bench(matmul_blocked_ijk, "Blocked i-j-k");

    std::cout << "\nDifference: Blocked i-k-j is " << std::fixed << std::setprecision(2)
              << t_ijk / t_ikj << "x faster than Blocked i-j-k!\n";

    free_matrix(A); free_matrix(B); free_matrix(C);
    return 0;
}
