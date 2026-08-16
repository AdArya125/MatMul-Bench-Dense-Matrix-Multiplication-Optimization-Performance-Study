// blocked.cpp - Cache-blocked (tiled) matrix multiplication.
// Build: g++ -O2 -std=c++17 -I../include blocked.cpp -o blocked

#include <iostream>
#include <algorithm>
#include <vector>
#include "matrix_utils.hpp"

// Divide matrix into BxB blocks to keep working tiles inside L1/L2 cache
void matmul_blocked(Matrix C, const Matrix A, const Matrix B, int N, int block) {
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

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? std::stoi(argv[1]) : 512;

    std::cout << "Cache-blocked matmul N=" << N << "\n\n";

    Matrix A = alloc_matrix(N);
    Matrix B = alloc_matrix(N);
    Matrix C = alloc_matrix(N);
    Matrix Cref = alloc_matrix(N);

    fill_random(A, N, 42);
    fill_random(B, N, 99);

    // Reference result using unblocked i-k-j
    zero_matrix(Cref, N);
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k) {
            scalar_t a = A[i * N + k];
            for (int j = 0; j < N; ++j)
                Cref[i * N + j] += a * B[k * N + j];
        }

    std::vector<int> block_sizes = {16, 32, 64, 128, 256};

    for (int blk : block_sizes) {
        if (blk > N) continue;

        matmul_blocked(C, A, B, N, blk);

        double best_ms = 1e18;
        for (int r = 0; r < 3; ++r) {
            Timer t;
            t.start();
            matmul_blocked(C, A, B, N, blk);
            best_ms = std::min(best_ms, t.elapsed_ms());
        }

        bool ok = matrices_close(C, Cref, N, 1e-2f);

        std::string label = "blocked-" + std::to_string(blk);
        std::cout << std::left << std::setw(15) << label
                  << std::right << std::setw(10) << std::fixed << std::setprecision(3) << best_ms << " ms"
                  << std::setw(10) << std::fixed << std::setprecision(2) << compute_gflops(N, best_ms) << " GFLOP/s"
                  << "  " << (ok ? "OK" : "FAIL") << "\n";
    }

    free_matrix(A);
    free_matrix(B);
    free_matrix(C);
    free_matrix(Cref);
    return 0;
}
