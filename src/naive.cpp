// naive.cpp - Baseline i-j-k matrix multiplication.
// Build: g++ -O2 -std=c++17 -I../include naive.cpp -o naive

#include <iostream>
#include <string>
#include "matrix_utils.hpp"

// Standard triple loop.
// Note: As k grows, B[k][j] jumps by N floats (stride of N), missing L1/L2 cache lines.
void matmul_naive(Matrix C, const Matrix A, const Matrix B, int N) {
    zero_matrix(C, N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            scalar_t sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? std::stoi(argv[1]) : 512;

    std::cout << "naive i-j-k N=" << N << "\n";

    Matrix A = alloc_matrix(N);
    Matrix B = alloc_matrix(N);
    Matrix C = alloc_matrix(N);

    fill_random(A, N, 42);
    fill_random(B, N, 99);

    // Warm-up run
    matmul_naive(C, A, B, N);

    const int REPS = 3;
    double best_ms = 1e18;
    for (int r = 0; r < REPS; ++r) {
        Timer t;
        t.start();
        matmul_naive(C, A, B, N);
        double ms = t.elapsed_ms();
        best_ms = std::min(best_ms, ms);
        std::cout << "  run " << r + 1 << ": " << std::fixed << std::setprecision(3) << ms << " ms\n";
    }

    std::cout << "\nbest: " << best_ms << " ms | "
              << std::fixed << std::setprecision(3) << compute_gflops(N, best_ms) << " GFLOP/s\n";

    free_matrix(A);
    free_matrix(B);
    free_matrix(C);
}
