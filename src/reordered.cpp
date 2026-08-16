// reordered.cpp - Loop reordering benchmark (i-j-k vs i-k-j vs j-k-i).
// Build: g++ -O2 -std=c++17 -I../include reordered.cpp -o reordered

#include <iostream>
#include <string>
#include "matrix_utils.hpp"

void matmul_ijk(Matrix C, const Matrix A, const Matrix B, int N) {
    zero_matrix(C, N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            scalar_t sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

void matmul_ikj(Matrix C, const Matrix A, const Matrix B, int N) {
    zero_matrix(C, N);
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k) {
            scalar_t a_ik = A[i * N + k];
            for (int j = 0; j < N; ++j)
                C[i * N + j] += a_ik * B[k * N + j];
        }
}

void matmul_jki(Matrix C, const Matrix A, const Matrix B, int N) {
    zero_matrix(C, N);
    for (int j = 0; j < N; ++j)
        for (int k = 0; k < N; ++k) {
            scalar_t b_kj = B[k * N + j];
            for (int i = 0; i < N; ++i)
                C[i * N + j] += A[i * N + k] * b_kj;
        }
}

double bench_kernel(void (*fn)(Matrix, const Matrix, const Matrix, int), Matrix C, const Matrix A, const Matrix B, int N, const std::string& label, int reps = 3) {
    fn(C, A, B, N);
    double best_ms = 1e18;
    for (int r = 0; r < reps; ++r) {
        Timer t;
        t.start();
        fn(C, A, B, N);
        double ms = t.elapsed_ms();
        best_ms = std::min(best_ms, ms);
    }
    BenchResult res{label, N, best_ms, compute_gflops(N, best_ms)};
    print_result(res);
    return best_ms;
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? std::stoi(argv[1]) : 512;

    std::cout << "Loop-reordered matmul N=" << N << "\n\n";

    Matrix A = alloc_matrix(N);
    Matrix B = alloc_matrix(N);
    Matrix C = alloc_matrix(N);

    fill_random(A, N, 42);
    fill_random(B, N, 99);

    double t_ijk = bench_kernel(matmul_ijk, C, A, B, N, "i-j-k (naive)");
    double t_ikj = bench_kernel(matmul_ikj, C, A, B, N, "i-k-j (best)");
    double t_jki = bench_kernel(matmul_jki, C, A, B, N, "j-k-i (worst)");

    std::cout << "\nSpeedup i-k-j over i-j-k: " << std::fixed << std::setprecision(2) << t_ijk / t_ikj << "x\n";
    std::cout << "Slowdown j-k-i vs i-j-k:  " << std::fixed << std::setprecision(2) << t_jki / t_ijk << "x\n";

    free_matrix(A);
    free_matrix(B);
    free_matrix(C);
    return 0;
}
