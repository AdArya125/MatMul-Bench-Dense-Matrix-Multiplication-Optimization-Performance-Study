// openmp.cpp - Parallel matrix multiplication using OpenMP.
// Build: g++ -O2 -fopenmp -std=c++17 -I../include openmp.cpp -o openmp_matmul

#include <omp.h>
#include <iostream>
#include <vector>
#include "matrix_utils.hpp"

// Baseline serial i-k-j
void matmul_serial(Matrix C, const Matrix A, const Matrix B, int N) {
    zero_matrix(C, N);
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k) {
            scalar_t a = A[i * N + k];
            for (int j = 0; j < N; ++j)
                C[i * N + j] += a * B[k * N + j];
        }
}

// Parallel outer loop (i) across threads with static scheduling
void matmul_omp_static(Matrix C, const Matrix A, const Matrix B, int N, int nthreads) {
    zero_matrix(C, N);
    omp_set_num_threads(nthreads);

    #pragma omp parallel for schedule(static) shared(A, B, C)
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            scalar_t a = A[i * N + k];
            for (int j = 0; j < N; ++j)
                C[i * N + j] += a * B[k * N + j];
        }
    }
}

// Collapsing outer i and j loops for fine-grained work items
void matmul_omp_collapse(Matrix C, const Matrix A, const Matrix B, int N, int nthreads) {
    zero_matrix(C, N);
    omp_set_num_threads(nthreads);

    #pragma omp parallel for collapse(2) schedule(static) shared(A, B, C)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            scalar_t sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? std::stoi(argv[1]) : 1024;
    int max_threads = omp_get_max_threads();

    std::cout << "OpenMP matmul N=" << N << " max_threads=" << max_threads << "\n\n";

    Matrix A = alloc_matrix(N);
    Matrix B = alloc_matrix(N);
    Matrix C = alloc_matrix(N);

    fill_random(A, N, 42);
    fill_random(B, N, 99);

    matmul_serial(C, A, B, N);
    double best_serial = 1e18;
    for (int r = 0; r < 3; ++r) {
        Timer t; t.start();
        matmul_serial(C, A, B, N);
        best_serial = std::min(best_serial, t.elapsed_ms());
    }
    std::cout << "Serial baseline: " << best_serial << " ms ("
              << compute_gflops(N, best_serial) << " GFLOP/s)\n\n";

    std::vector<int> thread_counts;
    for (int t = 1; t <= max_threads; t *= 2)
        thread_counts.push_back(t);
    if (thread_counts.back() != max_threads)
        thread_counts.push_back(max_threads);

    for (int T : thread_counts) {
        matmul_omp_static(C, A, B, N, T);
        double best_ms = 1e18;
        for (int r = 0; r < 3; ++r) {
            Timer t; t.start();
            matmul_omp_static(C, A, B, N, T);
            best_ms = std::min(best_ms, t.elapsed_ms());
        }
        double speedup = best_serial / best_ms;
        double efficiency = (speedup / T) * 100.0;

        std::cout << "threads=" << T << " time=" << std::fixed << std::setprecision(3) << best_ms << " ms | "
                  << std::setprecision(2) << compute_gflops(N, best_ms) << " GFLOP/s | speedup="
                  << speedup << "x | efficiency=" << efficiency << "%\n";
    }

    matmul_omp_collapse(C, A, B, N, max_threads);
    double best_ms = 1e18;
    for (int r = 0; r < 3; ++r) {
        Timer t; t.start();
        matmul_omp_collapse(C, A, B, N, max_threads);
        best_ms = std::min(best_ms, t.elapsed_ms());
    }
    std::cout << "\nomp-collapse(2) @ " << max_threads << " threads: "
              << best_ms << " ms | speedup=" << (best_serial / best_ms) << "x\n";

    free_matrix(A);
    free_matrix(B);
    free_matrix(C);
    return 0;
}
