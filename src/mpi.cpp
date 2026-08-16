// mpi.cpp - Distributed memory matrix multiplication using MPI.
// Build: mpicxx -O2 -std=c++17 -I../include mpi.cpp -o mpi_matmul

#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include "matrix_utils.hpp"

// Computes local rectangular row slice of A with full B
void matmul_local(float* C_local, const float* A_local, const float* B, int rows, int N) {
    memset(C_local, 0, static_cast<size_t>(rows) * N * sizeof(float));
    for (int i = 0; i < rows; ++i) {
        for (int k = 0; k < N; ++k) {
            float a = A_local[i * N + k];
            for (int j = 0; j < N; ++j)
                C_local[i * N + j] += a * B[k * N + j];
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int N = (argc > 1) ? std::stoi(argv[1]) : 1024;
    int rows_per_proc = N / nprocs;

    if (rank == 0) {
        std::cout << "MPI matmul N=" << N << " procs=" << nprocs << " rows/proc=" << rows_per_proc << "\n\n";
    }

    Matrix A_full = nullptr;
    Matrix C_full = nullptr;
    Matrix B_full = alloc_matrix(N);

    if (rank == 0) {
        A_full = alloc_matrix(N);
        C_full = alloc_matrix(N);
        fill_random(A_full, N, 42);
        fill_random(B_full, N, 99);
    }

    double t_comm_start = MPI_Wtime();
    if (rank != 0) fill_random(B_full, N, 99);
    MPI_Bcast(B_full, N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    double t_bcast = MPI_Wtime() - t_comm_start;

    float* A_local = nullptr;
    if (posix_memalign(reinterpret_cast<void**>(&A_local), 64, static_cast<size_t>(rows_per_proc) * N * sizeof(float)) != 0)
        return 1;

    double t_scatter_start = MPI_Wtime();
    MPI_Scatter(A_full, rows_per_proc * N, MPI_FLOAT,
                A_local, rows_per_proc * N, MPI_FLOAT,
                0, MPI_COMM_WORLD);
    double t_scatter = MPI_Wtime() - t_scatter_start;

    float* C_local = nullptr;
    if (posix_memalign(reinterpret_cast<void**>(&C_local), 64, static_cast<size_t>(rows_per_proc) * N * sizeof(float)) != 0)
        return 1;

    MPI_Barrier(MPI_COMM_WORLD);
    double t_compute_start = MPI_Wtime();
    matmul_local(C_local, A_local, B_full, rows_per_proc, N);
    MPI_Barrier(MPI_COMM_WORLD);
    double t_compute = MPI_Wtime() - t_compute_start;

    double t_gather_start = MPI_Wtime();
    MPI_Gather(C_local, rows_per_proc * N, MPI_FLOAT,
               C_full, rows_per_proc * N, MPI_FLOAT,
               0, MPI_COMM_WORLD);
    double t_gather = MPI_Wtime() - t_gather_start;

    if (rank == 0) {
        double t_total = t_bcast + t_scatter + t_compute + t_gather;
        std::cout << "Timing breakdown:\n";
        std::cout << "  Bcast B:    " << t_bcast * 1e3 << " ms\n";
        std::cout << "  Scatter A:  " << t_scatter * 1e3 << " ms\n";
        std::cout << "  Compute:    " << t_compute * 1e3 << " ms (" << compute_gflops(N, t_compute * 1000.0) << " GFLOP/s)\n";
        std::cout << "  Gather C:   " << t_gather * 1e3 << " ms\n";
        std::cout << "  Total:      " << t_total * 1e3 << " ms (" << compute_gflops(N, t_total * 1000.0) << " GFLOP/s e2e)\n";
    }

    if (rank == 0) {
        free_matrix(A_full);
        free_matrix(C_full);
    }
    free_matrix(B_full);
    free(A_local);
    free(C_local);

    MPI_Finalize();
    return 0;
}
