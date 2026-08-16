// hybrid.cpp - Hybrid MPI + OpenMP matrix multiplication.
// Build: mpicxx -O2 -fopenmp -std=c++17 -I../include hybrid.cpp -o hybrid_matmul

#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include "matrix_utils.hpp"

// Local computation multi-threaded via OpenMP
void matmul_omp_local(float* C_local, const float* A_local, const float* B, int rows, int N) {
    memset(C_local, 0, static_cast<size_t>(rows) * N * sizeof(float));

    #pragma omp parallel for schedule(static) shared(A_local, B, C_local)
    for (int i = 0; i < rows; ++i) {
        for (int k = 0; k < N; ++k) {
            float a = A_local[i * N + k];
            for (int j = 0; j < N; ++j)
                C_local[i * N + j] += a * B[k * N + j];
        }
    }
}

int main(int argc, char* argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int N = (argc > 1) ? std::stoi(argv[1]) : 1024;
    int nthreads = omp_get_max_threads();
    int rows_per_proc = N / nprocs;

    if (rank == 0) {
        std::cout << "Hybrid MPI+OpenMP matmul N=" << N << " procs=" << nprocs
                  << " threads/proc=" << nthreads << " total_threads=" << nprocs * nthreads << "\n\n";
    }

    Matrix B_full = alloc_matrix(N);
    Matrix A_full = nullptr, C_full = nullptr;

    if (rank == 0) {
        A_full = alloc_matrix(N);
        C_full = alloc_matrix(N);
        fill_random(A_full, N, 42);
        fill_random(B_full, N, 99);
    }

    MPI_Bcast(B_full, N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    float* A_local = nullptr;
    if (posix_memalign(reinterpret_cast<void**>(&A_local), 64, static_cast<size_t>(rows_per_proc) * N * sizeof(float)) != 0)
        return 1;

    MPI_Scatter(A_full, rows_per_proc * N, MPI_FLOAT,
                A_local, rows_per_proc * N, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    float* C_local = nullptr;
    if (posix_memalign(reinterpret_cast<void**>(&C_local), 64, static_cast<size_t>(rows_per_proc) * N * sizeof(float)) != 0)
        return 1;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    matmul_omp_local(C_local, A_local, B_full, rows_per_proc, N);
    MPI_Barrier(MPI_COMM_WORLD);
    double t_compute = MPI_Wtime() - t0;

    MPI_Gather(C_local, rows_per_proc * N, MPI_FLOAT,
               C_full, rows_per_proc * N, MPI_FLOAT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Compute time: " << t_compute * 1e3 << " ms | Throughput: "
                  << compute_gflops(N, t_compute * 1000.0) << " GFLOP/s\n";
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
