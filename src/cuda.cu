// cuda.cu - CUDA GPU matrix multiplication (Naive vs Shared Memory Tiled).
// Build: nvcc -O2 -arch=sm_75 -std=c++17 -I../include cuda.cu -o cuda_matmul

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <iomanip>
#include "matrix_utils.hpp"

#define cudaCheck(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define TILE_SZ 32

// Kernel 1: Each thread directly accesses global memory
__global__ void matmul_naive_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// Kernel 2: Threads load sub-tiles into shared memory to reuse data
__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_SZ][TILE_SZ];
    __shared__ float Bs[TILE_SZ][TILE_SZ];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SZ + ty;
    int col = blockIdx.x * TILE_SZ + tx;

    float sum = 0.0f;
    int num_phases = (N + TILE_SZ - 1) / TILE_SZ;

    for (int phase = 0; phase < num_phases; ++phase) {
        int a_col = phase * TILE_SZ + tx;
        As[ty][tx] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;

        int b_row = phase * TILE_SZ + ty;
        Bs[ty][tx] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int m = 0; m < TILE_SZ; ++m)
            sum += As[ty][m] * Bs[m][tx];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

float* gpu_alloc_and_copy(const float* h_data, int n) {
    float* d_ptr;
    size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
    cudaCheck(cudaMalloc(&d_ptr, bytes));
    cudaCheck(cudaMemcpy(d_ptr, h_data, bytes, cudaMemcpyHostToDevice));
    return d_ptr;
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? std::stoi(argv[1]) : 1024;

    cudaDeviceProp prop;
    cudaCheck(cudaGetDeviceProperties(&prop, 0));
    std::cout << "CUDA matmul N=" << N << " GPU=" << prop.name << "\n\n";

    Matrix h_A = alloc_matrix(N);
    Matrix h_B = alloc_matrix(N);
    Matrix h_C = alloc_matrix(N);

    fill_random(h_A, N, 42);
    fill_random(h_B, N, 99);
    zero_matrix(h_C, N);

    cudaEvent_t ev_start, ev_stop;
    cudaCheck(cudaEventCreate(&ev_start));
    cudaCheck(cudaEventCreate(&ev_stop));

    Timer cpu_timer;
    cpu_timer.start();
    float* d_A = gpu_alloc_and_copy(h_A, N);
    float* d_B = gpu_alloc_and_copy(h_B, N);
    float* d_C;
    cudaCheck(cudaMalloc(&d_C, (size_t)N * N * sizeof(float)));
    cudaCheck(cudaMemset(d_C, 0, (size_t)N * N * sizeof(float)));
    double h2d_ms = cpu_timer.elapsed_ms();
    std::cout << "Host->Device transfer: " << std::fixed << std::setprecision(3) << h2d_ms << " ms\n";

    dim3 block(TILE_SZ, TILE_SZ);
    dim3 grid((N + TILE_SZ - 1) / TILE_SZ, (N + TILE_SZ - 1) / TILE_SZ);

    auto bench_kernel = [&](auto kfn, const std::string& label) {
        kfn<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaCheck(cudaDeviceSynchronize());

        float best_ms = 1e9f;
        for (int r = 0; r < 3; ++r) {
            cudaCheck(cudaEventRecord(ev_start));
            kfn<<<grid, block>>>(d_A, d_B, d_C, N);
            cudaCheck(cudaEventRecord(ev_stop));
            cudaCheck(cudaEventSynchronize(ev_stop));

            float ms;
            cudaCheck(cudaEventElapsedTime(&ms, ev_start, ev_stop));
            best_ms = std::min(best_ms, ms);
        }
        std::cout << std::left << std::setw(25) << label
                  << std::right << std::setw(10) << std::fixed << std::setprecision(3) << best_ms << " ms | "
                  << std::setprecision(2) << compute_gflops(N, best_ms) << " GFLOP/s\n";
        return best_ms;
    };

    float t_naive = bench_kernel(matmul_naive_kernel, "gpu-naive");
    float t_tiled = bench_kernel(matmul_tiled_kernel, "gpu-tiled (shared mem)");

    cpu_timer.start();
    cudaCheck(cudaMemcpy(h_C, d_C, (size_t)N * N * sizeof(float), cudaMemcpyDeviceToHost));
    double d2h_ms = cpu_timer.elapsed_ms();
    std::cout << "Device->Host transfer: " << d2h_ms << " ms\n";
    std::cout << "Total end-to-end:      " << (h2d_ms + t_tiled + d2h_ms) << " ms\n\n";

    std::cout << "Speedup tiled over naive GPU: " << std::fixed << std::setprecision(2) << t_naive / t_tiled << "x\n";

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
    free_matrix(h_A); free_matrix(h_B); free_matrix(h_C);
    return 0;
}
