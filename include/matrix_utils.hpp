#pragma once

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <stdexcept>
#include <iostream>
#include <iomanip>

using scalar_t = float;
using Matrix = scalar_t*;

// 64-byte alignment satisfies both AVX (needs 32) and cache-line boundaries
inline Matrix alloc_matrix(int n) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 64, (size_t)n * n * sizeof(scalar_t)) != 0)
        throw std::bad_alloc();
    return static_cast<scalar_t*>(ptr);
}

inline void free_matrix(Matrix m) { free(m); }

inline void fill_random(Matrix m, int n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<scalar_t> dist(0.0f, 1.0f);
    for (int i = 0; i < n * n; ++i)
        m[i] = dist(rng);
}

inline void zero_matrix(Matrix m, int n) {
    memset(m, 0, (size_t)n * n * sizeof(scalar_t));
}

inline bool matrices_close(const Matrix a, const Matrix b, int n, float tol = 1e-3f) {
    for (int i = 0; i < n * n; ++i) {
        float diff = std::fabs(a[i] - b[i]);
        float mag = std::max(std::fabs(a[i]), std::fabs(b[i]));
        if (diff > tol * (mag + 1e-6f))
            return false;
    }
    return true;
}

struct Timer {
    using clock = std::chrono::high_resolution_clock;
    std::chrono::time_point<clock> t0;

    void start() { t0 = clock::now(); }

    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    }
};

inline double compute_gflops(int n, double time_ms) {
    return (2.0 * n * n * n) / (time_ms / 1000.0) / 1e9;
}

struct BenchResult {
    std::string label;
    int n;
    double time_ms;
    double gflops;
};

inline void print_result(const BenchResult& r) {
    std::cout << std::left << std::setw(20) << r.label
              << std::right << std::setw(6) << r.n << "x" << std::left << std::setw(6) << r.n
              << std::right << std::setw(12) << std::fixed << std::setprecision(3) << r.time_ms << " ms"
              << std::setw(10) << std::fixed << std::setprecision(2) << r.gflops << " GFLOP/s\n";
}
