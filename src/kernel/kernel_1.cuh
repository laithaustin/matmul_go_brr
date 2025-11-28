#pragma once

#include <cuda_runtime.h>

/*
 * Kernel 1: Naive SGEMM Implementation
 *
 * Each thread computes one element of C.
 * - No shared memory usage - non-coalesced
 * - Direct global memory access
 * - Expected performance: ~15% of cuBLAS
 * - Memory Bandwidth needed (Assuming NxK and KxM matrices):
 * - We do 2*K + 1 loads from GMEM per thread (assuming no caching)
 * - Assuming K,M,N = 4096 -> 4096^2 * (2*4096 + 1) * 4 (bytes in a float) = ~548 GB of memory traffic
 *
 * Block configuration: 32x32 threads (1024 threads per block)
 * Grid configuration: (M/32, N/32)
 */

__global__ void __launch_bounds__(1024)
mysgemm_v1(int M, int N, int K, float alpha, float *A, float *B,
           float beta, float *C) {
    // Calculate global row and column for this thread
    const int row = blockIdx.x * blockDim.x + threadIdx.x; // in this schema x increments then y
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    // Bounds check
    if (row < M && col < N) {
        float tmp = 0.0f;

        // Compute dot product
        for (int i = 0; i < K; ++i) {
            tmp += A[row * K + i] * B[i * N + col];
        }

        // GEMM: C = alpha * A @ B + beta * C
        // since row is changing this means our access pattern is non-coalesced for A, C
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}
