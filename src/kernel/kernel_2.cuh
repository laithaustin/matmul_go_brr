#pragma once

#include <cuda_runtime.h>

/*
 * Kernel 2: Coalesced GMEM Access
 *
 */
 __global__ void __launch_bounds__(1024)
mysgemm_v2(int M, int N, int K, float alpha, float *A,
                          float *B, float beta, float *C) {
    // Shared memory for tiles
    const int BLOCKSIZE = 32;

    // Calculate global row and column for this thread
    const int row = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int col = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    // Bounds check
    if (row < M && col < N) {
        float tmp = 0.0f;

        // Compute dot product
        for (int i = 0; i < K; ++i) {
            tmp += A[row * K + i] * B[i * N + col];
        }

        // GEMM: C = alpha * A @ B + beta * C
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}
