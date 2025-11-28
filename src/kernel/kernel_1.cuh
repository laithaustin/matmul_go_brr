#pragma once

#include <cuda_runtime.h>

/*
 * Kernel 1: Naive SGEMM Implementation
 * 
 * Each thread computes one element of C.
 * - No shared memory usage
 * - Direct global memory access
 * - Expected performance: ~15% of cuBLAS
 * 
 * Block configuration: 32x32 threads (1024 threads per block)
 * Grid configuration: (M/32, N/32)
 */

__global__ void __launch_bounds__(1024) 
mysgemm_v1(int M, int N, int K, float alpha, float *A, float *B, 
           float beta, float *C) {
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
