#pragma once

#include <cuda_runtime.h>

/*
 * Kernel 2: Shared Memory Tiling
 * 
 * Optimizations over kernel 1:
 * - Tiles of A and B are loaded into shared memory
 * - Each element in shared memory is reused multiple times
 * - Reduces global memory bandwidth requirements
 * - Expected performance: ~30% of cuBLAS
 * 
 * Template parameters:
 *   BM, BN: Block dimensions (e.g., 128x128)
 *   BK: Tile size along K dimension (e.g., 8)
 *   TM: Thread tile size (e.g., 64 elements per thread)
 */

template <const int BM, const int BN, const int BK, const int TM>
__global__ void mysgemm_v2(int M, int N, int K, float alpha, float *A, 
                          float *B, float beta, float *C) {
    // Shared memory for tiles
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    
    // Thread coordinates within block
    const int tx = threadIdx.x % BN;
    const int ty = threadIdx.x / BN;
    
    // Global row and column
    const int row = blockIdx.x * BM + ty;
    const int col = blockIdx.y * BN + tx;
    
    // Accumulator
    float tmp = 0.0f;
    
    // Loop over tiles along K dimension
    for (int bk = 0; bk < K; bk += BK) {
        // Load tile of A into shared memory
        for (int i = 0; i < BM; i += (BM * BN) / TM) {
            int a_row = blockIdx.x * BM + (threadIdx.x / BK) + i;
            int a_col = bk + (threadIdx.x % BK);
            if (a_row < M && a_col < K) {
                As[(threadIdx.x / BK + i) * BK + (threadIdx.x % BK)] = 
                    A[a_row * K + a_col];
            } else {
                As[(threadIdx.x / BK + i) * BK + (threadIdx.x % BK)] = 0.0f;
            }
        }
        
        // Load tile of B into shared memory
        for (int i = 0; i < BK; i += (BM * BN) / TM) {
            int b_row = bk + (threadIdx.x / BN) + i;
            int b_col = blockIdx.y * BN + (threadIdx.x % BN);
            if (b_row < K && b_col < N) {
                Bs[(threadIdx.x / BN + i) * BN + (threadIdx.x % BN)] = 
                    B[b_row * N + b_col];
            } else {
                Bs[(threadIdx.x / BN + i) * BN + (threadIdx.x % BN)] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute using shared memory
        for (int k = 0; k < BK; ++k) {
            tmp += As[ty * BK + k] * Bs[k * BN + tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}
