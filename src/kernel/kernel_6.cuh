#pragma once

#include <cuda_runtime.h>

/*
 * Kernel 6: Float4 Vectorization
 * 
 * Optimizations over kernel 5:
 * - Use float4 vector loads/stores
 * - Reduces memory instruction overhead
 * - Better utilization of memory bandwidth
 * - Expected performance: ~91% of cuBLAS
 */

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void mysgemm_v6(int M, int N, int K, float alpha, float *A, 
                          float *B, float beta, float *C) {
    // TODO: Implement float4 vectorization
    // Placeholder: same as kernel 5
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    
    const int tx = threadIdx.x % BN;
    const int ty = threadIdx.x / BN;
    const int row = blockIdx.x * BM + ty;
    const int col = blockIdx.y * BN + tx;
    
    float tmp = 0.0f;
    
    for (int bk = 0; bk < K; bk += BK) {
        for (int i = 0; i < BM; i += (BM * BN) / (TM * TN)) {
            int a_row = blockIdx.x * BM + (threadIdx.x / BK) + i;
            int a_col = bk + (threadIdx.x % BK);
            if (a_row < M && a_col < K) {
                As[(threadIdx.x / BK + i) * BK + (threadIdx.x % BK)] = 
                    A[a_row * K + a_col];
            } else {
                As[(threadIdx.x / BK + i) * BK + (threadIdx.x % BK)] = 0.0f;
            }
        }
        
        for (int i = 0; i < BK; i += (BM * BN) / (TM * TN)) {
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
        
        for (int k = 0; k < BK; ++k) {
            tmp += As[ty * BK + k] * Bs[k * BN + tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}
