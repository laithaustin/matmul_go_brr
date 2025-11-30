#pragma once

#include <cuda_runtime.h>

/*
 * Kernel 4: 1D Block Tiling
 *
 *
 * Template parameters:
 *   BM, BN: Block dimensions (e.g., 128x128)
 *   BK: Tile size along K dimension (e.g., 8)
 *   TM: Thread tile for computing a single slice of a col of C
 */

template <const uint BM, const uint BN, const uint BK, const uint TM>
__global__ void mysgemm_v4(int M, int N, int K, float alpha, float *A,
                          float *B, float beta, float *C) {
    // check tile sizes preemptively
    static_assert(BM * BK == (BM / TM) * BN, "Thread count must match A tile size");
    static_assert(BK * BN == (BM / TM) * BN, "Thread count must match B tile size");

    // setup SMEM (enough for a tile)
    __shared__ float As[BM * BK];
    __shared__ float Bs[BN * BK];
    // get thread, block row/col index data for loading
    uint cRow = blockIdx.y;
    uint cCol = blockIdx.x;
    uint threadColB = threadIdx.x % BN; // which col of B, C are we computing the partials for
    uint threadColA = threadIdx.x % BK;
    uint threadRowB = threadIdx.x / BN;
    uint threadRowA = threadIdx.x / BK;
    // for computation
    uint threadCol = threadIdx.x % BN;           // which column of C (0-63)
    uint threadRow = threadIdx.x / BN;           // which thread row (0-7), each handles TM rows of C
    // setup pointers to the right location
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    float threadResults[TM] = {0.0};

    // main logic is as follows:
    // we iterate through the K dimension on the outer loop with a (BmxBk) matrix in A
    for (int idx = 0; idx < K; idx += BK) {
        // first we load our SMEM into GMEM
        As[threadRowA * BK + threadColA] = A[threadRowA * K + threadColA];
        Bs[threadRowB * BN + threadColB] = B[threadRowB * N + threadColB];
        __syncthreads();
        // we iterate through the cols of A in the inner loop
        for(int colA = 0; colA < BK; colA++) {
            // we compute the partial dot product as we take each single value in B and multiply it by A
            // only grabbing one element of B for this given partial
            float tempB = Bs[threadCol + colA * BN];
            for (int rowA = 0; rowA < TM; rowA++) {
                threadResults[rowA] += tempB * As[(rowA + threadRow * TM) * BK + colA];
            }
        }

        // advance blocks and sync
        A += BK;
        B += BK * N;
        __syncthreads();
    }

    //write back to GMEM
    for (int i = 0; i < TM; ++i) {
        // tricky thing here: need to segment the A col that we work among multiple threads of size TM
        C[(threadRow * TM + i) * N + threadCol] = alpha * threadResults[i] + beta * C[(threadRow * TM + i) * N + threadCol];
    }
}
