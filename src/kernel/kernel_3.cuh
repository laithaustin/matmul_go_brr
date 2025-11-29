#pragma once

#include <cuda_runtime.h>
#include <sys/select.h>

/*
 * Kernel 3: SMEM Blocking
 *
 * Some notes to make for testing on the A10 gpu:
 * We have about 100KB of SMEM per SM or about 48 KB per block
 * That means we can fit about 25k float32s in SMEM per SM
 * However, realistically we are bounded to 48 KB per block w/out
 * CUDA yelling at us (needing to configure more memory for ourselves).
 *
 * So the plan here is to simply use blocking and move memory into SMEM
 */

template <const int BLOCKSIZE>
__global__ void mysgemm_v3(int M, int N, int K, float alpha, float *A,
                          float *B, float beta, float *C) {
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    // get row and col for the block
    uint cRow = blockIdx.x;
    uint cCol = blockIdx.y;
    uint threadRow = threadIdx.x / BLOCKSIZE;
    uint threadCol = threadIdx.x % BLOCKSIZE;

    // setup pointers to the right initial positions
    A += cRow * BLOCKSIZE * K;
    B += cCol * BLOCKSIZE;
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

    float tmp = 0.0f;
    for(uint dotIdx = 0; dotIdx < K; dotIdx += BLOCKSIZE) {
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

        // sync threads
        __syncthreads();
        // update A and B
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        // apply dot product
        // 32 threads means each one can handle one row and one col
        for (int i=0; i < BLOCKSIZE; i++) {
            tmp += As[threadRow * BLOCKSIZE + i] * Bs[i * BLOCKSIZE + threadCol];
        }
        // sync
        __syncthreads();
    }

    // write back to global memory
    C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}
