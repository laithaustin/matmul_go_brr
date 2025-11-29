#pragma once

#include <cuda_runtime.h>

/*
 * Kernel 2: Coalesced GMEM Access
 *
 * We get some easy gains here just by making sure that we can get coalesced reads within
 * warps to adjacent locations in memory for both A, C. Reason here is since GPU will be able to
 * cache data recent to previous threads that we are reading sequentially.
 *
 * Key here is understanding that warps executue in groups of 32 threads. Warps are assigned
 * to 4(typically on modern gpu architecture) warp schedulers per SM. these warps run parallel
 * in the schedulers and concurrently per scheduler. threadIds are dictated per warp and threadIdx.x
 * tells us threads that are adjacent to each other in their respective warp.
 *
 */
 __global__ void __launch_bounds__(1024)
mysgemm_v2(int M, int N, int K, float alpha, float *A,
                          float *B, float beta, float *C) {

    const uint BLOCKSIZE = 32;
    // Calculate global row and column for this thread
    const uint row = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);// threads in the same warp have same row
    const uint col = blockIdx.y * BLOCKSIZE  + (threadIdx.x % BLOCKSIZE); // col changes

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
