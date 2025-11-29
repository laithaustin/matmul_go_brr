#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "src/kernel/kernel_3.cuh"

#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

int main() {
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    printf("Profiling kernel 3 with size: %d x %d x %d\n", M, N, K);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    size_t size = M * K * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    // Initialize with some data (just zeros is fine for profiling)
    CUDA_CHECK(cudaMemset(d_A, 0, size));
    CUDA_CHECK(cudaMemset(d_B, 0, K * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));

    // Kernel 3 configuration
    const int BLOCKSIZE = 32;
    dim3 blockDim(BLOCKSIZE * BLOCKSIZE);
    dim3 gridDim(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE));

    printf("Grid: (%d, %d), Block: (%d)\n", gridDim.x, gridDim.y, blockDim.x);

    // Configure shared memory preference
    CUDA_CHECK(cudaFuncSetAttribute(mysgemm_v3<32>,
                                    cudaFuncAttributePreferredSharedMemoryCarveout,
                                    cudaSharedmemCarveoutMaxShared));

    // Warmup
    mysgemm_v3<32><<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Warmup complete. Ready for profiling.\n");
    printf("Running kernel...\n");

    // Main kernel call (this is what will be profiled)
    mysgemm_v3<32><<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Kernel execution complete.\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
