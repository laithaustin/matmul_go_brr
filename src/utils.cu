#include <sys/time.h>
#include <iostream>
#include <fstream>
#include "utils.cuh"
#include "kernel.cuh"

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// ==================== CUDA Operations ====================

void cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

void CudaDeviceInfo() {
    int deviceId;
    cudaGetDevice(&deviceId);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    printf("Device ID: %d\n", deviceId);
    printf("  Name: %s\n", props.name);
    printf("  Compute Capability: %d.%d\n", props.major, props.minor);
    printf("  Total Global Memory: %.2f GB\n",
           props.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
    printf("  Shared Memory per Block: %.2f KB\n",
           props.sharedMemPerBlock / 1024.0);
    printf("  Max Threads per Block: %d\n", props.maxThreadsPerBlock);
    printf("  Multiprocessor Count: %d\n", props.multiProcessorCount);
    printf("  Warp Size: %d\n", props.warpSize);
}

// ==================== Matrix Operations ====================

void randomize_matrix(float *mat, int N) {
    struct timeval time {};
    gettimeofday(&time, nullptr);
    srand(time.tv_usec);
    for (int i = 0; i < N; i++) {
        float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        mat[i] = tmp;
    }
}

void copy_matrix(float *src, float *dest, int N) {
    int size = N * sizeof(float);
    for (int i = 0; i < size; i++) {
        dest[i] = src[i];
    }
}

void print_matrix(const float *A, int M, int N, std::ofstream &fs) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            fs << A[i * N + j];
            if (j != N - 1) fs << ", ";
        }
        if (i != M - 1) fs << ";" << std::endl;
    }
}

bool verify_matrix(float *matRef, float *matOut, int N) {
    double diff = 0.0;
    int i;
    for (i = 0; i < N; i++) {
        diff = fabs((double)matRef[i] - (double)matOut[i]);
        if (diff > 1e-2) {
            printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
                   matRef[i], matOut[i], diff, i);
            return false;
        }
    }
    return true;
}

// ==================== Timing Operations ====================

float get_sec() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return (1e6 * time.tv_sec + time.tv_usec);
}

float cpu_elapsed_time(float &beg, float &end) {
    return 1e-6 * (end - beg);
}

// ==================== Kernel Dispatcher ====================

void test_kernel(int kernel_num, int M, int N, int K, float alpha,
                float *A, float *B, float beta, float *C,
                cublasHandle_t handle) {

    switch (kernel_num) {
        case 0:
            // cuBLAS baseline
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       N, M, K, &alpha, B, N, A, K, &beta, C, N);
            break;
        case 1:
            {
                // Naive kernel - 32x32x1 thread blocks
                dim3 blockDim(32, 32, 1);
                dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
                mysgemm_v1<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
            }
            break;
        case 2:
            {
                // Introduce Global Coalescing
                dim3 blockDim(32*32);
                dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
                mysgemm_v2<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
            }
            break;
        case 3:
            {
                // 1D thread tile kernel
                dim3 blockDim(32*32);
                dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
                // since we are only accessing GMEM via SMEM we can carve out more SMEM space
                // for L1 cache here!
                cudaFuncSetAttribute(mysgemm_v3<32>,
                                     cudaFuncAttributePreferredSharedMemoryCarveout,
                                     cudaSharedmemCarveoutMaxShared);
                mysgemm_v3<32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
            }
            break;
        case 4:
            {
                // 2D thread tile kernel
                const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
                dim3 blockDim((BM * BN) / (TM * TN));
                dim3 gridDim(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
                mysgemm_v4<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
            }
            break;
        case 5:
            {
                // Register caching kernel
                const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
                dim3 blockDim((BM * BN) / (TM * TN));
                dim3 gridDim(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
                mysgemm_v5<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
            }
            break;
        case 6:
            {
                // Float4 vectorization kernel
                const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
                dim3 blockDim((BM * BN) / (TM * TN));
                dim3 gridDim(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
                mysgemm_v6<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
            }
            break;
        case 7:
            {
                // Double buffering kernel
                const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
                dim3 blockDim((BM * BN) / (TM * TN));
                dim3 gridDim(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
                mysgemm_v7<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
            }
            break;
        default:
            printf("Invalid kernel number: %d\n", kernel_num);
            exit(EXIT_FAILURE);
    }
}
