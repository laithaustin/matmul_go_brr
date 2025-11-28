#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <fstream>
#include "src/utils.cuh"

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <kernel_num>\n", argv[0]);
        printf("  kernel_num: 0 (cuBLAS), 1-7 (custom kernels)\n");
        exit(EXIT_FAILURE);
    }

    int kernel_num = atoi(argv[1]);
    if (kernel_num < 0 || kernel_num > 7) {
        printf("Error: kernel_num must be between 0 and 7\n");
        exit(EXIT_FAILURE);
    }

    printf("========================================\n");
    printf("SGEMM Benchmark - Kernel %d\n", kernel_num);
    printf("========================================\n\n");

    // Initialize cuBLAS
    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        printf("Failed to create cuBLAS handle\n");
        exit(EXIT_FAILURE);
    }

    // Display GPU info
    CudaDeviceInfo();
    printf("\n");

    // GEMM parameters
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Matrix sizes to test (24 sizes from 256 to 6144)
    const int TESTNUM = 24;
    const int MAX_SIZE = 6144;
    
    // Allocate maximum size buffers
    const int max_elements = MAX_SIZE * MAX_SIZE;
    
    // Host memory
    float *h_A = (float *)malloc(max_elements * sizeof(float));
    float *h_B = (float *)malloc(max_elements * sizeof(float));
    float *h_C = (float *)malloc(max_elements * sizeof(float));
    float *h_C_ref = (float *)malloc(max_elements * sizeof(float));
    
    // Device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, max_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, max_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, max_elements * sizeof(float)));

    // Initialize random matrices once (will use subsets for different sizes)
    randomize_matrix(h_A, max_elements);
    randomize_matrix(h_B, max_elements);

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, max_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, max_elements * sizeof(float), cudaMemcpyHostToDevice));

    // CUDA events for timing
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));

    // Output file for results
    char filename[100];
    sprintf(filename, "test/test_kernel_%d.txt", kernel_num);
    std::ofstream output_file(filename);
    
    output_file << "Size,Time(ms),GFLOPS" << std::endl;
    
    printf("%-8s %-12s %-12s %-10s\n", "Size", "Time(ms)", "GFLOPS", "Status");
    printf("%-8s %-12s %-12s %-10s\n", "--------", "------------", "------------", "----------");

    // Test different matrix sizes
    for (int i = 0; i < TESTNUM; i++) {
        int M = 256 + i * 256;  // 256, 512, 768, ..., 6144
        int N = M;
        int K = M;

        // Reset output matrix
        CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));

        // === Correctness Test (skip for cuBLAS itself) ===
        bool correct = true;
        if (kernel_num != 0) {
            // Run cuBLAS reference
            CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));
            test_kernel(0, M, N, K, alpha, d_A, d_B, beta, d_C, handle);
            CUDA_CHECK(cudaMemcpy(h_C_ref, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

            // Run test kernel
            CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));
            test_kernel(kernel_num, M, N, K, alpha, d_A, d_B, beta, d_C, handle);
            CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

            // Verify
            correct = verify_matrix(h_C_ref, h_C, M * N);
        }

        // === Performance Test ===
        const int repeat_times = 10;
        float total_time = 0.0f;

        for (int r = 0; r < repeat_times; r++) {
            CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));
            
            CUDA_CHECK(cudaEventRecord(start));
            test_kernel(kernel_num, M, N, K, alpha, d_A, d_B, beta, d_C, handle);
            CUDA_CHECK(cudaEventRecord(end));
            CUDA_CHECK(cudaEventSynchronize(end));

            float elapsed_time;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, end));
            total_time += elapsed_time;
        }

        float avg_time = total_time / repeat_times;
        float gflops = (2.0f * M * N * K * 1e-9) / (avg_time * 1e-3);

        // Output results
        const char* status = correct ? "PASS" : "FAIL";
        printf("%-8d %-12.4f %-12.2f %-10s\n", M, avg_time, gflops, status);
        output_file << M << "," << avg_time << "," << gflops << std::endl;
    }

    printf("\n");
    printf("Results saved to: %s\n", filename);

    // Cleanup
    output_file.close();
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cublasDestroy(handle);

    return 0;
}
