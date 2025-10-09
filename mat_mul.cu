#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// __global__ keyword declares a GPU kernel
__global__ void naive_kernel(int M, int N, int K, float alpha,
                            const float *A, const float *B,
                            float beta, float *C) {
    int BLOCKSIZE = 32;
    
    const int row = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int col = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
    
    if (row < M && col < N) { // guard in case some threads are outside the range
        float tmp = 0.0;
        // compute dot product
        for (int i = 0; i < K; ++i) {
            tmp += A[row * K + i] * B[i * N + col];
        }
        // GEMM: C = alpha * A @ B + beta * C
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}

void initialize_matrix(float* mat, int rows, int cols, float value) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = value;
    }
}

void initialize_matrix_random(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void print_matrix(const float* mat, int rows, int cols, const char* name) {
    std::cout << name << ":\n";
    for (int i = 0; i < std::min(rows, 5); i++) {
        for (int j = 0; j < std::min(cols, 5); j++) {
            std::cout << mat[i * cols + j] << " ";
        }
        if (cols > 5) std::cout << "...";
        std::cout << "\n";
    }
    if (rows > 5) std::cout << "...\n";
    std::cout << "\n";
}

bool verify_result(const float* C_gpu, int M, int N, int K, 
                   const float* A, const float* B, 
                   float alpha, float beta, const float* C_original) {
    float max_error = 0.0f;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float expected = beta * C_original[i * N + j];
            for (int k = 0; k < K; k++) {
                expected += alpha * A[i * K + k] * B[k * N + j];
            }
            float error = std::abs(C_gpu[i * N + j] - expected);
            max_error = std::max(max_error, error);
        }
    }
    std::cout << "Max error: " << max_error << std::endl;
    return max_error < 1e-3;
}

int main() {
    // Matrix dimensions
    const int M = 512;  // rows of A and C
    const int N = 512;  // cols of B and C
    const int K = 512;  // cols of A and rows of B
    
    // GEMM parameters
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Allocate host memory
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    float *h_C_result = (float*)malloc(size_C);
    
    // Initialize matrices
    srand(42);
    initialize_matrix_random(h_A, M, K);
    initialize_matrix_random(h_B, K, N);
    initialize_matrix(h_C, M, N, 0.0f);
    
    // Keep a copy of original C for verification
    for (int i = 0; i < M * N; i++) {
        h_C_result[i] = h_C[i];
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));
    
    // Launch kernel
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
    dim3 blockDim(32 * 32);
    
    std::cout << "Launching kernel with:" << std::endl;
    std::cout << "  Grid: (" << gridDim.x << ", " << gridDim.y << ", " << gridDim.z << ")" << std::endl;
    std::cout << "  Block: (" << blockDim.x << ", " << blockDim.y << ", " << blockDim.z << ")" << std::endl;
    std::cout << "  Matrix sizes: A(" << M << "x" << K << "), B(" << K << "x" << N << "), C(" << M << "x" << N << ")" << std::endl;
    
    // Create events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Launch and time kernel
    CUDA_CHECK(cudaEventRecord(start));
    naive_kernel<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    CUDA_CHECK(cudaEventRecord(stop));
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Calculate elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C_result, d_C, size_C, cudaMemcpyDeviceToHost));
    
    // Print results
    std::cout << "\nKernel execution time: " << milliseconds << " ms" << std::endl;
    
    // Calculate performance metrics
    double flops = 2.0 * M * N * K;  // multiply-add counts as 2 ops
    double gflops = (flops / milliseconds) / 1e6;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    
    // Verify result
    std::cout << "\nVerifying result..." << std::endl;
    bool correct = verify_result(h_C_result, M, N, K, h_A, h_B, alpha, beta, h_C);
    std::cout << (correct ? "Result is CORRECT!" : "Result is INCORRECT!") << std::endl;
    
    // Print sample output
    print_matrix(h_C_result, M, N, "C (result, first 5x5)");
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_result);
    
    std::cout << "\nDone!" << std::endl;
    
    return 0;
}
