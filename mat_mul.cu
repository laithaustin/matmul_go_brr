#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>

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

// Idea for warp_tiling algo:
// 1. first we read from GMEM into SMEM.
// 2. Then we compute the partial dot product of our given tiles.
// 3. sync threads and then write to output matrix.
__global__ void warp_tiling(
    int M, int N, int K, float alpha,
    const float *A, const float *B,
    float beta, float *C 
) {
    const int TILE_SIZE = 32;
    
    // Shared memory for tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // blockDim.x = 1024 (32*32 threads)
    int tx = threadIdx.x % TILE_SIZE;  // column within tile (0-31)
    int ty = threadIdx.x / TILE_SIZE;  // row within tile (0-31)
    
    // Global row and column this thread computes
    int row = blockIdx.x * TILE_SIZE + ty;
    int col = blockIdx.y * TILE_SIZE + tx;
    
    // Accumulator for the dot product
    float tmp = 0.0f;
    
    // Loop over tiles along the K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        // Load tile of A into shared memory
        int aCol = t * TILE_SIZE + tx;
        if (row < M && aCol < K) {
            As[ty][tx] = A[row * K + aCol];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load tile of B into shared memory
        int bRow = t * TILE_SIZE + ty;
        if (bRow < K && col < N) {
            Bs[ty][tx] = B[bRow * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        // Synchronize to ensure all loads are complete
        __syncthreads();
        
        // Compute partial dot product using the loaded tiles
        // Each element in shared memory is reused TILE_SIZE times!
        for (int k = 0; k < TILE_SIZE; k++) {
            tmp += As[ty][k] * Bs[k][tx];
        }
        
        // Synchronize before loading next tiles
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < M && col < N) {
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}

__global__ void warp_tiling_improved(
    int M, int N, int K, float alpha,
    const float *A, const float *B,
    float beta, float *C 
) {
    const int TILE_SIZE = 32;
    
    // decrease bank conflicts
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
    
    // blockDim.x = 1024 (32*32 threads)
    int tx = threadIdx.x % TILE_SIZE;  // column within tile (0-31)
    int ty = threadIdx.x / TILE_SIZE;  // row within tile (0-31)
    
    // Global row and column this thread computes
    int row = blockIdx.x * TILE_SIZE + ty;
    int col = blockIdx.y * TILE_SIZE + tx;
    
    // Accumulator for the dot product
    float tmp = 0.0f;
    
    // Loop over tiles along the K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        // Load tile of A into shared memory
        int aCol = t * TILE_SIZE + tx;
        if (row < M && aCol < K) {
            As[ty][tx] = A[row * K + aCol];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load tile of B into shared memory
        int bRow = t * TILE_SIZE + ty;
        if (bRow < K && col < N) {
            Bs[ty][tx] = B[bRow * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        // Synchronize to ensure all loads are complete
        __syncthreads();
        
        // Compute partial dot product using the loaded tiles
        // Each element in shared memory is reused TILE_SIZE times!
        for (int k = 0; k < TILE_SIZE; k++) {
            tmp += As[ty][k] * Bs[k][tx];
        }
        
        // Synchronize before loading next tiles
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < M && col < N) {
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
    // Test multiple matrix sizes
    int sizes[] = {512, 1024, 2048, 4096};
    int num_sizes = 4;
    
    printf("=====================================================================================\n");
    printf("Matrix Multiplication Scaling Analysis\n");
    printf("=====================================================================================\n\n");
    
    printf("%-10s %12s %12s %12s %12s %12s\n", 
           "Size", "Naive (ms)", "Tiled (ms)", "Improved (ms)", "Speedup 1", "Speedup 2");
    printf("%-10s %12s %12s %12s %12s %12s\n", 
           "----------", "------------", "------------", "-------------", "------------", "------------");
    
    for (int s = 0; s < num_sizes; s++) {
        int M = sizes[s];
        int N = sizes[s];
        int K = sizes[s];
        
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
        
        // Initialize matrices
        srand(42);
        initialize_matrix_random(h_A, M, K);
        initialize_matrix_random(h_B, K, N);
        initialize_matrix(h_C, M, N, 0.0f);
        
        // Allocate device memory
        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, size_A));
        CUDA_CHECK(cudaMalloc(&d_B, size_B));
        CUDA_CHECK(cudaMalloc(&d_C, size_C));
        
        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
        
        // Kernel launch configuration
        dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
        dim3 blockDim(32 * 32);
        
        // Create events for timing
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        // ========== Test Naive Kernel ==========
        CUDA_CHECK(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));
        
        // Warmup
        naive_kernel<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Timed run (average of 3 runs)
        float naive_ms = 0;
        for (int i = 0; i < 3; i++) {
            CUDA_CHECK(cudaEventRecord(start));
            naive_kernel<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaDeviceSynchronize());
            
            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            naive_ms += ms;
        }
        naive_ms /= 3.0f;
        
        // ========== Test Tiled Kernel ==========
        CUDA_CHECK(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));
        
        // Warmup
        warp_tiling<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Timed run (average of 3 runs)
        float tiled_ms = 0;
        for (int i = 0; i < 3; i++) {
            CUDA_CHECK(cudaEventRecord(start));
            warp_tiling<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaDeviceSynchronize());
            
            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            tiled_ms += ms;
        }
        tiled_ms /= 3.0f;

        // ========== Test Improved Kernel ==========
        CUDA_CHECK(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));
        
        // Warmup
        warp_tiling_improved<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Timed run (average of 3 runs)
        float tiled2_ms = 0;
        for (int i = 0; i < 3; i++) {
            CUDA_CHECK(cudaEventRecord(start));
            warp_tiling_improved<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaDeviceSynchronize());
            
            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            tiled2_ms += ms;
        }
        tiled2_ms /= 3.0f;
        
        // Calculate metrics
        double naive_gflops = (2.0 * M * N * K / naive_ms) / 1e6;
        double tiled_gflops = (2.0 * M * N * K / tiled_ms) / 1e6;
        double tiled2_gflops = (2.0 * M * N * K / tiled2_ms) / 1e6;
        double speedup = naive_ms / tiled_ms;
        double speedup2 = naive_ms / tiled2_ms;
        
        // Print results in aligned columns
        printf("%-10d %12.3f %12.3f %12.3f %12.2fx %12.2fx\n", 
               M, naive_ms, tiled_ms, tiled2_ms, speedup, speedup2);
        printf("%-10s %12.1f %12.1f %12.1f GFLOPS\n", 
               "", naive_gflops, tiled_gflops, tiled2_gflops);
        printf("\n");
        
        // Cleanup
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        free(h_A);
        free(h_B);
        free(h_C);
    }
    
    return 0;
}