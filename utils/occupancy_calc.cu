#include <cuda_runtime.h>
#include <stdio.h>
#include "src/kernel.cuh"

int main() {
    // Get device properties
    int device;
    cudaDeviceProp props;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);

    printf("=== GPU Device Info ===\n");
    printf("Device: %s\n", props.name);
    printf("Compute Capability: %d.%d\n", props.major, props.minor);
    printf("SMs: %d\n", props.multiProcessorCount);
    printf("Max threads per SM: %d\n", props.maxThreadsPerMultiProcessor);
    printf("Max threads per block: %d\n", props.maxThreadsPerBlock);
    printf("Shared memory per block: %zu bytes\n", props.sharedMemPerBlock);
    printf("Shared memory per SM: %zu bytes\n", props.sharedMemPerMultiprocessor);
    printf("Registers per block: %d\n", props.regsPerBlock);
    printf("Registers per SM: %d\n", props.regsPerMultiprocessor);
    printf("Warp size: %d\n", props.warpSize);
    printf("\n");

    // Kernel 3 configuration
    const int BLOCKSIZE = 32;
    const int threadsPerBlock = 32 * 32;  // 1024 threads
    const size_t sharedMemPerBlock = 2 * BLOCKSIZE * BLOCKSIZE * sizeof(float);  // 8 KB

    printf("=== Kernel 3 Configuration ===\n");
    printf("Threads per block: %d\n", threadsPerBlock);
    printf("Shared memory per block: %zu bytes (%.2f KB)\n",
           sharedMemPerBlock, sharedMemPerBlock / 1024.0);
    printf("\n");

    // Calculate occupancy
    int maxActiveBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks,
        mysgemm_v3<BLOCKSIZE>,
        threadsPerBlock,
        sharedMemPerBlock
    );

    printf("=== Occupancy Statistics ===\n");
    printf("Max active blocks per SM: %d\n", maxActiveBlocks);
    printf("Max active warps per SM: %d\n", maxActiveBlocks * (threadsPerBlock / props.warpSize));
    printf("Max active threads per SM: %d\n", maxActiveBlocks * threadsPerBlock);

    float occupancy = (maxActiveBlocks * threadsPerBlock * 100.0f) / props.maxThreadsPerMultiProcessor;
    printf("Theoretical occupancy: %.2f%%\n", occupancy);
    printf("\n");

    // Resource usage analysis
    printf("=== Resource Limits ===\n");

    // Threads limit
    int maxBlocksFromThreads = props.maxThreadsPerMultiProcessor / threadsPerBlock;
    printf("Max blocks per SM (threads): %d\n", maxBlocksFromThreads);

    // Shared memory limit
    int maxBlocksFromSharedMem = props.sharedMemPerMultiprocessor / sharedMemPerBlock;
    printf("Max blocks per SM (shared mem): %d\n", maxBlocksFromSharedMem);

    // Determine limiting factor
    printf("\nLimiting factor: ");
    if (maxActiveBlocks == maxBlocksFromThreads && maxActiveBlocks < maxBlocksFromSharedMem) {
        printf("Threads per SM\n");
    } else if (maxActiveBlocks == maxBlocksFromSharedMem && maxActiveBlocks < maxBlocksFromThreads) {
        printf("Shared memory per SM\n");
    } else {
        printf("Multiple factors or register usage (check with nvcc --ptxas-options=-v)\n");
    }

    return 0;
}
