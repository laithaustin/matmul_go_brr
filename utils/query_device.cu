#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceId;
    cudaGetDevice(&deviceId);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    printf("Device ID: %d\n", deviceId);
    printf("  Name: %s\n", props.name);
    printf("  Compute Capability: %d.%d\n", props.major, props.minor);
    printf("  Number of SMs: %d\n", props.multiProcessorCount);
    printf("  Total Global Memory: %.2f GB\n",
           props.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
    printf("\n");
    printf("  Shared Memory per SM: %zu bytes (%.2f KB)\n",
           props.sharedMemPerMultiprocessor,
           props.sharedMemPerMultiprocessor / 1024.0);
    printf("  Shared Memory per Block: %zu bytes (%.2f KB)\n",
           props.sharedMemPerBlock,
           props.sharedMemPerBlock / 1024.0);
    printf("\n");
    printf("  Max Threads per Block: %d\n", props.maxThreadsPerBlock);
    printf("  Max Threads per SM: %d\n", props.maxThreadsPerMultiProcessor);
    printf("  Warp Size: %d\n", props.warpSize);
    printf("  Registers per Block: %d\n", props.regsPerBlock);
    printf("  Registers per SM: %d\n", props.regsPerMultiprocessor);

    return 0;
}
