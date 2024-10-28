#include <cuda.h>
#include <stdio.h>

void displayDeviceProperties() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int device = 0; device < deviceCount; device++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        printf("Device %d: %s\n", device, deviceProp.name);
        printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Total global memory: %zu MB\n", deviceProp.totalGlobalMem / (1024 * 1024));
        printf("  Shared memory per block: %zu KB\n", deviceProp.sharedMemPerBlock / 1024);
        printf("  Registers per block: %d\n", deviceProp.regsPerBlock);
        printf("  Warp size: %d\n", deviceProp.warpSize);
        printf("  Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Max blocks in each dimension: %d x %d x %d\n", 
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("  Max threads in each dimension: %d x %d x %d\n", 
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("  Number of multiprocessors: %d\n", deviceProp.multiProcessorCount);
        printf("  Clock rate: %.2f MHz\n", deviceProp.clockRate / 1000.0);
        printf("  Memory clock rate: %.2f MHz\n", deviceProp.memoryClockRate / 1000.0);
        printf("  Memory bus width: %d bits\n", deviceProp.memoryBusWidth);
        printf("\n");
    }
}

int main() {
    displayDeviceProperties();
    return 0;
}