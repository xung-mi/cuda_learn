#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "No CUDA-compatible device found!" << std::endl;
        return 1;
    }

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
        std::cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max Threads per Dimension: (" 
                  << deviceProp.maxThreadsDim[0] << ", " 
                  << deviceProp.maxThreadsDim[1] << ", " 
                  << deviceProp.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "Max Grid Size: (" 
                  << deviceProp.maxGridSize[0] << ", " 
                  << deviceProp.maxGridSize[1] << ", " 
                  << deviceProp.maxGridSize[2] << ")" << std::endl;
        std::cout << "Clock Rate: " << deviceProp.clockRate / 1000 << " MHz" << std::endl;
        std::cout << "Shared Memory per Block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
        std::cout << "============================" << std::endl;
    }

    return 0;
}

