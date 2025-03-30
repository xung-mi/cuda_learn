#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CUDA_CHECK(call) { 
    cudaError_t err = call;
    if (err != cudaSuccess) { 
        printf("Error: %s:%d, ", __FILE__, __LINE__); 
        printf("code: %d, reason: %s\n", err, cudaGetErrorString(err)); 
        exit(1); 
    } 
}

__global__ void print_hello()
{
    printf("Hello Cu\n");
}

int main()
{
    print_hello<<<1, 10>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    // cudaDeviceReset();
    return 0;
}



