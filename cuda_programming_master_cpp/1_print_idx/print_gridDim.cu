
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void print_threadIds(){
       printf("threadIdx.x: %d, threadIdx.y : %d, threadIdx.z : %d\n", 
              threadIdx.x, threadIdx.y, threadIdx.z);
}

__global__ void print_blockIds(){
       printf("blockIdx.x: %d, blockIdx.y : %d, blockIdx.z : %d\n", 
              blockIdx.x, blockIdx.y, blockIdx.z);
}

__global__ void print_blockDim(){
       printf("blockDim.x: %d, blockDim.y : %d, blockDim.z : %d\n", 
              blockDim.x, blockDim.y, blockDim.z);
}

__global__ void print_gridDim(){
       printf("gridDim.x: %d, gridDim.y : %d, gridDim.z : %d\n", 
              gridDim.x, gridDim.y, gridDim.z);
}

int main() {
    int nx, ny;
    nx = 8;
    ny = 8;

    dim3 block(4,4);
    dim3 grid(nx/block.x, ny/block.y);

    print_gridDim<<<grid, block>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}
