
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#
//xác định chỉ số duy nhất cho mỗi thread
__global__ void unique_idx_calc_threadId(int *input){
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int nx = blockDim.x * gridDim.x;
    int idx = iy*nx + ix;

    printf("idx: %d, value : %d\n", idx , input[idx]);
}

int main() {
    int array_size = 16;
    int array_byte_size = sizeof(int) * array_size;
    int h_data[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};

    int *d_data;
    cudaMalloc(&d_data, array_byte_size);
    cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

    dim3 block(2,2);
    dim3 grid(2,2);

    unique_idx_calc_threadId<<<grid, block>>>(d_data);
    cudaDeviceSynchronize();
   
    cudaDeviceReset();
    return 0;
}

