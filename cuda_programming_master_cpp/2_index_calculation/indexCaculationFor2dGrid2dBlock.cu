
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#
//xác định chỉ số duy nhất cho mỗi thread
__global__ void unique_idx_calc_threadId(int *input){
    int num_threads_in_a_block = blockDim.x * blockDim.y;
    int block_offset = num_threads_in_a_block * blockIdx.x;

    int num_threads_in_a_row = num_threads_in_a_block * gridDim.x;
    int row_offset =  num_threads_in_a_row * blockIdx.y;

    int tid = threadIdx.y*blockDim.x + threadIdx.x;
    int gid = row_offset + block_offset + tid;
    printf("gid: %d, value : %d\n", gid , input[gid]);
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

