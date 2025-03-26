
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//xác định chỉ số duy nhất cho mỗi thread
__global__ void unique_idx_calc_threadId(int *input){
    int rowoffset = gridDim.x * blockDim.x * blockIdx.y;
    int blockoffset = blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int gid = rowoffset + blockoffset + tid;
    printf("threadIdx: %d, blockIdx : %d, gid: %d, value : %d\n", 
              threadIdx.x, blockIdx.x, gid , input[gid]);
}

int main() {
    int array_size = 16;
    int array_byte_size = sizeof(int) * array_size;
    int h_data[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};

    int *d_data;
    cudaMalloc(&d_data, array_byte_size);
    cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

    dim3 block(4);
    dim3 grid(2,2);

    unique_idx_calc_threadId<<<grid, block>>>(d_data);
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}
