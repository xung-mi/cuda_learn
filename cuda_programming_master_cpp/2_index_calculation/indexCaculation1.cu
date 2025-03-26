
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//xác định chỉ số duy nhất cho mỗi thread
__global__ void unique_idx_calc_threadId(int *input){
    int tid = threadIdx.x;
    printf("threadIdx: %d, value : %d\n", 
              tid, input[tid]);
}

int main() {
    int array_size = 8;
    int array_byte_size = sizeof(int) * array_size;
    int h_data[] = {1,2,3,4,5,6,7,8};

    int *d_data;
    cudaMalloc(&d_data, array_byte_size);
    cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

    dim3 block(8);
    dim3 grid(1);

    unique_idx_calc_threadId<<<grid, block>>>(d_data);
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}
