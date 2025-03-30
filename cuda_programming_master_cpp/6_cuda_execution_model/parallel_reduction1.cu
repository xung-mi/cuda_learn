#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024
#define THREADS_PER_BLOCK 512

__global__ void parallelReduction(int *input, int *output) {
    __shared__ int shared_data[THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + tid;

    shared_data[tid] = (global_id < N) ? input[global_id] : 0;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();   
    }

    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

int main() {
    int *h_input, *h_output;
    int *d_input, *d_output;
    int num_blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    h_input = (int*)malloc(N * sizeof(int));
    h_output = (int*)malloc(num_blocks * sizeof(int));

    for (int i = 0; i < N; i++) {
        h_input[i] = 1;  
    }

    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_output, num_blocks * sizeof(int));

    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    parallelReduction<<<num_blocks, THREADS_PER_BLOCK>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);

    int total_sum = 0;
    for (int i = 0; i < num_blocks; i++) {
        total_sum += h_output[i];
    }

    printf("Total sum: %d\n", total_sum);

    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

