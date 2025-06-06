
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void sum_array_gpu(int *a, int *b, int *c, int size)
{
       int gid = blockIdx.x * blockDim.x + threadIdx.x;

       if (gid < size)
       {
              c[gid] = a[gid] + b[gid];
       }
}

void sum_array_cpu(int *a, int *b, int *c, int size)
{
       for (int i = 0; i < 1000; i++)
       {
              c[i] = a[i] + b[i];
       }
}

bool compareArrays(const int *arr1, const int *arr2, size_t size)
{
       for (size_t i = 0; i < size; i++)
       {
              if (arr1[i] != arr2[i])
              {
                     return false;
              }
       }
       return true;
}

int main()
{
       int size = 1000;
       int block_size = 128;

       int NO_BYTES = size * sizeof(int);

       int *h_a, *h_b, *gpu_result, *h_c;

       h_a = (int *)malloc(NO_BYTES);
       h_b = (int *)malloc(NO_BYTES);
       h_c = (int *)malloc(NO_BYTES);

       gpu_result = (int *)malloc(NO_BYTES);

       for (int i = 0; i < size; i++)
       {
              h_a[i] = i;
              h_b[i] = 2 * i;
       }

       sum_array_cpu(h_a, h_b, h_c, size);

       int *d_a, *d_b, *d_c;
       cudaMalloc((int **)&d_a, NO_BYTES);
       cudaMalloc((int **)&d_b, NO_BYTES);
       cudaMalloc((int **)&d_c, NO_BYTES);

       cudaMemcpy(d_a, h_a, NO_BYTES, cudaMemcpyHostToDevice);
       cudaMemcpy(d_b, h_b, NO_BYTES, cudaMemcpyHostToDevice);

       dim3 block(block_size);
       dim3 grid((size / block.x) + 1);

       sum_array_gpu<<<grid, block>>>(d_a, d_b, d_c, size);
       cudaDeviceSynchronize(); // block the host execution until kernel function finish

       cudaMemcpy(gpu_result, d_c, NO_BYTES, cudaMemcpyDeviceToHost);

       // compare arrays
       if (compareArrays(h_c, gpu_result, size)){
              printf("Same result!\n");
       } else {
              printf("Different result\n");
       }

       cudaFree(d_c);
       cudaFree(d_b);
       cudaFree(d_a);

       free(gpu_result);
       free(h_a);
       free(h_b);
       return 0;
}
