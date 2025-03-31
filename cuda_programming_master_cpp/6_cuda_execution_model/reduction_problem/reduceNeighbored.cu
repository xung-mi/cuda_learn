#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdio.h>

#define NX (1 << 12) // Giảm kích thước để tránh lỗi bộ nhớ
#define NY (1 << 12)

__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n) {
  // set thread ID
  unsigned int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int *idata = g_idata + blockIdx.x * blockDim.x;
  if (idx >= n)
    return;

  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if ((tid % (2 * stride)) == 0) {
      idata[tid] += idata[tid + stride];
    }

    __syncthreads();
  }

  if (tid == 0)
    g_odata[blockIdx.x] = idata[0];
}

int recursiveReduce(int *data, int const size) {
  if (size == 1)
    return data[0];
  int const stride = size / 2;

  for (int i = 0; i < stride; i++) {
    data[i] += data[i + stride];
  }

  return recursiveReduce(data, stride);
}

int main(int argc, char **argv) {
  int size = 1 << 24;
  printf(" with array size %d ", size);

  int blocksize = 512;
  if (argc > 1) {
    blocksize = atoi(argv[1]);
  }

  dim3 block(blocksize, 1);
  dim3 grid((size + block.x - 1) / block.x, 1);
  printf("grid %d block %d\n", grid.x, block.x);

  // allocate host memory
  size_t bytes = size * sizeof(int);
  int *h_idata = (int *)malloc(bytes);
  int *h_odata = (int *)malloc(grid.x * sizeof(int));
  int *tmp = (int *)malloc(bytes);

  // initialize the array
  for (int i = 0; i < size; i++) {
    // mask off high 2 bytes to force max number to 255
    h_idata[i] = (int)(rand() & 0xFF);
  }
  memcpy(tmp, h_idata, bytes);

  clock_t iStart, iElaps;
  int gpu_sum = 0;

  // allocate device memory
  int *d_idata = NULL;
  int *d_odata = NULL;
  cudaMalloc((void **)&d_idata, bytes);
  cudaMalloc((void **)&d_odata, grid.x * sizeof(int));

  // cpu reduction
  iStart = clock();
  int cpu_sum = recursiveReduce(tmp, size);
  iElaps = (long int)(clock() - iStart);
  printf("cpu reduce elapsed %ld ms cpu_sum: %d\n", iElaps, cpu_sum);

  // kernel 1: reduceNeighbored
  cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  iStart = clock();
  reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
  cudaDeviceSynchronize();
  iElaps = (long int)(clock() - iStart);
  cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x; i++)
    gpu_sum += h_odata[i];
  printf("gpu Neighbored elapsed %ld ms gpu_sum: %d <<<grid %d block %d>>>\n",
         iElaps, gpu_sum, grid.x, block.x);

  free(h_idata);
  free(h_odata);
  cudaFree(d_idata);
  cudaFree(d_odata);
  cudaDeviceReset();

  bool bResult = (gpu_sum == cpu_sum);
  if (!bResult)
    printf("Test failed!\n");
  else {
    printf("cpu_sum = gpu_sum\n");
  }
  return EXIT_SUCCESS;
}
