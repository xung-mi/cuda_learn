#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdio.h>

#define NX (1 << 16) // Giảm kích thước để tránh lỗi bộ nhớ
#define NY (1 << 16)

__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n)
{
  // set thread ID
  unsigned int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int *idata = g_idata + blockIdx.x * blockDim.x;
  if (idx >= n)
    return;

  for (int stride = 1; stride < blockDim.x; stride *= 2)
  {
    if ((tid % (2 * stride)) == 0)
    {
      idata[tid] += idata[tid + stride];
    }

    __syncthreads();
  }

  if (tid == 0)
    g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceNeighboredLess(int *g_idata, int *g_odata,
                                     unsigned int n)
{
  // set thread ID
  unsigned int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int *idata = g_idata + blockIdx.x * blockDim.x;
  if (idx >= n)
    return;

  for (int stride = 1; stride < blockDim.x; stride *= 2)
  {
    int index = 2 * stride * tid;
    if (index < blockDim.x)
    {
      idata[index] += idata[index + stride];
    }

    __syncthreads();
  }

  if (tid == 0)
    g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n)
{
  // set thread ID
  unsigned int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int *idata = g_idata + blockIdx.x * blockDim.x;
  if (idx >= n)
    return;

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
  {
    if (tid < stride)
    {
      idata[tid] += idata[tid + stride];
    }

    __syncthreads();
  }

  if (tid == 0)
    g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling2(int *g_idata, int *g_odata, unsigned int n)
{
  // set thread ID
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  // convert global data pointer to the local pointer of this block
  int *idata = g_idata + blockIdx.x * blockDim.x * 2;
  // unrolling 2 data blocks
  if (idx + blockDim.x < n)
    g_idata[idx] += g_idata[idx + blockDim.x];
  __syncthreads();
  // in-place reduction in global memory
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
  {
    if (tid < stride)
    {
      idata[tid] += idata[tid + stride];
    }
    // synchronize within threadblock
    __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0)
    g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling4(int *g_idata, int *g_odata, unsigned int n)
{
  // set thread ID
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;
  // convert global data pointer to the local pointer of this block
  int *idata = g_idata + blockIdx.x * blockDim.x * 4;
  // unrolling4 data blocks
  if (idx + 3 * blockDim.x < n)
    g_idata[idx] += g_idata[idx + blockDim.x] + g_idata[idx + 2 * blockDim.x] + g_idata[idx + 3 * blockDim.x];
  __syncthreads();
  // in-place reduction in global memory
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
  {
    if (tid < stride)
    {
      idata[tid] += idata[tid + stride];
    }
    // synchronize within threadblock
    __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0)
    g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceCompleteUnrollWarp8(int *g_idata, int *g_odata, int n)
{
  int tid = threadIdx.x;
  // global index of data element in data block
  int idx = blockIdx.x * blockDim.x * 8 + tid;

  // convert global data pointer to the local pointer of this block
  int *idata = g_idata + blockIdx.x * blockDim.x * 8;

  // unrolling 8
  if (idx + 7 * blockDim.x < n)
  {
    int a1 = g_idata[idx];
    int a2 = g_idata[idx + blockDim.x];
    int a3 = g_idata[idx + 2 * blockDim.x];
    int a4 = g_idata[idx + 3 * blockDim.x];
    int b1 = g_idata[idx + 4 * blockDim.x];
    int b2 = g_idata[idx + 5 * blockDim.x];
    int b3 = g_idata[idx + 6 * blockDim.x];
    int b4 = g_idata[idx + 7 * blockDim.x];
    g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
  }
  __syncthreads();

  // in-place reduction and complete unroll
  if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];
  __syncthreads();
  if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];
  __syncthreads();
  if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];
  __syncthreads();
  if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];
  __syncthreads();

  // unrolling warp
  if (tid < 32)
  {
    volatile int *vsmem = idata;
    vsmem[tid] += vsmem[tid + 32];
    vsmem[tid] += vsmem[tid + 16];
    vsmem[tid] += vsmem[tid + 8];
    vsmem[tid] += vsmem[tid + 4];
    vsmem[tid] += vsmem[tid + 2];
    vsmem[tid] += vsmem[tid + 1];
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

template <unsigned int iBlockSize>
__global__ void reduceCompleteUnrollTemplate(int *g_idata, int *g_odata, int n)
{
  int tid = threadIdx.x;
  // global index of data element in data block
  int idx = blockIdx.x * blockDim.x * 8 + tid;

  // convert global data pointer to the local pointer of this block
  int *idata = g_idata + blockIdx.x * blockDim.x * 8;

  // unrolling 8
  if (idx + 7 * blockDim.x < n)
  {
    int a1 = g_idata[idx];
    int a2 = g_idata[idx + blockDim.x];
    int a3 = g_idata[idx + 2 * blockDim.x];
    int a4 = g_idata[idx + 3 * blockDim.x];
    int b1 = g_idata[idx + 4 * blockDim.x];
    int b2 = g_idata[idx + 5 * blockDim.x];
    int b3 = g_idata[idx + 6 * blockDim.x];
    int b4 = g_idata[idx + 7 * blockDim.x];
    g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
  }
  __syncthreads();

  // in-place reduction and complete unroll
  if (iBlockSize >= 1024 && tid < 512) idata[tid] += idata[tid + 512];
  __syncthreads();
  if (iBlockSize >= 512 && tid < 256)  idata[tid] += idata[tid + 256];
  __syncthreads();
  if (iBlockSize >= 256 && tid < 128)  idata[tid] += idata[tid + 128];
  __syncthreads();
  if (iBlockSize >= 128 && tid < 64)   idata[tid] += idata[tid + 64];
  __syncthreads();

  // unrolling warp
  if (tid < 32)
  {
    volatile int *vsmem = idata;
    vsmem[tid] += vsmem[tid + 32];
    vsmem[tid] += vsmem[tid + 16];
    vsmem[tid] += vsmem[tid + 8];
    vsmem[tid] += vsmem[tid + 4];
    vsmem[tid] += vsmem[tid + 2];
    vsmem[tid] += vsmem[tid + 1];
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

int recursiveReduce(int *data, int const size)
{
  if (size == 1)
    return data[0];
  int const stride = size / 2;

  for (int i = 0; i < stride; i++)
  {
    data[i] += data[i + stride];
  }

  return recursiveReduce(data, stride);
}

int main(int argc, char **argv)
{
  int size = 1 << 24;
  printf("With array size %d ", size);

  int blocksize = 512;
  if (argc > 1)
  {
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
  for (int i = 0; i < size; i++)
  {
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

  // kernel 2: reduceNeighboredLess
  cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  iStart = clock();
  reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, size);
  cudaDeviceSynchronize();
  iElaps = (long int)(clock() - iStart);
  cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x; i++)
    gpu_sum += h_odata[i];
  printf("gpu NeighboredL elapsed %ld ms gpu_sum: %d <<<grid %d block %d>>>\n",
         iElaps, gpu_sum, grid.x, block.x);

  // kernel 3: reduceInterLeaved
  cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  iStart = clock();
  reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
  cudaDeviceSynchronize();
  iElaps = (long int)(clock() - iStart);
  cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x; i++)
    gpu_sum += h_odata[i];
  printf("gpu Interleaved elapsed %ld ms gpu_sum: %d <<<grid %d block %d>>>\n",
         iElaps, gpu_sum, grid.x, block.x);

  // kernel 4: reduceUnrolling2
  cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  iStart = clock();
  reduceUnrolling2<<<grid.x / 2, block>>>(d_idata, d_odata, size);
  cudaDeviceSynchronize();
  iElaps = (long int)(clock() - iStart);
  cudaMemcpy(h_odata, d_odata, grid.x / 2 * sizeof(int), cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x / 2; i++)
    gpu_sum += h_odata[i];
  printf("gpu unroll2 elapsed %ld ms gpu_sum: %d <<<grid %d block %d>>>\n",
         iElaps, gpu_sum, grid.x / 2, block.x);

  // kernel 5: reduceUnrolling4
  cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  iStart = clock();
  reduceUnrolling4<<<grid.x / 4, block>>>(d_idata, d_odata, size);
  cudaDeviceSynchronize();
  iElaps = (long int)(clock() - iStart);
  cudaMemcpy(h_odata, d_odata, grid.x / 4 * sizeof(int), cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x / 4; i++)
    gpu_sum += h_odata[i];
  printf("gpu unroll4 elapsed %ld ms gpu_sum: %d <<<grid %d block %d>>>\n",
         iElaps, gpu_sum, grid.x / 4, block.x);

  // kernel 5: reducecompleteUnrolling4
  cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  iStart = clock();
  reduceCompleteUnrollWarp8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
  cudaDeviceSynchronize();
  iElaps = (long int)(clock() - iStart);
  cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x / 8; i++)
    gpu_sum += h_odata[i];
  printf("gpu complete unroll elapsed %ld ms gpu_sum: %d <<<grid %d block %d>>>\n",
         iElaps, gpu_sum, grid.x / 8, block.x);

  // kernel 6: reducecompleteUnrolling4
  cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  iStart = clock();
  reduceCompleteUnrollTemplate<256><<<grid.x / 8, block>>>(d_idata, d_odata, size);
  cudaDeviceSynchronize();
  iElaps = (long int)(clock() - iStart);
  cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x / 8; i++)
    gpu_sum += h_odata[i];
  printf("gpu template complete unroll elapsed %ld ms gpu_sum: %d <<<grid %d block %d>>>\n",
         iElaps, gpu_sum, grid.x / 8, block.x);

  free(h_idata);
  free(h_odata);
  cudaFree(d_idata);
  cudaFree(d_odata);
  cudaDeviceReset();

  bool bResult = (gpu_sum == cpu_sum);
  if (!bResult)
    printf("Test failed!\n");
  else
  {
    printf("cpu_sum = gpu_sum\n");
  }
  return EXIT_SUCCESS;
}
