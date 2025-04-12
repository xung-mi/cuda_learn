/*
 * File        : hello_world.cu
 * Author      : XungVV
 * Email       : xungmi909@gmail.com
 * GitHub      : https://github.com/xung-mi
 * Created     : 2025-04-10
 * Updated     : 2025-04-10
 * Description : build command with these flags:
        nvcc hello_world.cu -o a -lcudadevrt -rdc=true
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdio.h>

__global__ void nestedHelloWorld(int const iSize, int iDepth)
{
    int tid = threadIdx.x;
    int blockId = blockIdx.x;
    printf("Hello World from thread %d, Depth: %d, blockIdx : %d\n", tid, iDepth, blockId);

    // condition of thread num to stop recursive execution
    if (iSize == 1)
        return;

    int nthreads = iSize >> 1;

    if (tid == 0 && nthreads > 0)
    {
        nestedHelloWorld<<<1, nthreads>>>(nthreads, ++iDepth);
        printf("-------> nested execution depth: %d\n", iDepth);
    }
}

__global__ void nestedHelloWorldWith2Block(int const iSize, int iDepth)
{
    int tid = threadIdx.x;
    int blockId = blockIdx.x;
    printf("Hello World from thread %d, Depth: %d, blockIdx : %d\n", tid, iDepth, blockIdx.x);

    // condition of thread num to stop recursive execution
    if (iSize == 1)
        return;

    int nthreads = iSize >> 1;

    if (tid == 0 && blockId == 0 && nthreads > 0)
    {
        nestedHelloWorldWith2Block<<<2, nthreads / 2>>>(nthreads, ++iDepth);
        printf("-------> nested execution depth: %d\n", iDepth);
    }
}

/*
    - gpuRecursiveReduce() proves: a large amount of kernel invocation and synchronization is likely the
        main cause for poor kernel performance.
    - gpuRecursiveReduceNosync() improves disadvantage of gpuRecursiveReduce()
        by Removing all synchronization operations
    - gpuRecursiveReduce2() improves disadvantage of gpuRecursiveReduce() by reducing large number of
        invocation.


*/
__global__ void gpuRecursiveReduce(int *g_idata, int *g_odata,
                                   unsigned int isize)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;
    int *odata = &g_odata[blockIdx.x];
    // stop condition
    if (isize == 2 && tid == 0)
    {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }
    // nested invocation
    int istride = isize >> 1;
    if (istride > 1 && tid < istride)
    {
        // in place reduction
        idata[tid] += idata[tid + istride];
    }
    // sync at block level
    __syncthreads();
    // nested invocation to generate child grids
    if (tid == 0)
    {
        gpuRecursiveReduce<<<1, istride>>>(idata, odata, istride);
        // sync all child grids launched in this block
        cudaDeviceSynchronize();
    }
    // sync at block level again
    __syncthreads();
}

__global__ void gpuRecursiveReduceNosync(int *g_idata, int *g_odata,
                                         unsigned int isize)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;
    int *odata = &g_odata[blockIdx.x];
    // stop condition
    if (isize == 2 && tid == 0)
    {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }
    // nested invoke
    int istride = isize >> 1;
    if (istride > 1 && tid < istride)
    {
        idata[tid] += idata[tid + istride];
        if (tid == 0)
        {
            gpuRecursiveReduceNosync<<<1, istride>>>(idata, odata, istride);
        }
    }
}

__global__ void gpuRecursiveReduce2(int *g_idata, int *g_odata, int iStride,
                                    int const iDim)
{
    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * iDim;
    // stop condition
    if (iStride == 1 && threadIdx.x == 0)
    {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }
    // in place reduction
    idata[threadIdx.x] += idata[threadIdx.x + iStride];
    // nested invocation to generate child grids
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        gpuRecursiveReduce2<<<gridDim.x, iStride / 2>>>(
            g_idata, g_odata, iStride / 2, iDim);
    }
}

int main()
{
    int size = 16;
    int blockSize = 8;
    dim3 block(blockSize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);

    size_t bytes = size * sizeof(int);
    int *h_idata = (int *)malloc(bytes);

    for (int i = 0; i < size; i++)
    {
        h_idata[i] = (int)(rand() & 0xFF);
    }

    // allocate device memory
    int *d_idata = NULL;
    cudaMalloc(&d_idata, bytes);

    /*
    //nestedHelloWorldWith2Block
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    nestedHelloWorld<<<grid, block>>>(size, 0);
    cudaDeviceSynchronize();
    */

    /*
    // nestedHelloWorldWith2Block
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    nestedHelloWorldWith2Block<<<grid, block>>>(size, 0);
    cudaDeviceSynchronize();
    */

    free(h_idata);
    cudaFree(d_idata);
    cudaDeviceReset();
}