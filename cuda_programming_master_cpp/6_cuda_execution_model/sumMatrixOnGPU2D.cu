#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define NX (1 << 12) // Giảm kích thước để tránh lỗi bộ nhớ
#define NY (1 << 12)

__global__ void sumMatrixOnGPU2D(float *A, float *B, float *C, int nx, int ny) {
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * nx + ix;
    if (ix < nx && iy < ny) {
        C[idx] = A[idx] + B[idx];
    }
}

void initMatrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = static_cast<float>(rand() % 100);
    }
}

int main(int argc, char *argv[]) {
    int nx = NX, ny = NY;
    int dimx = 16, dimy = 16;

    if (argc > 2) {
        dimx = atoi(argv[1]);
        dimy = atoi(argv[2]);
    }

    int size = nx * ny;
    size_t bytes = size * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    srand(time(0));
    initMatrix(h_A, size);
    initMatrix(h_B, size);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // Thêm sự kiện đo thời gian
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sumMatrixOnGPU2D<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // In kết quả
    std::cout << "sumMatrixOnGPU2D <<< (" << grid.x << "," << grid.y << "), ("
              << block.x << "," << block.y << ") >>> elapsed " << milliseconds << " ms\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

