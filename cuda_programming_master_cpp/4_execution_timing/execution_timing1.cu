#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

__global__ void simpleKernel(clock_t* d_time) {
    clock_t start, end;
    start = clock();
    for (int i = 0; i < 10000000000000; i++);
    end = clock();
    d_time[threadIdx.x] = (end - start)/CLOCKS_PER_SEC;
}

int main() {
    const int THREADS = 256;
    clock_t h_time[THREADS], * d_time;

    cudaMalloc((void**)&d_time, THREADS * sizeof(clock_t));
    simpleKernel<<<1, THREADS>>>(d_time);
    cudaDeviceSynchronize();
    cudaMemcpy(h_time, d_time, THREADS * sizeof(clock_t), cudaMemcpyDeviceToHost);
    cudaFree(d_time);

    for (int i = 0; i < 10; i++) {
        std::cout << "Thread " << i << " execution time (clock cycles): " << h_time[i] << std::endl;
    }

    return 0;
}

