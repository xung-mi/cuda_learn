#include <cuda_runtime.h>
#include <iostream>

int main() {
    int *d_ptr;
    cudaError_t err = cudaMalloc((void**)&d_ptr, -1); // Cấp phát kích thước âm -> lỗi

    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }

    return 0;
}

