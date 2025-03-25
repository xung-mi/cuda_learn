#include <stdio.h>
#include <stdlib.h>

#define N 100

__global__ void vectorAdd(int *a, int *b, int *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    int *h_a, *h_b, *h_c; // Vector trên CPU
    int *d_a, *d_b, *d_c; // Vector trên GPU

    // Khởi tạo vector trên CPU
    h_a = (int *)malloc(N * sizeof(int));
    h_b = (int *)malloc(N * sizeof(int));
    h_c = (int *)malloc(N * sizeof(int));

    // Khởi tạo vector ngẫu nhiên
    for (int i = 0; i < N; i++) {
        h_a[i] = rand() % 10;
        h_b[i] = rand() % 10;
    }

    // Khởi tạo vector trên GPU
    cudaMalloc((void **)&d_a, N * sizeof(int));
    cudaMalloc((void **)&d_b, N * sizeof(int));
    cudaMalloc((void **)&d_c, N * sizeof(int));

    // Sao chép dữ liệu từ CPU sang GPU
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

  

    // Gọi kernel CUDA để thực hiện phép cộng
    vectorAdd<<<2, 50>>>(d_a, d_b, d_c);

    // Sao chép kết quả từ GPU sang CPU
    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // In kết quả
    for (int i = 0; i < N; i++) {
    printf("h_a[%d] %d + h_b[%d] %d = %d\n", i, h_a[i], i, h_b[i], h_c[i] );
}

    // Giải phóng bộ nhớ
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
