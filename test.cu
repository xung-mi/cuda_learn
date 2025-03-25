
#include <stdio.h>
#include <cuda_runtime.h>
#define N 10
__global__ void a(){

        printf("%d\n", threadIdx.x *threadIdx.x );
  
}
int main(){
    a <<<1, N>>>();
    cudaDeviceSynchronize();
   
    return 0;
}

