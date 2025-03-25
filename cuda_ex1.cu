#include <stdio.h>

__device__ void device1()
{
	printf("hello world");
}

__global__ void kernel()
{
        device1();
}

void subHostFunction(){
        kernel<<<2,2>>>();
        cudaDeviceSynchronize();
}

int main()
{
        subHostFunction();
        return 0;
}
