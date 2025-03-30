#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void print_hello(){
    printf("Hello Cu\n");
}

int main(){
    print_hello<<<1,10>>>();
    cudaDeviceSynchronize();
    // cudaDeviceReset();
    return 0;
}
