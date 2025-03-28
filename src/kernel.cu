#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void hello_kernel() {
    printf("Hello World from CUDA kernel!\n");
}

extern "C" void call_hello_kernel_() {
    hello_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();  // Wait for the kernel to finish.
}