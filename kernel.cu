#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_kernel() {
    printf("Hello World from CUDA kernel!\n");
}

extern "C" void call_hello_kernel() {
    hello_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();  // Wait for the kernel to finish.
}

int main() {
    // Call the wrapper function for testing purposes.
    call_hello_kernel();
    return 0;
}