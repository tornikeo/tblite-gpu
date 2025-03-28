#include <cstdio>

// Declare the function from the CUDA module
extern "C" void call_hello_kernel_();

int main() {
    printf("Running CUDA kernel test...\n");
    call_hello_kernel_();
    printf("CUDA kernel test completed.\n");
    return 0;
}
