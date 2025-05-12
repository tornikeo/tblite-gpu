#include <cstdio>

// Declare the function from the CUDA module
// extern "C" void call_hello_kernel_();

int main() {
    printf("Running CUDA kernel test (this does nothing). At %s %i\n", __FILE__, __LINE__);
    // call_hello_kernel_();
    // printf("CUDA kernel test completed.\n");
    return 0;
}
