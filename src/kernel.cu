#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void hello_kernel() {
    printf("Hello World from CUDA kernel!\n");
}

extern "C" {
    void get_vec_(
            const double* xyz_iat,
            const double* xyz_jat,
            const double* trans,
            double vec[3]
        ) {
        printf("C: XYZ (atom iat): %f, %f, %f\n", 
               xyz_iat[0], 
               xyz_iat[1], 
               xyz_iat[2]);
        printf("C: XYZ (atom jat): %f, %f, %f\n", 
               xyz_jat[0], 
               xyz_jat[1], 
               xyz_jat[2]);
        printf("C: TRANS   : %f, %f, %f\n", 
               trans[0], 
               trans[1], 
               trans[2]);
        // for (size_t i = 0; i < 3; i++)
        // {
        //     printf("%d, ", vec[i]);
        // }
        // printf("\n");
        
        for (size_t k = 0; k < 3; ++k) {
            vec[k] = xyz_iat[k] - xyz_jat[k] - trans[k];
        }
        printf("C: Computed vec: %f, %f, %f\n", vec[0], vec[1], vec[2]);
        // printf("C: Result");
        // for (size_t i = 0; i < 3; i++)
        // {
        //     printf("%d, ", vec[i]);
        // }
        // printf("\n");
    }
}


extern "C" void call_hello_kernel_() {
    hello_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();  // Wait for the kernel to finish.
}