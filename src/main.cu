#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <math.h>

__device__ __constant__ double s3;
__device__ __constant__ double s3_4;
__device__ __constant__ double dtrafo[5][6];
__device__ __constant__ double d32;
__device__ __constant__ double s3_8;
__device__ __constant__ double s5_8;
__device__ __constant__ double s6;
__device__ __constant__ double s15;
__device__ __constant__ double s15_4;
__device__ __constant__ double s45;
__device__ __constant__ double s45_8;
__device__ __constant__ double ftrafo[7][10];
__device__ __constant__ double d38;
__device__ __constant__ double d34;
__device__ __constant__ double s5_16;
__device__ __constant__ double s10;
__device__ __constant__ double s10_8;
__device__ __constant__ double s35_4;
__device__ __constant__ double s35_8;
__device__ __constant__ double s35_64;
__device__ __constant__ double s45_4;
__device__ __constant__ double s315_8;
__device__ __constant__ double s315_16;
__device__ __constant__ double gtrafo[9][15];

__global__ 
void hello_kernel()
{
    printf("%i %i Says Hello!");
}

// Kernel to test the constants
__global__ void testKernel()
{
    printf("s3: %f, s3_4: %f, dtrafo[0][2]: %f, ftrafo[0][4]: %f, gtrafo[0][4]: %f\n",
           s3, s3_4, dtrafo[0][2], ftrafo[0][4], gtrafo[0][4]);
}

void init_constants()
{
    // Initialize the values for the constants
    double h_s3 = sqrt(3.0);
    double h_s3_4 = h_s3 * 0.5;
    double h_dtrafo[5][6] = {
        {0.0, 0.0, -0.5, 0.0, h_s3_4, 0.0},  // xx
        {0.0, 0.0, -0.5, 0.0, -h_s3_4, 0.0}, // yy
        {0.0, 0.0, 1.0, 0.0, 0.0, 0.0},      // zz
        {h_s3, 0.0, 0.0, 0.0, 0.0, 0.0},     // xy
        {0.0, 0.0, 0.0, h_s3, 0.0, 0.0}      // xz
    };
    double h_d32 = 3.0 / 2.0;
    double h_s3_8 = sqrt(3.0 / 8.0);
    double h_s5_8 = sqrt(5.0 / 8.0);
    double h_s6 = sqrt(6.0);
    double h_s15 = sqrt(15.0);
    double h_s15_4 = sqrt(15.0 / 4.0);
    double h_s45 = sqrt(45.0);
    double h_s45_8 = sqrt(45.0 / 8.0);
    double h_ftrafo[7][10] = {
        {0.0, 0.0, 0.0, 0.0, -h_s3_8, 0.0, h_s5_8},   // xxx
        {-h_s5_8, 0.0, -h_s3_8, 0.0, 0.0, 0.0, 0.0},  // yyy
        {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},          // zzz
        {h_s45_8, 0.0, -h_s3_8, 0.0, 0.0, 0.0, 0.0},  // xxy
        {0.0, 0.0, 0.0, -h_d32, 0.0, h_s15_4, 0.0},   // xxz
        {0.0, 0.0, 0.0, 0.0, -h_s3_8, 0.0, -h_s45_8}, // xyy
        {0.0, 0.0, 0.0, -h_d32, 0.0, -h_s15_4, 0.0}   // yyz
    };
    double h_d38 = 3.0 / 8.0;
    double h_d34 = 3.0 / 4.0;
    double h_s5_16 = sqrt(5.0 / 16.0);
    double h_s10 = sqrt(10.0);
    double h_s10_8 = sqrt(10.0 / 8.0);
    double h_s35_4 = sqrt(35.0 / 4.0);
    double h_s35_8 = sqrt(35.0 / 8.0);
    double h_s35_64 = sqrt(35.0 / 64.0);
    double h_s45_4 = sqrt(45.0 / 4.0);
    double h_s315_8 = sqrt(315.0 / 8.0);
    double h_s315_16 = sqrt(315.0 / 16.0);
    double h_gtrafo[9][15] = {
        {0.0, 0.0, 0.0, 0.0, h_d38, 0.0, -h_s5_16, 0.0, h_s35_64}, // xxxx
        {0.0, 0.0, 0.0, 0.0, h_d38, 0.0, h_s5_16, 0.0, h_s35_64},  // yyyy
        {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},             // zzzz
        {h_s35_4, 0.0, -h_s10_8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},    // xxxy
        {0.0, 0.0, 0.0, 0.0, 0.0, -h_s45_8, 0.0, h_s35_8, 0.0},    // xxxz
        {-h_s35_4, 0.0, -h_s10_8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},   // xyyy
        {0.0, -h_s35_8, 0.0, -h_s45_8, 0.0, 0.0, 0.0, 0.0, 0.0},   // yyyz
        {0.0, 0.0, 0.0, 0.0, 0.0, h_s10, 0.0, 0.0, 0.0},           // xzzz
        {0.0, 0.0, 0.0, h_s10, 0.0, 0.0, 0.0, 0.0, 0.0}            // yzzz
    };

    // Copy the values to the device constants
    cudaMemcpyToSymbol(s3, &h_s3, sizeof(double));
    cudaMemcpyToSymbol(s3_4, &h_s3_4, sizeof(double));
    cudaMemcpyToSymbol(dtrafo, h_dtrafo, sizeof(double) * 5 * 6);
    cudaMemcpyToSymbol(d32, &h_d32, sizeof(double));
    cudaMemcpyToSymbol(s3_8, &h_s3_8, sizeof(double));
    cudaMemcpyToSymbol(s5_8, &h_s5_8, sizeof(double));
    cudaMemcpyToSymbol(s6, &h_s6, sizeof(double));
    cudaMemcpyToSymbol(s15, &h_s15, sizeof(double));
    cudaMemcpyToSymbol(s15_4, &h_s15_4, sizeof(double));
    cudaMemcpyToSymbol(s45, &h_s45, sizeof(double));
    cudaMemcpyToSymbol(s45_8, &h_s45_8, sizeof(double));
    cudaMemcpyToSymbol(ftrafo, h_ftrafo, sizeof(double) * 7 * 10);
    cudaMemcpyToSymbol(d38, &h_d38, sizeof(double));
    cudaMemcpyToSymbol(d34, &h_d34, sizeof(double));
    cudaMemcpyToSymbol(s5_16, &h_s5_16, sizeof(double));
    cudaMemcpyToSymbol(s10, &h_s10, sizeof(double));
    cudaMemcpyToSymbol(s10_8, &h_s10_8, sizeof(double));
    cudaMemcpyToSymbol(s35_4, &h_s35_4, sizeof(double));
    cudaMemcpyToSymbol(s35_8, &h_s35_8, sizeof(double));
    cudaMemcpyToSymbol(s35_64, &h_s35_64, sizeof(double));
    cudaMemcpyToSymbol(s45_4, &h_s45_4, sizeof(double));
    cudaMemcpyToSymbol(s315_8, &h_s315_8, sizeof(double));
    cudaMemcpyToSymbol(s315_16, &h_s315_16, sizeof(double));
    cudaMemcpyToSymbol(gtrafo, h_gtrafo, sizeof(double) * 9 * 15);
}

// Assuming constants like s3, s3_4, dtrafo, ftrafo, and gtrafo are already defined as __device__ __constant__

__device__ void transform0(int lj, int li, const double *cart, double *sphr, int cart_rows, int cart_cols)
{
    switch (li)
    {
    case 0:
    case 1:
        switch (lj)
        {
        case 0:
        case 1:
            // Copy cart to sphr
            for (int i = 0; i < cart_rows; ++i)
            {
                for (int j = 0; j < cart_cols; ++j)
                {
                    sphr[i * cart_cols + j] = cart[i * cart_cols + j];
                }
            }
            break;
        case 2:
            // sphr = matmul(dtrafo, cart)
            sphr[0] = cart[2] - 0.5 * (cart[0] + cart[1]);
            sphr[1] = s3 * cart[4];
            sphr[2] = s3 * cart[5];
            sphr[3] = s3_4 * (cart[0] - cart[1]);
            sphr[4] = s3 * cart[3];
            break;
        case 3:
            // sphr = matmul(ftrafo, cart)
            for (int i = 0; i < 7; ++i)
            {
                sphr[i] = 0.0;
                for (int j = 0; j < 10; ++j)
                {
                    sphr[i] += ftrafo[i][j] * cart[j];
                }
            }
            break;
        case 4:
            // sphr = matmul(gtrafo, cart)
            for (int i = 0; i < 9; ++i)
            {
                sphr[i] = 0.0;
                for (int j = 0; j < 15; ++j)
                {
                    sphr[i] += gtrafo[i][j] * cart[j];
                }
            }
            break;
        default:
            printf("[Fatal] Moments higher than g are not supported\n");
            return;
        }
        break;

    case 2:
        switch (lj)
        {
        case 0:
        case 1:
            // sphr = matmul(cart, transpose(dtrafo))
            for (int i = 0; i < cart_rows; ++i)
            {
                sphr[i * 5 + 0] = cart[i * 6 + 2] - 0.5 * (cart[i * 6 + 0] + cart[i * 6 + 1]);
                sphr[i * 5 + 1] = s3 * cart[i * 6 + 4];
                sphr[i * 5 + 2] = s3 * cart[i * 6 + 5];
                sphr[i * 5 + 3] = s3_4 * (cart[i * 6 + 0] - cart[i * 6 + 1]);
                sphr[i * 5 + 4] = s3 * cart[i * 6 + 3];
            }
            break;
        case 2:
            // sphr = matmul(dtrafo, matmul(cart, transpose(dtrafo)))
            // This is a simplified example; the full implementation would require nested loops
            printf("[Fatal] Higher-order transformations not implemented\n");
            return;
        case 3:
            // sphr = matmul(ftrafo, matmul(cart, transpose(dtrafo)))
            printf("[Fatal] Higher-order transformations not implemented\n");
            return;
        case 4:
            // sphr = matmul(gtrafo, matmul(cart, transpose(dtrafo)))
            printf("[Fatal] Higher-order transformations not implemented\n");
            return;
        default:
            printf("[Fatal] Moments higher than g are not supported\n");
            return;
        }
        break;

    case 3:
        switch (lj)
        {
        case 0:
        case 1:
            // sphr = matmul(cart, transpose(ftrafo))
            for (int i = 0; i < cart_rows; ++i)
            {
                for (int j = 0; j < 7; ++j)
                {
                    sphr[i * 7 + j] = 0.0;
                    for (int k = 0; k < 10; ++k)
                    {
                        sphr[i * 7 + j] += cart[i * 10 + k] * ftrafo[j][k];
                    }
                }
            }
            break;
        case 2:
        case 3:
        case 4:
            printf("[Fatal] Higher-order transformations not implemented\n");
            return;
        default:
            printf("[Fatal] Moments higher than g are not supported\n");
            return;
        }
        break;

    case 4:
        switch (lj)
        {
        case 0:
        case 1:
            // sphr = matmul(cart, transpose(gtrafo))
            for (int i = 0; i < cart_rows; ++i)
            {
                for (int j = 0; j < 9; ++j)
                {
                    sphr[i * 9 + j] = 0.0;
                    for (int k = 0; k < 15; ++k)
                    {
                        sphr[i * 9 + j] += cart[i * 15 + k] * gtrafo[j][k];
                    }
                }
            }
            break;
        case 2:
        case 3:
        case 4:
            printf("[Fatal] Higher-order transformations not implemented\n");
            return;
        default:
            printf("[Fatal] Moments higher than g are not supported\n");
            return;
        }
        break;

    default:
        printf("[Fatal] Moments higher than g are not supported\n");
        return;
    }
}

extern "C"
{
    void get_vec_(
        const double *xyz_iat,
        const double *xyz_jat,
        const double *trans,
        double vec[3])
    {
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

        for (size_t k = 0; k < 3; ++k)
        {
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

    void call_hello_kernel_()
    {
        hello_kernel<<<1, 1>>>();
        cudaDeviceSynchronize(); // Wait for the kernel to finish.
    }
    // Suppose we define
    // void get_hamiltonian(mol, lattr, list, ...) {}
    //
    // void get_hamiltonian_cu(
    //     structure_type mol
    // ) {
    //     //
    // }
}

// int main(void) {
//     init_constants();
//     testKernel<<<1,1>>>();
//     cudaDeviceSynchronize();
// }