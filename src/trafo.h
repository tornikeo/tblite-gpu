#ifndef TRAFO_H
#define TRAFO_H
#include <stddef.h>
#include <stdio.h>
#include <cuda_runtime.h>


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


#endif