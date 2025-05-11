#include <cstdio>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include "utils.h"
#include "types.h"

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

__global__ void hello_kernel()
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
}
__global__ void get_hamiltonian(
  const structure_type mol, 
  const tensor2d_t<double> trans,
  const adjacency_list alist,
  const basis_type bas
)
{
  printf("================= KERNEL =================\n");
  printf("mol = \n");
  printstruct(mol);

  printf("trans = \n");
  for(int i = 0; i < trans.dim1; ++i)
  {
    for(int j = 0; j < trans.dim2; ++j)
    {
      printf("%f ", trans(i,j));
    }
  }
  printf("\n");

  printf("alist = \n");
  printstruct(alist);

  printf("bas = \n");
  printstruct(bas);
}
extern "C" void cuda_get_hamiltonian_kernel_(
    int nao,
    int nelem,

    /* structure_type */
    const int mol_nat,
    const int mol_nid,
    const int mol_nbd,
    const int *mol_id, int mol_id_dim1,
    const int *mol_num, int mol_num_dim1,
    const double *mol_xyz, int mol_xyz_dim1, int mol_xyz_dim2,
    const int mol_uhf,
    const double mol_charge,
    const double *mol_lattice, int mol_lattice_dim1, int mol_lattice_dim2,
    const int *mol_periodic, int mol_periodic_dim1,
    const int *mol_bond, int mol_bond_dim1, int mol_bond_dim2,

    /* trans for lattice */
    const double *trans, const int trans_dim1, const int trans_dim2,

    /* adjacency_list */
    const int *alist_inl, int alist_inl_dim1,
    const int *alist_nnl, int alist_nnl_dim1,
    const int *alist_nlat, int alist_nlat_dim1,
    const int *alist_nltr, int alist_nltr_dim1,

    /* basis_type */
    const int bas_maxl,
    const int bas_nsh,
    const int bas_nao,
    const double bas_intcut,
    const double bas_min_alpha,
    const int *bas_nsh_id, int bas_nsh_id_dim1,
    const int *bas_nsh_at, int bas_nsh_at_dim1,
    const int *bas_nao_sh, int bas_nao_sh_dim1,
    const int *bas_iao_sh, int bas_iao_sh_dim1,
    const int *bas_ish_at, int bas_ish_at_dim1,
    const int *bas_ao2at, int bas_ao2at_dim1,
    const int *bas_ao2sh, int bas_ao2sh_dim1,
    const int *bas_sh2at, int bas_sh2at_dim1,
    const cgto_type *cgto, int cgto_dim1, int cgto_dim2,

    /* tb_hamiltonian */
    const double *h0_selfenergy, int h0_selfenergy_dim1, int h0_selfenergy_dim2,
    const double *h0_kcn, int h0_kcn_dim1, int h0_kcn_dim2,
    const double *h0_kq1, int h0_kq1_dim1, int h0_kq1_dim2,
    const double *h0_kq2, int h0_kq2_dim1, int h0_kq2_dim2,
    const double *h0_hscale, int h0_hscale_dim1, int h0_hscale_dim2, int h0_hscale_dim3, int h0_hscale_dim4,
    const double *h0_shpoly, int h0_shpoly_dim1, int h0_shpoly_dim2,
    const double *h0_rad, int h0_rad_dim1,
    const double *h0_refocc, int h0_refocc_dim1, int h0_refocc_dim2,

    // Diagonal elememts of the Hamiltonian  (nel)
    const double *selfenergy,
    // Overlap integral matrix (nao, nao)
    double *overlap,
    // Dipole moment integral matrix (nao, nao, 3)
    double *dpint,
    // Quadrupole moment integral matrix (nao, nao, 6)
    double *qpint,
    // Hamiltonian matrix (nao, nao)
    double *hamiltonian)
{
  /* Pack args into structures */
  const adjacency_list alist{
      tensor1d_t(alist_inl, alist_inl_dim1),
      tensor1d_t(alist_nnl, alist_nnl_dim1),
      tensor1d_t(alist_nlat, alist_nlat_dim1),
      tensor1d_t(alist_nltr, alist_nltr_dim1)};

  const structure_type mol{
      mol_nat, mol_nid, mol_nbd,
      tensor1d_t(mol_id, mol_id_dim1),
      tensor1d_t(mol_num, mol_num_dim1),
      tensor2d_t(mol_xyz, mol_xyz_dim1, mol_xyz_dim2),
      mol_uhf,
      mol_charge,
      tensor2d_t(mol_lattice, mol_lattice_dim1, mol_lattice_dim2),
      tensor1d_t(mol_periodic, mol_periodic_dim1),
      tensor2d_t(mol_bond, mol_bond_dim1, mol_bond_dim2)};

  const tb_hamiltonian h0{
      h0_selfenergy, h0_selfenergy_dim1, h0_selfenergy_dim2,
      h0_kcn, h0_kcn_dim1, h0_kcn_dim2,
      h0_kq1, h0_kq1_dim1, h0_kq1_dim2,
      h0_kq2, h0_kq2_dim1, h0_kq2_dim2,
      h0_hscale, h0_hscale_dim1, h0_hscale_dim2, h0_hscale_dim3, h0_hscale_dim4,
      h0_shpoly, h0_shpoly_dim1, h0_shpoly_dim2,
      h0_rad, h0_rad_dim1,
      h0_refocc, h0_refocc_dim1, h0_refocc_dim2};
  const basis_type bas{
      bas_maxl,
      bas_nsh,
      bas_nao,
      bas_intcut,
      bas_min_alpha,
      // bas_nsh_id, bas_nsh_id_dim1,
      tensor1d_t(bas_nsh_id, bas_nsh_id_dim1),
      // bas_nsh_at, bas_nsh_at_dim1,
      tensor1d_t(bas_nsh_at, bas_nsh_at_dim1),
      // bas_nao_sh, bas_nao_sh_dim1,
      tensor1d_t(bas_nao_sh, bas_nao_sh_dim1),
      // bas_iao_sh, bas_iao_sh_dim1,
      tensor1d_t(bas_iao_sh, bas_iao_sh_dim1),
      // bas_ish_at, bas_ish_at_dim1,
      tensor1d_t(bas_ish_at, bas_ish_at_dim1),
      // bas_ao2at, bas_ao2at_dim1,
      tensor1d_t(bas_ao2at, bas_ao2at_dim1),
      // bas_ao2sh, bas_ao2sh_dim1,
      tensor1d_t(bas_ao2sh, bas_ao2sh_dim1),
      // bas_sh2at, bas_sh2at_dim1,
      tensor1d_t(bas_sh2at, bas_sh2at_dim1),
      // cgto, cgto_dim1, cgto_dim2};
      tensor2d_t(cgto, cgto_dim1, cgto_dim2)};
  printf("================= CUDA =================\n");


  const structure_type d_mol{
    mol.nat,
    mol.nid,
    mol.nbd,
    mol.id.to_device(),
    mol.num.to_device(),
    mol.xyz.to_device(),
    mol.uhf,
    mol.charge,
    mol.lattice.to_device(),
    mol.periodic.to_device(),
    mol.bond.to_device()};

  const tensor2d_t<double> d_trans = tensor2d_t<double>(trans, trans_dim1, trans_dim2).to_device();

  const adjacency_list d_alist{
    alist.inl.to_device(),
    alist.nnl.to_device(),
    alist.nlat.to_device(),
    alist.nltr.to_device()};

  const basis_type d_basis{
    bas_maxl,
    bas_nsh,
    bas_nao,
    bas_intcut,
    bas_min_alpha,
    // tensor1d_t(bas.nsh_id, bas.nsh_id_dim1).to_device(),
    bas.nsh_id.to_device(),
    // tensor1d_t(bas.nsh_at, bas.nsh_at_dim1).to_device(),
    bas.nsh_at.to_device(),
    // tensor1d_t(bas.nao_sh, bas.nao_sh_dim1).to_device(),
    bas.nao_sh.to_device(),
    // tensor1d_t(bas.iao_sh, bas.iao_sh_dim1).to_device(),
    bas.iao_sh.to_device(),
    // tensor1d_t(bas.ish_at, bas.ish_at_dim1).to_device(),
    bas.ish_at.to_device(),
    // tensor1d_t(bas.ao2at, bas.ao2at_dim1).to_device(),
    bas.ao2at.to_device(),
    // tensor1d_t(bas.ao2sh, bas.ao2sh_dim1).to_device(),
    bas.ao2sh.to_device(),
    // tensor1d_t(bas.sh2at, bas.sh2at_dim1).to_device(),
    bas.sh2at.to_device(),
    // cgto, cgto_dim1, cgto_dim2};
    // tensor2d_t<cgto_type>(bas.cgto, bas.cgto_dim1, bas.cgto_dim2).to_device()};
    bas.cgto.to_device()};

  // Start timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  
  // Launch kernel
  get_hamiltonian<<<1, 1>>>(
    d_mol,
    d_trans,
    d_alist,
    d_basis
  );

  // Stop timing
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
    return;
  }

  // Calculate elapsed time
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel execution time: %f ms\n", milliseconds);

  // Clean up events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // printf("at %s:%i\n", __func__, __LINE__);
  // printf("nao = %i\n", nao);
  // printf("nelem = %i\n", nelem);
  // printstruct(bas);
  // printstruct(alist);
  // printstruct(mol);
  // printstruct(h0);

  // printf("bas_nsh_id = \n");
  // printr(bas_nsh_id_dim1, bas_nsh_id);
  // printf("bas_nsh_at = \n");
  // printr(bas_nsh_at_dim1, bas_nsh_at);
  // printf("bas_nao_sh = \n");
  // printr(bas_nao_sh_dim1, bas_nao_sh);
  // printf("bas_iao_sh = \n");
  // printr(bas_iao_sh_dim1, bas_iao_sh);
  // printf("bas_ish_at = \n");
  // printr(bas_ish_at_dim1, bas_ish_at);
  // printf("bas_ao2at = \n");
  // printr(bas_ao2at_dim1, bas_ao2at);
  // printf("bas_ao2sh = \n");
  // printr(bas_ao2sh_dim1, bas_ao2sh);
  // printf("bas_sh2at = \n");
  // printr(bas_sh2at_dim1, bas_sh2at);

  // for (int i = 0; i < cgto_dim1; ++i)
  // {
  //   for (int j = 0; j < cgto_dim2; ++j)
  //   {
  //     printf("cgto[%d][%d] = ", i, j);
  //     printstruct(cgto[i * cgto_dim2 + j]);
  //   }
  // }

  // printf("h0_selfenergy = \n");
  // printr(h0_selfenergy_dim1, h0_selfenergy_dim2, h0_selfenergy);
  // printf("h0_kcn = \n");
  // printr(h0_kcn_dim1, h0_kcn_dim2, h0_kcn);
  // printf("h0_kq1 = \n");
  // printr(h0_kq1_dim1, h0_kq1_dim2, h0_kq1);
  // printf("h0_kq2 = \n");
  // printr(h0_kq2_dim1, h0_kq2_dim2, h0_kq2);
  // printf("h0_hscale = \n");
  // printr(h0_hscale_dim1, h0_hscale_dim2, h0_hscale_dim3, h0_hscale_dim4, h0_hscale);

  // printf("selfenergy = \n"); printr(nelem, selfenergy);
  // printf("overlap = \n"); printr(nao, nao, overlap);
  // printf("dpint = \n"); printr(nao, nao, 3, dpint);
  // printf("qpint = \n"); printr(nao, nao, 6, qpint);
  // printf("hamiltonian = \n"); printr(nao, nao, hamiltonian);

  // get_hamiltonian<<<1,1>>>();
  // cudaDeviceSynchronize();
}
