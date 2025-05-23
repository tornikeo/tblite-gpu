#ifndef TYPES_H
#define TYPES_H
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <float.h> // For huge value if needed
#include "tensor.h"

#define MAXG 12
#define MAXL 4
#define MAXL2 (MAXL * 2)

/*    const int mol_nat,
    const int mol_nid,
    const int mol_nbd,
    const int *mol_id, int mol_id_dim1,
    const int *mol_num, int mol_num_dim1,
    const double *mol_xyz, int mol_xyz_dim1, int mol_xyz_dim2,
    const int mol_uhf, 
    const double mol_charge,
    const double *mol_lattice, int mol_lattice_dim1, int mol_lattice_dim2,
    const int *mol_periodic, int mol_periodic_dim1,
    const int *mol_bond, int mol_bond_dim1, int mol_bond_dim2,*/
typedef struct {
    const int nat;
    const int nid;
    const int nbd;
    const tensor1d_t<const int> id; // 1D array of atom IDs
    const tensor1d_t<const int> num; // 1D array of atom numbers
    const tensor2d_t<const double> xyz; // 2D array of atom coordinates
    const int uhf;
    const double charge;
} structure_type;

__host__ __device__
void printstruct(const structure_type &str)
{
  printf("structure_type:\n");
  printf("  nat: %d\n", str.nat);
  printf("  nid: %d\n", str.nid);
  printf("  nbd: %d\n", str.nbd);
  printf("  id: ");
  for (int i = 0; i < str.id.dim1; ++i)
  {
    printf("%d, ", str.id[i]);
  }
  printf("\n");
  printf("  num: ");
  for (int i = 0; i < str.num.dim1; ++i)
  {
    printf("%d, ", str.num[i]);
  }
  printf("\n");
  printf("  xyz: ");
  for (int i = 0; i < str.xyz.dim1; ++i)
  {
    for (int j = 0; j < str.xyz.dim2; ++j)
    {
      printf("%f, ", str.xyz(i, j));
    }
    printf("\n");
  }
  printf("  uhf: %d\n", str.uhf);
  printf("  charge: %f\n", str.charge);
  printf("\n");
}

typedef struct {
  const tensor1d_t<const int> inl; // Offset index in the neighbour map (dynamic array)
  const tensor1d_t<const int> nnl; // Number of neighbours for each atom (dynamic array)
  const tensor1d_t<const int> nlat; // Index of the neighbouring atom (dynamic array)
  const tensor1d_t<const int> nltr; // Cell index of the neighbouring atom (dynamic array)
} adjacency_list;

__device__ __host__ 
inline void printstruct(const adjacency_list &adj)
{
  printf("adjacency_list:\n");
  printf("  inl: ");
  for (int i = 0; i < adj.inl.dim1; ++i)
  {
    printf("%d, ", adj.inl[i]);
  }
  printf("\n");
  printf("  nnl: ");
  for (int i = 0; i < adj.nnl.dim1; ++i)
  {
    printf("%d, ", adj.nnl[i]);
  }
  printf("\n");
  printf("  nlat: ");
  for (int i = 0; i < adj.nlat.dim1; ++i)
  {
    printf("%d, ", adj.nlat[i]);
  }
  printf("\n");
  printf("  nltr: ");
  for (int i = 0; i < adj.nltr.dim1; ++i)
  {
    printf("%d, ", adj.nltr[i]);
  }
}

typedef struct {
  const int ang;               // Angular momentum of this basis function
  const int nprim;             // Contraction length of this basis function
  const double alpha[MAXG];    // Exponent of the primitive Gaussian functions
  const double coeff[MAXG];    // Contraction coefficients of the primitive Gaussian functions
} cgto_type;

__host__ __device__
void printstruct(const cgto_type &cgto)
{
  printf("cgto_type:\n");
  printf("  ang: %d\n", cgto.ang);
  printf("  nprim: %d\n", cgto.nprim);
  printf("  alpha: ");
  for (int i = 0; i < cgto.nprim; ++i)
  {
    printf("%f, ", cgto.alpha[i]);
  }
  printf("\n");
  printf("  coeff: ");
  for (int i = 0; i < cgto.nprim; ++i)
  {
    printf("%f, ", cgto.coeff[i]);
  }
  printf("\n");
}

typedef struct {
  const int maxl;
  const int nsh;
  const int nao;
  const double intcut;
  const double min_alpha;
  const tensor1d_t<const int> nsh_id; // 1D array of shell IDs
  const tensor1d_t<const int> nsh_at; // 1D array of atom IDs for each shell
  const tensor1d_t<const int> nao_sh; // 1D array of shell indices for each AO
  const tensor1d_t<const int> iao_sh; // 1D array of shell indices for each AO
  const tensor1d_t<const int> ish_at; // 1D array of atom indices for each shell
  const tensor1d_t<const int> ao2at; // 1D array of atom IDs for each AO
  const tensor1d_t<const int> ao2sh; // 1D array of shell indices for each AO
  const tensor1d_t<const int> sh2at; // 1D array of atom IDs for each shell
  const tensor2d_t<const cgto_type> cgto; // 2D array of Gaussian-type orbitals
} basis_type;

__host__ __device__
void printstruct(const basis_type &bas)
{
  printf("basis_type:\n");
  printf("  maxl: %d\n", bas.maxl);
  printf("  nsh: %d\n", bas.nsh);
  printf("  nao: %d\n", bas.nao);
  printf("  intcut: %f\n", bas.intcut);
  printf("  min_alpha: %f\n", bas.min_alpha);
  printf("  nsh_id: ");
  for (int i = 0; i < bas.nsh_id.dim1; ++i)
  {
    printf("%d, ", bas.nsh_id[i]);
  }
  printf("\n");
  printf("  nsh_at: ");
  for (int i = 0; i < bas.nsh_at.dim1; ++i)
  {
    printf("%d, ", bas.nsh_at[i]);
  }
  printf("\n");
  printf("  nao_sh: ");
  for (int i = 0; i < bas.nao_sh.dim1; ++i)
  {
    printf("%d, ", bas.nao_sh[i]);
  }
  printf("\n");
  printf("  iao_sh: ");
  for (int i = 0; i < bas.iao_sh.dim1; ++i)
  {
    printf("%d, ", bas.iao_sh[i]);
  }
  printf("\n");
  printf("  ish_at: ");
  for (int i = 0; i < bas.ish_at.dim1; ++i)
  {
    printf("%d, ", bas.ish_at[i]);
  }
  printf("\n");
  printf("  ao2at: ");
  for (int i = 0; i < bas.ao2at.dim1; ++i)
  {
    printf("%d, ", bas.ao2at[i]);
  }
  printf("\n");
  printf("  ao2sh: ");
  for (int i = 0; i < bas.ao2sh.dim1; ++i)
  {
    printf("%d, ", bas.ao2sh[i]);
  }
  printf("\n");
  printf("  sh2at: ");
  for (int i = 0; i < bas.sh2at.dim1; ++i)
  {
    printf("%d, ", bas.sh2at[i]);
  }
  printf("\n");
  printf("  cgto: ");
  // DEBUG
  printf("cgto dims are %d, %d\n", bas.cgto.dim1 + 1, bas.cgto.dim2 + 1);
  for(int i = 0; i < bas.cgto.dim1; ++i)
  {
    for (int j = 0; j < bas.cgto.dim2; ++j)
    {
      printf("cgto[%d][%d] = \n", i, j);
      printstruct(bas.cgto(i,j));
    }
  }
}

// Hamiltonian interaction data structure
typedef struct {
  const tensor2d_t<const double> selfenergy; // 2D array of self-energy
  const tensor2d_t<const double> kcn; // 2D array of kcn
  const tensor2d_t<const double> kq1; // 2D array of kq1
  const tensor2d_t<const double> kq2; // 2D array of kq2
  const tensor4d_t<const double> hscale; // 4D array of hscale
  const tensor2d_t<const double> shpoly; // 2D array of shpoly
  const tensor1d_t<const double> rad; // 1D array of radial functions
  const tensor2d_t<const double> refocc; // 2D array of reference occupations
} tb_hamiltonian;

__host__ __device__
void printstruct(const tb_hamiltonian &h0)
{
  printf("tb_hamiltonian:\n");
  printf("  selfenergy: ");
  for (int i = 0; i < h0.selfenergy.dim1; ++i)
  {
    for (int j = 0; j < h0.selfenergy.dim2; ++j)
    {
      printf("%f, ", h0.selfenergy(i, j));
    }
    printf("\n");
  }
  printf("  kcn: ");
  for (int i = 0; i < h0.kcn.dim1; ++i)
  {
    for (int j = 0; j < h0.kcn.dim2; ++j)
    {
      printf("%f, ", h0.kcn(i, j));
    }
    printf("\n");
  }
  printf("  kq1: ");
  for (int i = 0; i < h0.kq1.dim1; ++i)
  {
    for (int j = 0; j < h0.kq1.dim2; ++j)
    {
      printf("%f, ", h0.kq1(i, j));
    }
    printf("\n");
  }
  printf("  kq2: ");
  for (int i = 0; i < h0.kq2.dim1; ++i)
  {
    for (int j = 0; j < h0.kq2.dim2; ++j)
    {
      printf("%f, ", h0.kq2(i, j));
    }
    printf("\n");
  }
  printf("  hscale: ");
  h0.hscale.print(); 
  printf("  shpoly: ");
  for (int i = 0; i < h0.shpoly.dim1; ++i)
  {
    for (int j = 0; j < h0.shpoly.dim2; ++j)
    {
      printf("%f, ", h0.shpoly(i, j));
    }
    printf("\n");
  }
  printf("  rad: ");
  for (int i = 0; i < h0.rad.dim1; ++i)
  {
    printf("%f, ", h0.rad[i]);
  }
  printf("\n");
  printf("  refocc: ");
  for (int i = 0; i < h0.refocc.dim1; ++i)
  {
    for (int j = 0; j < h0.refocc.dim2; ++j)
    {
      printf("%f, ", h0.refocc(i, j));
    }
    printf("\n");
  }
}

#endif