#ifndef TYPES_H
#define TYPES_H
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <float.h> // For huge value if needed
#include "tensor.h"
#define MAXG 12 
#define SYMBOL_LENGTH 10 // Replace with the actual value of symbol_length in Fortran

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
    // const int *id;  const int id_dim1;
    const tensor1d_t<int> id; // 1D array of atom IDs
    // const int *num; const int num_dim1;
    const tensor1d_t<int> num; // 1D array of atom numbers
    // const double *xyz; const int xyz_dim1; const int xyz_dim2;
    const tensor2d_t<double> xyz; // 2D array of atom coordinates
    const int uhf;
    const double charge;
    // const double *lattice; const int lattice_dim1; const int lattice_dim2;
    const tensor2d_t<double> lattice; // 2D array of lattice vectors
    // const int *periodic; const int periodic_dim1;
    const tensor1d_t<int> periodic; // 1D array of periodicity flags
    // const int *bond; const int bond_dim1; const int bond_dim2;
    const tensor2d_t<int> bond; // 2D array of bond information
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
  printf("  lattice: ");
  for (int i = 0; i < str.lattice.dim1; ++i)
  {
    for (int j = 0; j < str.lattice.dim2; ++j)
    {
      printf("%f, ", str.lattice(i, j));
    }
    printf("\n");
  }
  printf("  periodic: ");
  for (int i = 0; i < str.periodic.dim1; ++i)
  {
    printf("%d, ", str.periodic[i]);
  }
  printf("\n");
  printf("  bond: ");
  for (int i = 0; i < str.bond.dim1; ++i)
  {
    for (int j = 0; j < str.bond.dim2; ++j)
    {
      printf("%d, ", str.bond(i, j));
    }
    printf("\n");
  }
}

typedef struct {
  const tensor1d_t<int> inl; // Offset index in the neighbour map (dynamic array)
  const tensor1d_t<int> nnl; // Number of neighbours for each atom (dynamic array)
  const tensor1d_t<int> nlat; // Index of the neighbouring atom (dynamic array)
  const tensor1d_t<int> nltr; // Cell index of the neighbouring atom (dynamic array)
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
    int ang;               // Angular momentum of this basis function
    int nprim;             // Contraction length of this basis function
    double alpha[MAXG];    // Exponent of the primitive Gaussian functions
    double coeff[MAXG];    // Contraction coefficients of the primitive Gaussian functions
} cgto_type;

void printstruct(const cgto_type cgto)
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



// Equivalent C struct for basis_type
typedef struct {
  /*        !> basis_type
        integer(c_int), value :: bas_maxl
        integer(c_int), value :: bas_nsh
        integer(c_int), value :: bas_nao
        real(c_double), value :: bas_intcut
        real(c_double), value :: bas_min_alpha
        integer(c_int), intent(in) :: bas_nsh_id(*)
        integer(c_int), value :: bas_nsh_id_dim1
        integer(c_int), intent(in) :: bas_nsh_at(*)
        integer(c_int), value :: bas_nsh_at_dim1
        integer(c_int), intent(in) :: bas_nao_sh(*)
        integer(c_int), value :: bas_nao_sh_dim1
        integer(c_int), intent(in) :: bas_iao_sh(*)
        integer(c_int), value :: bas_iao_sh_dim1
        integer(c_int), intent(in) :: bas_ish_at(*)
        integer(c_int), value :: bas_ish_at_dim1
        integer(c_int), intent(in) :: bas_ao2at(*)
        integer(c_int), value :: bas_ao2at_dim1
        integer(c_int), intent(in) :: bas_ao2sh(*)
        integer(c_int), value :: bas_ao2sh_dim1
        integer(c_int), intent(in) :: bas_sh2at(*)
        integer(c_int), value :: bas_sh2at_dim1
        type(cgto_type), intent(in) :: cgto(*)
        integer(c_int), value :: cgto_dim1, cgto_dim2*/
  const int bas_maxl;
  const int bas_nsh;
  const int bas_nao;
  const double bas_intcut;
  const double bas_min_alpha;
  const int *nsh_id; const int nsh_id_dim1;
  const int *nsh_at; const int nsh_at_dim1;
  const int *nao_sh; const int nao_sh_dim1;
  const int *iao_sh; const int iao_sh_dim1;
  const int *ish_at; const int ish_at_dim1;
  const int *ao2at; const int ao2at_dim1;
  const int *ao2sh; const int ao2sh_dim1;
  const int *sh2at; const int sh2at_dim1;
  const cgto_type *cgto; const int cgto_dim1; const int cgto_dim2;
} basis_type;

void printstruct(const basis_type bas)
{
  printf("basis_type:\n");
  printf("  bas_maxl: %d\n", bas.bas_maxl);
  printf("  bas_nsh: %d\n", bas.bas_nsh);
  printf("  bas_nao: %d\n", bas.bas_nao);
  printf("  bas_intcut: %f\n", bas.bas_intcut);
  printf("  bas_min_alpha: %f\n", bas.bas_min_alpha);
  printf("  nsh_id: ");
  for (int i = 0; i < bas.nsh_id_dim1; ++i)
  {
    printf("%d, ", bas.nsh_id[i]);
  }
  printf("\n");
  printf("  nsh_at: ");
  for (int i = 0; i < bas.nsh_at_dim1; ++i)
  {
    printf("%d, ", bas.nsh_at[i]);
  }
  printf("\n");
  printf("  nao_sh: ");
  for (int i = 0; i < bas.nao_sh_dim1; ++i)
  {
    printf("%d, ", bas.nao_sh[i]);
  }
  printf("\n");
  printf("  iao_sh: ");
  for (int i = 0; i < bas.iao_sh_dim1; ++i)
  {
    printf("%d, ", bas.iao_sh[i]);
  }
  printf("\n");
  printf("  ish_at: ");
  for (int i = 0; i < bas.ish_at_dim1; ++i)
  {
    printf("%d, ", bas.ish_at[i]);
  }
  printf("\n");
  printf("  ao2at: ");
  for (int i = 0; i < bas.ao2at_dim1; ++i)
  {
    printf("%d, ", bas.ao2at[i]);
  }
  printf("\n");
  printf("  ao2sh: ");
  for (int i = 0; i < bas.ao2sh_dim1; ++i)
  {
    printf("%d, ", bas.ao2sh[i]);
  }
  printf("\n");
  printf("  sh2at: ");
  for (int i = 0; i < bas.sh2at_dim1; ++i)
  {
    printf("%d, ", bas.sh2at[i]);
  }
  printf("\n");
  printf("  cgto: ");
  // DEBUG
  printf("cgto dims are %d, %d\n", bas.cgto_dim1 + 1, bas.cgto_dim2 + 1);
  for(int i = 0; i < bas.cgto_dim1; ++i)
  {
    for (int j = 0; j < bas.cgto_dim2; ++j)
    {
      printf("cgto[%d][%d] = \n", i, j);
      printstruct(bas.cgto[i * bas.cgto_dim2 + j]);
    }
  }
}

// Hamiltonian interaction data structure
typedef struct {
  /*        !> tb_hamiltonian
        real(c_double), intent(in) :: h0_selfenergy(*)
        integer(c_int), value :: h0_selfenergy_dim1, h0_selfenergy_dim2
        real(c_double), intent(in) :: h0_kcn(*)
        integer(c_int), value :: h0_kcn_dim1, h0_kcn_dim2
        real(c_double), intent(in) :: h0_kq1(*)
        integer(c_int), value :: h0_kq1_dim1, h0_kq1_dim2
        real(c_double), intent(in) :: h0_kq2(*)
        integer(c_int), value :: h0_kq2_dim1, h0_kq2_dim2
        real(c_double), intent(in) :: h0_hscale(*)
        integer(c_int), value :: h0_hscale_dim1, h0_hscale_dim2, h0_hscale_dim3, h0_hscale_dim4
        real(c_double), intent(in) :: h0_shpoly(*)
        integer(c_int), value :: h0_shpoly_dim1, h0_shpoly_dim2
        real(c_double), intent(in) :: h0_rad(*)
        integer(c_int), value :: h0_rad_dim1
        real(c_double), intent(in) :: h0_refocc(*)
        integer(c_int), value :: h0_refocc_dim1, h0_refocc_dim2*/
  const double *selfenergy; const int selfenergy_dim1; const int selfenergy_dim2;
  const double *kcn; const int kcn_dim1; const int kcn_dim2;
  const double *kq1; const int kq1_dim1; const int kq1_dim2;
  const double *kq2; const int kq2_dim1; const int kq2_dim2;
  const double *hscale; const int hscale_dim1; const int hscale_dim2; const int hscale_dim3; const int hscale_dim4;
  const double *shpoly; const int shpoly_dim1; const int shpoly_dim2;
  const double *rad; const int rad_dim1;
  const double *refocc; const int refocc_dim1; const int refocc_dim2;
} tb_hamiltonian;

void printstruct(const tb_hamiltonian h)
{
  printf("tb_hamiltonian:\n");
  printf("  selfenergy: ");
  for (int i = 0; i < h.selfenergy_dim1; ++i)
  {
    for (int j = 0; j < h.selfenergy_dim2; ++j)
    {
      printf("%f, ", h.selfenergy[i * h.selfenergy_dim2 + j]);
    }
    printf("\n");
  }
  printf("  kcn: ");
  for (int i = 0; i < h.kcn_dim1; ++i)
  {
    for (int j = 0; j < h.kcn_dim2; ++j)
    {
      printf("%f, ", h.kcn[i * h.kcn_dim2 + j]);
    }
    printf("\n");
  }
  printf("  kq1: ");
  for (int i = 0; i < h.kq1_dim1; ++i)
  {
    for (int j = 0; j < h.kq1_dim2; ++j)
    {
      printf("%f, ", h.kq1[i * h.kq1_dim2 + j]);
    }
    printf("\n");
  }
  printf("  kq2: ");
  for (int i = 0; i < h.kq2_dim1; ++i)
  {
    for (int j = 0; j < h.kq2_dim2; ++j)
    {
      printf("%f, ", h.kq2[i * h.kq2_dim2 + j]);
    }
    printf("\n");
  }
  printf("  hscale: ");
  for (int i = 0; i < h.hscale_dim1; ++i)
  {
    for (int j = 0; j < h.hscale_dim2; ++j)
    {
      for (int k = 0; k < h.hscale_dim3; ++k)
      {
        for (int l = 0; l < h.hscale_dim4; ++l)
        {
          printf("%f, ", h.hscale[i * h.hscale_dim2 * h.hscale_dim3 * h.hscale_dim4 + j * h.hscale_dim3 * h.hscale_dim4 + k * h.hscale_dim4 + l]);
        }
      }
      printf("\n");
    }
    printf("\n");
  }
  printf("  shpoly: ");
  for (int i = 0; i < h.shpoly_dim1; ++i)
  {
    for (int j = 0; j < h.shpoly_dim2; ++j)
    {
      printf("%f, ", h.shpoly[i * h.shpoly_dim2 + j]);
    }
    printf("\n");
  }
  printf("  rad: ");
  for (int i = 0; i < h.rad_dim1; ++i)
  {
    printf("%f, ", h.rad[i]);
  }
  printf("\n");
  printf("  refocc: ");
  for (int i = 0; i < h.refocc_dim1; ++i)
  {
    for (int j = 0; j < h.refocc_dim2; ++j)
    {
      printf("%f, ", h.refocc[i * h.refocc_dim2 + j]);
    }
    printf("\n");
  }
}

#endif