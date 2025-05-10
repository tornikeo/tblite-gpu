#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <float.h> // For huge value if needed

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
    const int *id;  const int id_dim1;
    const int *num; const int num_dim1;
    const double *xyz; const int xyz_dim1; const int xyz_dim2;
    const int uhf;
    const double charge;
    const double *lattice; const int lattice_dim1; const int lattice_dim2;
    const int *periodic; const int periodic_dim1;
    const int *bond; const int bond_dim1; const int bond_dim2;
} structure_type;

void printstruct(const structure_type str)
{
  printf("structure_type:\n");
  printf("  nat: %d\n", str.nat);
  printf("  nid: %d\n", str.nid);
  printf("  nbd: %d\n", str.nbd);
  printf("  id: ");
  for (int i = 0; i < str.id_dim1; ++i)
  {
    printf("%d, ", str.id[i]);
  }
  printf("\n");
  printf("  num: ");
  for (int i = 0; i < str.num_dim1; ++i)
  {
    printf("%d, ", str.num[i]);
  }
  printf("\n");
  printf("  xyz: ");
  for (int i = 0; i < str.xyz_dim1; ++i)
  {
    for (int j = 0; j < str.xyz_dim2; ++j)
    {
      printf("%f, ", str.xyz[i * str.xyz_dim2 + j]);
    }
    printf("\n");
  }
  printf("  uhf: %d\n", str.uhf);
  printf("  charge: %f\n", str.charge);
  printf("  lattice: ");
  for (int i = 0; i < str.lattice_dim1; ++i)
  {
    for (int j = 0; j < str.lattice_dim2; ++j)
    {
      printf("%f, ", str.lattice[i * str.lattice_dim2 + j]);
    }
    printf("\n");
  }
  printf("  periodic: ");
  for (int i = 0; i < str.periodic_dim1; ++i)
  {
    printf("%d, ", str.periodic[i]);
  }
  printf("\n");
  printf("  bond: ");
  for (int i = 0; i < str.bond_dim1; ++i)
  {
    for (int j = 0; j < str.bond_dim2; ++j)
    {
      printf("%d, ", str.bond[i * str.bond_dim2 + j]);
    }
    printf("\n");
  }
}

typedef struct {
  const int *inl;  // Offset index in the neighbour map (dynamic array)
  const int inl_dim1;
  const int *nnl;  // Number of neighbours for each atom (dynamic array)
  const int nnl_dim1;
  const int *nlat; // Index of the neighbouring atom (dynamic array)
  const int nlat_dim1;
  const int *nltr; // Cell index of the neighbouring atom (dynamic array)
  const int nltr_dim1;
} adjacency_list;

void printstruct(const adjacency_list adj)
{
  printf("adjacency_list:\n");
  printf("  inl: ");
  for (int i = 0; i < adj.inl_dim1; ++i)
  {
    printf("%d, ", adj.inl[i]);
  }
  printf("\n");
  printf("  nnl: ");
  for (int i = 0; i < adj.nnl_dim1; ++i)
  {
    printf("%d, ", adj.nnl[i]);
  }
  printf("\n");
  printf("  nlat: ");
  for (int i = 0; i < adj.nlat_dim1; ++i)
  {
    printf("%d, ", adj.nlat[i]);
  }
  printf("\n");
  printf("  nltr: ");
  for (int i = 0; i < adj.nltr_dim1; ++i)
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
    int maxl;              // Maximum angular momentum of all basis functions
    int nsh;               // Number of shells in this basis set
    int nao;               // Number of spherical atomic orbitals in this basis set
    double intcut;         // Integral cutoff
    double min_alpha;      // Smallest primitive exponent in the basis set
    int *nsh_id;           // Number of shells for each species (dynamic array)
    int *nsh_at;           // Number of shells for each atom (dynamic array)
    int *nao_sh;           // Number of spherical atomic orbitals for each shell (dynamic array)
    int *iao_sh;           // Index offset for each shell in the atomic orbital space (dynamic array)
    int *ish_at;           // Index offset for each atom in the shell space (dynamic array)
    int *ao2at;            // Mapping from spherical atomic orbitals to the respective atom (dynamic array)
    int *ao2sh;            // Mapping from spherical atomic orbitals to the respective shell (dynamic array)
    int *sh2at;            // Mapping from shells to the respective atom (dynamic array)
    cgto_type *cgto;      // Contracted Gaussian basis functions (dynamic 2D array)
} basis_type;

// Hamiltonian interaction data structure
typedef struct {
    // Atomic level information
    // Contiguous 2D array (size: mshell * mol_nid)
    double *selfenergy;

    // Coordination number dependence of the atomic levels
    // Contiguous 2D array (size: mshell * mol_nid)
    double *kcn;

    // Charge dependence of the atomic levels
    // Contiguous 2D array (size: mshell * mol_nid)
    double *kq1;

    // Charge dependence of the atomic levels
    // Contiguous 2D array (size: mshell * mol_nid)
    double *kq2;

    // Enhancement factor to scale the Hamiltonian elements
    // Contiguous 4D array (size: mshell * mshell * mol_nid * mol_nid)
    double *hscale;

    // Polynomial coefficients for distance-dependent enhancement factor
    // Contiguous 2D array (size: mshell * mol_nid)
    double *shpoly;

    // Atomic radius for polynomial enhancement
    // Contiguous 1D array (size: mol_nid)
    double *rad;

    // Reference occupation numbers
    // Contiguous 2D array (size: mshell * mol_nid)
    double *refocc;
} tb_hamiltonian;