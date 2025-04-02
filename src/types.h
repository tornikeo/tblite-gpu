#include <stdbool.h>
#include <stdlib.h>
// Equivalent C struct for cgto_type
#include <float.h> // For huge value if needed

#define MAXG 12 // Replace with the actual value of maxg in Fortran


#define SYMBOL_LENGTH 10 // Replace with the actual value of symbol_length in Fortran

typedef struct {
    int nat;                // Number of atoms
    int nid;                // Number of unique species
    int nbd;                // Number of bonds
    int *id;                // Species identifier (dynamic array)
    int *num;               // Atomic number for each species (dynamic array)
    char (*sym)[SYMBOL_LENGTH]; // Element symbol for each species (dynamic array of strings)
    double **xyz;           // Cartesian coordinates, in Bohr (dynamic 2D array)
    int uhf;                // Number of unpaired electrons
    double charge;          // Total charge
    double **lattice;       // Lattice parameters (dynamic 2D array)
    bool *periodic;         // Periodic directions (dynamic array)
    int **bond;             // Bond indices (dynamic 2D array)
    char *comment;          // Comment, name, or identifier for this structure (dynamic string)
} structure_type;

typedef struct {
    int *inl;  // Offset index in the neighbour map (dynamic array)
    int *nnl;  // Number of neighbours for each atom (dynamic array)
    int *nlat; // Index of the neighbouring atom (dynamic array)
    int *nltr; // Cell index of the neighbouring atom (dynamic array)
} adjacency_list;

typedef struct {
    int ang;               // Angular momentum of this basis function
    int nprim;             // Contraction length of this basis function
    double alpha[MAXG];    // Exponent of the primitive Gaussian functions
    double coeff[MAXG];    // Contraction coefficients of the primitive Gaussian functions
} cgto_type;

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
    cgto_type **cgto;      // Contracted Gaussian basis functions (dynamic 2D array)
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