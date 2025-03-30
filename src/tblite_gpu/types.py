import torch

class StructureType:
    def __init__(self, 
                 nat=0, 
                 nid=0, 
                 nbd=0, 
                 id=None, 
                 num=None, 
                 sym=None, 
                 xyz=None, 
                 uhf=0, 
                 charge=0.0, 
                 lattice=None, 
                 periodic=None, 
                 bond=None, 
                 comment=""):
        """
        Python equivalent of the Fortran structure_type.

        Parameters:
        - nat (int): Number of atoms.
        - nid (int): Number of unique species.
        - nbd (int): Number of bonds.
        - id (torch.Tensor): Species identifier.
        - num (torch.Tensor): Atomic number for each species.
        - sym (list of str): Element symbol for each species.
        - xyz (torch.Tensor): Cartesian coordinates, in Bohr.
        - uhf (int): Number of unpaired electrons.
        - charge (float): Total charge.
        - lattice (torch.Tensor): Lattice parameters.
        - periodic (torch.Tensor): Periodic directions.
        - bond (torch.Tensor): Bond indices.
        - comment (str): Comment, name, or identifier for this structure.
        """
        self.nat = nat
        self.nid = nid
        self.nbd = nbd
        self.id = id
        self.num = num
        self.sym = sym
        self.xyz = xyz
        self.uhf = uhf
        self.charge = charge
        self.lattice = lattice
        self.periodic = periodic
        self.bond = bond
        self.comment = comment


class AdjacencyList:
    def __init__(self, inl=None, nnl=None, nlat=None, nltr=None):
        """
        Python equivalent of the Fortran adjacency_list.

        Parameters:
        - inl (torch.Tensor): Offset index in the neighbour map.
        - nnl (torch.Tensor): Number of neighbours for each atom.
        - nlat (torch.Tensor): Index of the neighbouring atom.
        - nltr (torch.Tensor): Cell index of the neighbouring atom.
        """
        self.inl = inl 
        self.nnl = nnl 
        self.nlat = nlat
        self.nltr = nltr

class CgtoType:
    def __init__(self, ang=-1, nprim=0, alpha=None, coeff=None):
        """
        Python equivalent of the Fortran cgto_type.

        Parameters:
        - ang (int): Angular momentum of this basis function.
        - nprim (int): Contraction length of this basis function.
        - alpha (torch.Tensor): Exponents of the primitive Gaussian functions.
        - coeff (torch.Tensor): Contraction coefficients of the primitive Gaussian functions.
        """
        self.ang = ang
        self.nprim = nprim
        self.alpha = alpha if alpha is not None else torch.zeros(12, dtype=torch.float32)
        self.coeff = coeff if coeff is not None else torch.zeros(12, dtype=torch.float32)

class BasisType:
    def __init__(self, 
                 maxl=0, 
                 nsh=0, 
                 nao=0, 
                 intcut=0.0, 
                 min_alpha=float('inf'), 
                 nsh_id=None, 
                 nsh_at=None, 
                 nao_sh=None, 
                 iao_sh=None, 
                 ish_at=None, 
                 ao2at=None, 
                 ao2sh=None, 
                 sh2at=None, 
                 cgto=None):
        """
        Python equivalent of the Fortran basis_type.

        Parameters:
        - maxl (int): Maximum angular momentum of all basis functions.
        - nsh (int): Number of shells in this basis set.
        - nao (int): Number of spherical atomic orbitals in this basis set.
        - intcut (float): Integral cutoff for Gaussian product theorem.
        - min_alpha (float): Smallest primitive exponent in the basis set.
        - nsh_id (torch.Tensor): Number of shells for each species.
        - nsh_at (torch.Tensor): Number of shells for each atom.
        - nao_sh (torch.Tensor): Number of spherical atomic orbitals for each shell.
        - iao_sh (torch.Tensor): Index offset for each shell in the atomic orbital space.
        - ish_at (torch.Tensor): Index offset for each atom in the shell space.
        - ao2at (torch.Tensor): Mapping from spherical atomic orbitals to the respective atom.
        - ao2sh (torch.Tensor): Mapping from spherical atomic orbitals to the respective shell.
        - sh2at (torch.Tensor): Mapping from shells to the respective atom.
        - cgto (torch.Tensor): Contracted Gaussian basis functions forming the basis set.
        """
        self.maxl = maxl
        self.nsh = nsh
        self.nao = nao
        self.intcut = intcut
        self.min_alpha = min_alpha
        self.nsh_id = nsh_id
        self.nsh_at = nsh_at
        self.nao_sh = nao_sh
        self.iao_sh = iao_sh
        self.ish_at = ish_at
        self.ao2at = ao2at
        self.ao2sh = ao2sh
        self.sh2at = sh2at
        self.cgto = cgto

class TbHamiltonian:
    def __init__(self, 
                 selfenergy=None, 
                 kcn=None, 
                 kq1=None, 
                 kq2=None, 
                 hscale=None, 
                 shpoly=None, 
                 rad=None, 
                 refocc=None):
        """
        Python equivalent of the Fortran tb_hamiltonian.

        Parameters:
        - selfenergy (torch.Tensor): Atomic level information.
        - kcn (torch.Tensor): Coordination number dependence of the atomic levels.
        - kq1 (torch.Tensor): Charge dependence of the atomic levels (linear term).
        - kq2 (torch.Tensor): Charge dependence of the atomic levels (quadratic term).
        - hscale (torch.Tensor): Enhancement factor to scale the Hamiltonian elements.
        - shpoly (torch.Tensor): Polynomial coefficients for distance-dependent enhancement factor.
        - rad (torch.Tensor): Atomic radius for polynomial enhancement.
        - refocc (torch.Tensor): Reference occupation numbers.
        """
        self.selfenergy = selfenergy
        self.kcn = kcn
        self.kq1 = kq1
        self.kq2 = kq2
        self.hscale = hscale
        self.shpoly = shpoly
        self.rad = rad
        self.refocc = refocc