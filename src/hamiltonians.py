""" Generates Hamiltonians with translational symmetry baked in. """

import numpy as np
from typing import Tuple, List
from numpy.typing import ArrayLike
from geometry import TBGGeom, get_tbg_unit_cell

def mlg_k_hamiltonian(kpts: ArrayLike, t: float, deltas: ArrayLike) -> ArrayLike:
    """ Returns the analytic Hamiltonian for a monolayer graphene lattice at specific
    k points """
    Nk = kpts.shape[0]
    Hk = np.zeros((Nk, 2, 2), dtype=complex)
    phases = np.tensordot(kpts, deltas, axes=[1,1])
    J = -t * np.exp(1.j * phases).sum(1)
    Hk[:, 0, 1] = J
    Hk[:, 1, 0] = J.conj()
    return Hk

def oned_chain_k_hamiltonian(kpts: ArrayLike, a: ArrayLike, J: float, h: float, dimer: float) -> ArrayLike:
    """
    Setup the k point Hamiltonian for a dimerized spin chain 
    Args:
        n (int): number of spins
        J (float): coupling strength
        h (float): external field strength
        pbc (int): periodic boundary conditions. 1 for periodic, 0 for open, -1 for anti-periodic.
        dimer (float): If true, flips strength of every other bond
    """
    t1 = J * (1 - dimer)
    t2 = J * (1 + dimer)
    phases = np.exp(1.j*np.einsum("ik,k->i", kpts, a))
    Hk = -t2 * phases[:, np.newaxis, np.newaxis] * np.array([[0., 1.],[0., 0.]])
    Hk += Hk.transpose((0,2,1)).conj()
    Hk += -t1 * np.array([[0., 1.],[1., 0.]])
    return Hk

def oned_chain_hamiltonian(n, J=1, h=0., bc=1, dimer=0.0, nbands=1) -> ArrayLike:
    """
    Setup the Hamiltonian matrix for a spin chain.
    Args:
        n (int): number of spins
        J (float): coupling strength
        h (float): external field strength
        pbc (int): periodic boundary conditions. 1 for periodic, 0 for open, 
            -1 for anti-periodic.
        dimer (float): If true, flips strength of every other bond
    """
    H = np.zeros((n, n))
    for i in range(n-1):
        H[i, i] = h
        H[i, i + 1] = H[i + 1, i] = J * (1 - dimer * (-1)**i)
    H[0, n - 1] = H[n - 1, 0] = J * (1 - dimer * (-1)**(n-1)) * bc
    return H

def mlg_hamiltonian(N: Tuple[int, int],
                    a: ArrayLike,
                    t: float=1., 
                    bc: Tuple[int, int]=(1, 1)) -> ArrayLike:
    """ Sets up the Hamiltonian for a monolayer graphene lattice. Note: this
    Hamiltonian function implicitly assumes that N = (N1, N2) corresponds to N1
    tilings of the vector (\sqrt{3}/2, 1/2) and N2 tilings of the vector (1,
    0). I *think* it should work for any lattice, but I haven't checked this.
    Use geometry.MLGGeom to get the correct geometry for consistency.

    This function right now currently returns the Hamiltonian, the basis, and
    the translation vectors. Ideally the latter two should be separate, but for
    now keep it here because it makes getting the translation vectors easy, and
    if these things arne't consistent you'll run into problems with forming the
    crystal momentum operator.

    Args:
        N (tuple): Number of cells in the chain.
        J (float): Coupling
        bc (int): boundary conditions
    Returns:
        H (np.array): Hamiltonian matrix
        basis (np.array): basis vectors, (Norb, Ncells, N)
        T (np.array): translation vectors
    """
    H = np.zeros((2, *N, 2, *N), dtype=complex)
    basis = np.zeros((2, *N, 2, *N), dtype=complex)
    T = np.zeros((*N, 2))
    # 2 atoms per cell, one cell for lattice vector. 

    mn = np.array(np.meshgrid(np.arange(N[0]), np.arange(N[1]))).T.reshape(-1, 2)

    for (m, n) in mn:
        m1, n1 = (m + 1) % N[0], n
        m2, n2 = (m - 1) % N[0], (n + 1) % N[1]
        m3, n3 = (m - 1) % N[0], n
        m4, n4 = (m + 1) % N[0], (n - 1) % N[1]

        H[1, m, n, 0, m1, n1] += -t * (bc[0] if m == N[0] - 1 else 1)
        H[0, m, n, 1, m2, n2] += -t * (bc[0] if m == 0 else 1) *\
                                        (bc[1] if n == N[1] - 1 else 1)

        H[0, m, n, 1, m3, n3] += -t * (bc[0] if m == 0 else 1)
        H[1, m, n, 0, m4, n4] += -t * (bc[0] if m == N[0] - 1 else 1) *\
                                        (bc[1] if n == 0 else 1)

        # same unit cell
        H[0, m, n, 1, m, n] += -t
        H[1, m, n, 0, m, n] += -t

        T[m, n] = m*a[0] + n*a[1]

        basis[0, m, n, 0, m, n] = 1
        basis[1, m, n, 1, m, n] = 1


    H = H.reshape((2*N[0]*N[1], 2*N[0]*N[1]))
    basis = basis.reshape((2, N[0]*N[1], 2*N[0]*N[1])).transpose((1,2,0))
    return H, basis, T

def _tbg_tb_hamiltonian(cell1: ArrayLike, cell2: ArrayLike, d: float, h: float) -> ArrayLike:
    """ Returns the tight binding matrix elements between two cells, using the Slater-
    Koster parameters for TBG.
    """
    r0 = 0.184 * np.sqrt(3) * d # decay length
    R = cell1[:, np.newaxis, :] - cell2[np.newaxis, :, :]
    Rn = np.linalg.norm(R, axis=2)
    # Vppx = -2.7 * np.exp(-(Rn - d) / r0)
    Vppx = -2.7 * (Rn <= d + 1.e-10)
    Vppz = 0.48 * np.exp(-(Rn - h) / r0)
    decay = R[:, :, 2] / Rn
    decay *= decay
    t = -Vppx * (1. - decay) - Vppz * decay
    t[np.diag_indices_from(t)] = 0.
    return t


def tbg_neighbor_cell_hamiltonian(m: int=31,
                                  d: float=1.,
                                  t: float=1.,
                                  mn: List[List[int]]=None) -> ArrayLike:
    """ Returns the real space effects of all the neighboring unit cells and 
    the corresponding translation vectors. Useful so you don't need to keep 
    rebuilding this object.
    Args:
        m (int): number of unit cells in each direction
        d (float): dimension of the unit cell
        t (float): hopping strength
        mn
    """
    geom = TBGGeom(d=d, m=m)
    if mn is None:
        mn = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],\
                [1, -1], [-1, 1], [-1, -1], [1,1]])
    Ts = np.tensordot(geom.a, mn, axes=[0,1]).T
    uc = get_tbg_unit_cell(geom, z=True)
    all_pts = np.vstack([uc[pt] for pt in uc])

    Ts = np.hstack([Ts, np.zeros((Ts.shape[0], 1))])
    HR = np.array([_tbg_tb_hamiltonian(all_pts, all_pts+T, d, geom.h) for T in Ts])
    return HR, Ts

def tbg_k_hamiltonian(kpts: ArrayLike,
                      m: int=31,
                      d: float=1.,
                      t: float=1.,
                      HR: ArrayLike=None,
                      Ts: ArrayLike=None) -> ArrayLike:
    """
    Setup the k point Hamiltonian for a tbg lattice. Assumes constant interlayer 
    spacing -- not strictly correct. This isn't very efficient since we compute all the
    differences, not just the nearest neighbor ones.

    Note: One can implement a batched version of this function, but this is 
    much, much slower at large system sizes and makes no difference at small system sizes.
    Args:
        kpts (ArrayLike): k points
        m (int): Integer characterizing rotation angle (m=31 is 1.05 degrees).

    """
    if HR is None or Ts is None:
        HR, Ts = tbg_neighbor_cell_hamiltonian(m=m, d=d, t=t)
    phases = np.exp(1.j * np.einsum("ki,ti->kt", kpts, Ts[:, :2]))
    Hk = np.tensordot(phases, HR, axes=[1,0])

    return Hk / len(Ts)


