""" Tight binding codes for bilayer graphene. These codes use Convention II (see
    https://www.physics.rutgers.edu/pythtb/_downloads/915304f3240dca549efa8f49146
    3a797/pythtb-formalism.pdf for details). The only difference is that the phase
    factors within a cell are not included (i.e., we don't have terms like e^{ik\delta_j},
    where $\delta_j$ points to an atom within a unit cell.
"""
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import time
from typing import Tuple

from geometry import MLGGeom
from utils import partition_indices_by_value

def oned_chain_hamiltonian(n, J=1, h=0., bc=1, dimer=0.0, nbands=1) -> ArrayLike:
    """
    Setup the Hamiltonian matrix for a spin chain.
    Args:
        n (int): number of spins
        J (float): coupling strength
        h (float): external field strength
        pbc (int): periodic boundary conditions. 1 for periodic, 0 for open, -1 for anti-periodic.
        dimer (float): If true, flips strength of every other bond
    """
    H = np.zeros((n, n))
    for i in range(n-1):
        H[i, i] = h
        H[i, i + 1] = H[i + 1, i] = J * (1 - dimer * (-1)**i)
    H[0, n - 1] = H[n - 1, 0] = J * (1 - dimer * (-1)**(n-1)) * bc
    return H

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


def mlg_hamiltonian(N: Tuple[int, int], t: float=1., bc=1) -> ArrayLike:
    """ Sets up the Hamiltonian for a monolayer graphene lattice. Note: this Hamiltonian function
    implicitly assumes that N = (N1, N2) corresponds to N1 tilings of the vector (\sqrt{3}/2, 1/2)
    and N2 tilings of the vector (1, 0). I *think* it should work for any lattice, but I haven't
    checked this. Use geometry.MLGGeom to get the correct geometry for consistency.
    Args:
        N (tuple): Number of cells in the chain.
        J (float): Coupling
        bc (int): boundary conditions
    Returns:
    """
    H = np.zeros((2, *N, 2, *N), dtype=complex)
    # 2 atoms per cell, one cell for lattice vector. 

    mn = np.array(np.meshgrid(np.arange(N[0]), np.arange(N[1]))).T.reshape(-1, 2)

    for (m, n) in mn:
        m1, n1 = (m + 1) % N[0], n
        m2, n2 = (m - 1) % N[0], (n + 1) % N[1]
        m3, n3 = (m - 1) % N[0], n
        m4, n4 = (m + 1) % N[0], (n - 1) % N[1]

        H[1, m, n, 0, m1, n1] += -t * (bc if m == N[0] - 1 else 1)
        H[0, m, n, 1, m2, n2] += -t * (bc if m == 0 or n == N[1] - 1 else 1)
        H[0, m, n, 1, m3, n3] += -t * (bc if m == 0 else 1)
        H[1, m, n, 0, m4, n4] += -t * (bc if m == N[0] - 1 or n == 0 else 1)

        H[0, m, n, 1, m, n] += -t
        H[1, m, n, 0, m, n] += -t


    H = H.reshape((2*N[0]*N[1], 2*N[0]*N[1]))
    return H

def mlg_k_hamiltonian(kpts: ArrayLike, t: float, d: float) -> ArrayLike:
    """ Returns the analytic Hamiltonian for a monolayer graphene lattice at specific
    k points """
    deltas = MLGGeom(d=d).deltas
    Nk = kpts.shape[0]
    Hk = np.zeros((Nk, 2, 2), dtype=complex)
    phases = np.tensordot(kpts, deltas, axes=[1,1])
    J = -t * np.exp(1.j * phases).sum(1)
    Hk[:, 0, 1] = J
    Hk[:, 1, 0] = J.conj()
    return Hk

def get_kpt_mesh(Ncells, B, shift=None):
    """
    Get k points.
    Args:
        Ncells (tuple): Tuple with form (Nx, Ny, Nz) representing the number of
            cells in each direction.
        B: (np.array): Reciprocal lattice matrix, B[i] is the ith reciprocal lattice vector.
    """
    if shift is None:
        shift = np.zeros(B.shape[1])
    ndim = len(Ncells)
    kpts = []
    kmesh = np.array(np.meshgrid(*[np.arange(-n//2+1, n//2+1) for n in Ncells])).reshape(ndim, -1).T
    kmesh = kmesh / np.array(Ncells)
    kpts = np.tensordot(kmesh, B, axes=(1, 0))
    kpts += shift
    return kpts

def high_symmetry_path(kpts: ArrayLike, pts_per_line: int) -> ArrayLike:
    """ Returns a list of points connecting high symmetry points along straight lines
    in BZ.
    Args:
        kpts (np.array): k points
        pts_per_line (int): number of points per line segment.
    """
    kpath = [kpts[0]]
    l = np.linspace(0., 1., pts_per_line + 1)[:-1]
    for j in range(1, len(kpts)):
        kvector = kpts[j] - kpts[j-1]
        kpath = np.vstack([kpath, kpts[j-1] + np.outer(l, kvector)])
    dkpath = np.cumsum(np.linalg.norm(np.diff(kpath, axis=0), axis=1))
    return kpath, dkpath

def get_translation_vectors(Ncells: Tuple[int], a: ArrayLike):
    """ Returns translation vectors for a given lattice.
    Args:
        Ncells (tuple): Tuple with form (Nx, Ny, Nz) representing the number of 
        cells in each direction.
        a (np.array): Lattice vectors (a[i] is the ith lattice vector)
    """
    ndim = len(Ncells)
    mn = np.array(np.meshgrid(*[np.arange(-n//2+1, n//2+1) for n in Ncells])).reshape(ndim, -1).T
    T = np.tensordot(mn, a, axes=(1, 0))
    return T

def embed_vectors(V, N, start_coords):
    """ Embeds vectors V into a larger space of size N 
    Args:
        V (np.array): List of vectors to embed. Last dimension should index vectors (same
            convention as linalg.eigh)
        N (int): Size of the larger space.
        start_coords (np.array): List of starting coordinates for each vector.
    Returns:
        np.array: Embedded vectors.
    """
    Nvectors = V.shape[-1]
    Nembed = len(start_coords)
    embed_vecs = np.zeros((Nembed, N, Nvectors))
    for i in range(len(start_coords)):
        for j in range(Nvectors):
            v = np.zeros(N)
            v[start_coords[i]:start_coords[i] + V.shape[-2]] = V[:, j]
            embed_vecs[i, :, j] = np.array(v)
    return embed_vecs

def get_momentum_operator(V, ks):
    """
    Get the momentum operator for a list of vectors.
    Args:
        V (np.array):  Momentum eigenstates with shape (Nk, N, Norb) = (number of k points,
            dimension, number of orbitals taken.)
        ks (np.array): List of points in k space.
    """
    Nk, N, Norb = V.shape
    assert ks.shape == (Nk,)
    P = np.zeros((N, N), dtype=complex)
    for orb_ix in range(Norb):
        for k_ix in range(Nk):
            P += ks[k_ix] * np.einsum('i,j->ij', V[k_ix, :, orb_ix].conj(),\
                    V[k_ix, :, orb_ix])
    return P

def get_bloch_wavefunction(ks, T, basis) -> np.array:
    """
    Get the Bloch wavefunction for a spin chain. The basis is a list of vectors that live
    in the larger space of the Hamiltonian (i.e., if the Hamiltonian is N x N, then each
    vector should also be of length N). Obviously for graphene you can do this more efficiently,
    but I think the general case is a bit harder.
    Args:
        k (np.array): List of points in k space.
        T (np.array): List of lattice translation vectors.
        basis (np.array): Basis vectors. Should have shape (len(T), N, Norb), where
            Norb is the number of orbitals in the basis and N is the dimension of the
            space (consider calling embed_vectors to get the correct shape). 
    Returns:
        np.array: Bloch wavefunction with shape (Nk, N, Norb) = (number of k points, dimension
            of space, number of orbitals taken.)
    """
    Nk, N, Norb = basis.shape
    assert (Nk == len(ks)) and (Nk == len(T))
    print(T.shape)

    phases = np.exp(1.j * np.einsum("ij,lj->il", ks, T))
    chi = np.tensordot(phases, basis, [1, 0]) / np.sqrt(Nk)

    return chi

def expectation(chi, A, diag=False):
    """
    Args:
        chi (np.array): Bloch wavefunction with shape (Nk, N, Norb) = (number of k points,
            dimension of space, number of orbitals taken.)
        A (np.array): Matrix to take expectation with. Should have shape (N, N).
    Returns:
        np.array: Expectation value of A with respect to chi.
    """
    if diag:
        return np.einsum("kia,ij,kja->ka", chi.conj(), A, chi)
    return np.einsum("kia,ij,kjb->kab", chi.conj(), A, chi)

def get_block_transformation_matrix(u: np.array, Nblocks: int) -> np.array:
    """
    Get the block transformation matrix for a given unitary.
    Args:
        u (np.array): Unitary.
        Nblocks (int): Number of blocks.
    Returns:
        np.array: Block transformation matrix.
    """
    nr, nc = u.shape
    U = np.zeros((nr * Nblocks, nc * Nblocks), dtype=complex)
    for i in range(Nblocks):
        U[i * nr: (i + 1) * nr, i * nc: (i + 1) * nc] = u
    return U


def basis_transform_and_relabel(Hlatt: np.array,
                                Platt: np.array,
                                U: np.array,
                                correct_k: bool=False) -> np.array:
    """ Transforms the lattice Hamiltonian to the cluster basis, diagonalizes, and 
    returns the new k points and momentum eigenstates. 
    Args:
        Hlatt: (np.array): Lattice Hamiltonian with shape (N, N)
        Platt: (np.array): Lattice crystal momentum operator with shape (N, N)
        U: (np.array): Transformation matrix from lattice basis to cluster basis.
    Returns:
        np.array: New k points in cluster basis.
        np.array: New momentum eigenstates in cluster basis.
    """
    Hclust = U.T.conj() @ Hlatt @ U
    Pclust = U.T.conj() @ Platt @ U

    ecluster, Vcluster = np.linalg.eigh(Hclust)
    degen_indices = partition_indices_by_value(ecluster)

    kout, Vout = np.zeros(ecluster.shape, dtype=complex), np.zeros(Vcluster.shape, dtype=complex)

    for index_group in degen_indices:
        Vdegen = Vcluster[:, index_group]
        Pproj = np.einsum("im,ij,jn->mn", Vdegen.conj(), Pclust, Vdegen)
        kproj, Ck = np.linalg.eigh(Pproj)
        chik_degen = np.tensordot(Ck, Vdegen, [0,1])
        k_new = np.einsum("mi,ij,mj->m", chik_degen.conj(), Pclust, chik_degen)
        kout[index_group] = k_new
        Vout[:, index_group] = chik_degen.T
    if correct_k:
        # only one dimension at a time
        zero_ix = np.isclose(kout, 0)
        Pclust2 = Pclust @ Pclust
        k_new = np.sqrt(np.einsum("im,ij,jm->m", Vout[:, zero_ix].conj(), Pclust2, Vout[:, zero_ix]))
        kout[zero_ix] = k_new

    assert np.allclose(kout.imag, 0.)
    return kout.real, Vout, ecluster

def correct_k2(P: ArrayLike, k: ArrayLike, chik: ArrayLike) -> ArrayLike:
    """ Note: can we use general powers of P for this? 
    Args:
        P: (np.array): Crystal momentum operator with shape (N, N)
        k: (np.array): k points in cluster basis with shape (N, dim)
        chik: (np.array): Bloch wavefunction with shape (Nk, N, Norb)
    Returns:
        k: (np.array): Replaced k points in cluster basis with shape (N, dim)
    """
    k = k.reshape((len(k), -1))
    P2 = P @ P
    kzero = np.all(np.isclose(k, 0.), axis=1)
    k2 = np.einsum("kim,ij,kjm->km", chik[kzero].conj(), P2, chik[kzero])
    assert np.allclose(k2.imag, 0.)

    k[kzero] = np.sqrt(k2.real)
    return k


if __name__ == '__main__':
    N = 100
    f = 1.
    H = setup_hamiltonian(N, J=1, factor=f)
    T = np.arange(N).reshape((N, 1))
    b = 2*np.pi*np.linalg.inv([[1.]])
    k = get_kpts((N,), b)
    basis = np.eye(N).reshape((len(T), N, 1))

    chik = get_bloch_wavefunction(k, T, basis)
    chik_orig = chik.copy()
    Hk = expectation(chik, H)

    e, V = np.linalg.eigh(Hk)
    
    plt.plot(k.ravel(), np.real(e))
    P = get_momentum_operator(chik, k)

    k = expectation(chik, P, diag=True)
    plt.scatter(k.ravel(), np.real(e), c='C1', marker="+", s=150)
    #    breakpoint()

    Nb = 50 # 5 blocks of length N // Nb
    m = N // Nb
    assert m * Nb == N

    Hp = setup_hamiltonian(m, J=1, factor=f)
    _, Vp = np.linalg.eigh(Hp)
    phip = embed_vectors(Vp, N, np.arange(0, N, m))


    Norb = m

    b_ = 2*np.pi*np.linalg.inv([[m]])
    T_ = (np.arange(Nb) * m).reshape((Nb, 1))
    k_ = get_kpts((Nb,), b_)
    H = H.reshape((N, N))
    chik = get_bloch_wavefunction(k_, T_, phip)

    Hk_ = expectation(chik, H)


    # Take all m orbitals. We have Nb of m orbitals each, and they'rea ll linearly independent,
    # so it's just a change of basis.

    kx = expectation(chik, P, diag=True)


    e, V = np.linalg.eigh(Hk_)
    k_extended, e_extended = [], []
    for i in range(Nb):
        for j in range(m):
            kpt = expectation(chik[i, :, j].reshape((1, N, 1)), P, diag=True)
            k_extended.append(kpt.ravel())
            e_extended.append(e[i, j])
    k_extended = np.array(k_extended).ravel()
    breakpoint()
    
    plt.scatter(k_extended, np.real(e_extended))
    plt.show()
        
