""" Tight binding codes for blayer graphene. These codes use Convention II (see
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

from geometry import MLGGeom, TBGGeom
from utils import partition_indices_by_value, simdiag


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

def embed_vectors(V: ArrayLike, N: int, start_coords: ArrayLike):
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

def get_bloch_wavefunction(ks, T, basis, shift=None) -> ArrayLike:
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

    phases = np.exp(1.j * np.einsum("ij,lj->il", ks, T))
    chi = np.tensordot(phases, basis, [1, 0]) / np.sqrt(Nk)
    return chi

def expectation(chi: ArrayLike, A: ArrayLike, diag: bool=False):
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
                                Platt: Tuple[np.array],
                                U: np.array,
                                Norb) -> np.array:
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
    Pclust = [U.T.conj() @ P @ U for P in Platt]
    ndim = len(Pclust)
    data, V = simdiag([Hclust, *Pclust])
    kr = np.vstack(data[1:]).T.round(12)
    er = data[0]
    if kr.shape[1] == 2:
        ix = np.lexsort((kr[:, 1], kr[:, 0]))
    else: 
        ix = np.lexsort((kr[:, 0],))
    kr = kr[ix]
    er = er[ix].reshape((-1, Norb))

    for start in range(0, len(kr), Norb):
        for i in range(start, start + Norb - 1):
            # check that we get the same k points at different orbital indices
            assert np.allclose(kr[i], kr[i+1]), "We aren't producing the right"\
                    " k point structure -- possibly a sorting issue"
    kr = kr[::Norb]

    return kr, er, V

def correct_k2(V: ArrayLike, P: ArrayLike, k: ArrayLike=None) -> ArrayLike:
    """ 
    Args:
        V: (np.array): Bloch functions with shape (N, N=Nk*Norb). Note that this
            will typically be an output of basis_transform_and_relabel.
        P: (np.array): Crystal momentum operator with shape (dim, N, N)
        k: (np.array): k points in cluster basis with shape (N, dim). If None,
            then all the k points are estimated; otherwise, only the zero valued
            ones.
    Returns:
        k: (np.array): Replaced k points in cluster basis with shape (N, dim)
    """
    raise ValueError("Deprecated...and probably useless except as sanity check.")
    P2 = np.array([p @ p for p in P])
    knew = k.reshape((len(k), -1))
    for d in range(P2.shape[0]):
        ki = knew[:, d]
        zero_ix = np.isclose(ki, 0.)
        knew_i = np.einsum("im,ij,jm->m", V[:, zero_ix].conj(), P2[d], V[:, zero_ix])
        assert np.allclose(knew_i.imag, 0.)
        knew[zero_ix, d] = np.sqrt(knew_i.real)
    return knew

def get_k2_error_bars(V, P):
    P2 = [p @ p for p in P]
    P2_avg = np.einsum("im,dij,jm->dm", V.conj(), P2, V)
    Pavg_2 = np.einsum("im,dij,jm->dm", V.conj(), P, V)**2
    return np.sqrt(P2_avg - Pavg_2)

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
        
