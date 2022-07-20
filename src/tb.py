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

from geometry import MLGGeom, TBGGeom
from utils import partition_indices_by_value, simdiag

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

def _tbg_tb_hamiltonian_batched(cell1: ArrayLike, cells2: ArrayLike, d: float, h: float) -> ArrayLike:
    """ Returns the tight binding matrix elements between two cells, using the Slater-
    Koster parameters for TBG. NOTE: batching doesn't seem to be significantly
    faster than a regular for-loop.
    """
    r0 = 0.184 * np.sqrt(3) * d # decay length
    R = cell1[np.newaxis, :, np.newaxis, :] - cells2[:, np.newaxis, :, :]
    # 
    Rn = np.linalg.norm(R, axis=3)
    # Vppx = -2.7 * np.exp(-(Rn - d) / r0)
    Vppx = -2.7 * (Rn <= d + 1.e-10)
    Vppz = 0.48 * np.exp(-(Rn - h) / r0)
    decay = R[:, :, :, 2] / Rn
    decay *= decay
    t = -Vppx * (1. - decay) - Vppz * decay
    Nt, N, _ = t.shape
    t[:, np.arange(N), np.arange(N)] = 0.
    return t

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
                                correct_k: bool=False,
                                square: bool=False) -> np.array:
    """ Transforms the lattice Hamiltonian to the cluster basis, diagonalizes, and 
        returns the new k points and momentum eigenstates. 
        
        Note: there is a significantly faster way to compute the correct k point values. Even
        though there can be mixing between equal valued k points, it turns out that the largest overlap
        is between the `correct` k points. I implemented this at one point and it was significantly
        faster than the current method. Unclear if it's always correct. 
    Args:
        Hlatt: (np.array): Lattice Hamiltonian with shape (N, N)
        Platt: (np.array): Lattice crystal momentum operator with shape (N, N)
        U: (np.array): Transformation matrix from lattice basis to cluster basis.
        square: (bool): If true, diagonalizes the momentum squared to correct for mixing.
    Returns:
        np.array: New k points in cluster basis.
        np.array: New momentum eigenstates in cluster basis.
    """
    Hclust = U.T.conj() @ Hlatt @ U
    Pclust = [U.T.conj() @ P @ U for P in Platt]
    ndim = len(Pclust)
    data, V = simdiag([Hclust, *Pclust])

    if correct_k:
        data["k2_corrected"] = correct_k2(V, Pclust, data[len(Platt)])
    return data, V

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
        
