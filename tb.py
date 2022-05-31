import numpy as np
import matplotlib.pyplot as plt
import time
from utils import partition_indices_by_value

def setup_hamiltonian(n, J=1, h=0., bc=1, factor=1.):
    """
    Setup the Hamiltonian matrix for a spin chain.
    Args:
        n (int): number of spins
        J (float): coupling strength
        h (float): external field strength
        pbc (int): periodic boundary conditions. 1 for periodic, 0 for open, -1 for anti-periodic.
        factor (float): factor to scale the Hamiltonian by.
    """
    H = np.zeros((n, n))
    for i in range(n):
        H[i, i] = h
        Jscaled = J
        if i % 2 == 0:
            Jscaled *= factor
        if i > 0:
            H[i, i - 1] = Jscaled
        if i < n - 1:
            H[i, i + 1] = Jscaled
    H[0, n - 1] = J * bc
    H[n-1, 0] = J * bc
    return H

def get_kpts(Ncells, B):
    """
    Get k points.
    Args:
        Ncells (tuple): Tuple with form (Nx, Ny, Nz) representing the number of
            cells in each direction.
        B: (np.array): Reciprocal lattice matrix, B[i] is the ith reciprocal lattice vector.
    """
    ndim = len(Ncells)
    kpts = []
    kmesh = np.array(np.meshgrid(*[np.arange(n) for n in Ncells])).reshape(ndim, -1).T
    kmesh = kmesh / np.array(Ncells)
    kpts = np.tensordot(kmesh, B, axes=(1, 0))
    return kpts

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
    P = np.zeros((N, N), dtype=complex)
    assert Norb == 1 # for now -- later we deal with the 2 band casee
    # TOD  deal with 2D case
    for i in range(Nk):
        P += ks[i][0] * np.outer(V[i].ravel(), V[i].ravel().conj())
    return P

def get_bloch_wavefunction(ks, T, basis) -> np.array:
    """
    Get the Bloch wavefunction for a spin chain.
    Args:
        k (np.array): List of points in k space.
        T (np.array): List of lattice translation vectors.
        basis (np.array): Basis vectors. Should have shape (len(T), N, Norb), where
            Norb is the number of orbitals in the basis and N is the dimension of the
            space (call embed_vectors to get the correct shape).
    Returns:
        np.array: Bloch wavefunction with shape (Nk, N, Norb) = (number of k points, dimension
            of space, number of orbitals taken.)
    """
    print("getter")
    Nk, N, Norb = basis.shape
    assert (Nk == len(ks)) and (Nk == len(T))

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
                                U: np.array) -> np.array:
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

    assert np.allclose(kout.imag, 0.)
    return kout.real, Vout, ecluster

# TODO for Friday
# Resolve this weird ordering issue
# Go to a larger space
# Start truncation.
    


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
        
