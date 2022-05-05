import numpy as np
import matplotlib.pyplot as plt
import time

def setup_hamiltonian(n, J=1, h=0., pbc=1):
    """
    Setup the Hamiltonian matrix for a spin chain.
    Args:
        n (int): number of spins
        J (float): coupling strength
        h (float): external field strength
        pbc (int): periodic boundary conditions. 1 for periodic, 0 for open, -1 for anti-periodic.
    """
    H = np.zeros((n, n))
    for i in range(n):
        H[i, i] = h
        if i > 0:
            H[i, i - 1] = J
        if i < n - 1:
            H[i, i + 1] = J
    H[0, n - 1] = J * pbc
    H[n-1, 0] = J * pbc
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
        V (np.array): List of vectors. Last dimension should index vectors (same
            convention as linalg.eigh)
        ks (np.array): List of points in k space.
    """
    Nvectors = V.shape[-1]
    d = V.shape[-2]
    Nk = len(ks)
    P = np.zeros((d, d), dtype=complex)
    for i in range(Nvectors):
        P += ks[i] * np.outer(V[:, i], V[:, i])
    return P

def get_bloch_wavefunction(ks, T, basis):
    """
    Get the Bloch wavefunction for a spin chain.
    Args:
        k (np.array): List of points in k space.
        T (np.array): List of lattice translation vectors.
        basis (np.array): Basis vectors. Should have shape (len(T), N, Norb), where
            Norb is the number of orbitals in the basis and N is the dimension of the
            space (call embed_vectors to get the correct shape).
    """
    Nk, N, Norb = basis.shape
    assert (Nk == len(ks)) and (Nk == len(T))

    phases = np.exp(1.j * np.einsum("ij,lj->il", ks, T))
    chi = np.tensordot(phases, basis, [1, 0]) / np.sqrt(Nk)

    return chi

def get_bloch_hamiltonian(chi, H):
    return np.einsum("kia,ij,kjb->kab", chi.conj(), H, chi)

if __name__ == '__main__':
    N = 100
    H = setup_hamiltonian(N, J=1)
    T = np.arange(N).reshape((N, 1))
    b = 2*np.pi*np.linalg.inv([[1.]])
    k = get_kpts((N,), b)
    basis = np.eye(N).reshape((len(T), N, 1))

    chik = get_bloch_wavefunction(k, T, basis)
    Hk = get_bloch_hamiltonian(chik, H)

    e, V = np.linalg.eigh(Hk)

    plt.plot(k.ravel(), np.real(e))
    plt.show()
    P = get_momentum_operator(chi, k)


    Nb = 5 # 5 blocks of length N // Nb
    m = N // Nb
    assert m * Nb == N

    Hp = setup_hamiltonian(m, J=1).reshape((m, m))
    # Ha = setup_hamiltonian(N // Nb, J=1, pbc=-1).reshape((1, 1, m, m))
    _, Vp = np.linalg.eigh(Hp)
    # _, Va = np.linalg.eigh(Ha)
    phip = embed_vectors(Vp, N, np.arange(0, N, m))


    # Need to change this to the length of basis
    Norb = m


    b_ = 2*np.pi*np.linalg.inv([[m]])
    T_ = (np.arange(Nb) * m).reshape((Nb, 1))
    k_ = get_kpts((Nb,), b_)
    # change this before it gives me ulceritis
    H = H.reshape((N, N))

    # Take all m orbitals. We have Nb of m orbitals each, and they'rea ll linearly independent,
    # so it's just a change of basis.
    Hk_ = get_bloch_hamiltonian(H, T_, k_, phip)

    e, V = np.linalg.eigh(Hk_)
    plt.plot(k_.ravel(), np.real(e), c='C1')

    chi_folded, Hk_folded = get_bloch_wavefunction(k_, T_, phip, H)
    E2 = np.linalg.eigvalsh(Hk_folded)

    plt.plot(k_.ravel(), np.real(E2), ls="dashed", c='C2')


    #e = np.linalg.eigvalsh(Hk)
    #plt.plot(k.ravel(), np.real(e.ravel()))
    plt.show()
        
