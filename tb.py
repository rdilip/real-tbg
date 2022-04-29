import numpy as np
import matplotlib.pyplot as plt

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

def get_bloch_hamiltonian(h, T, k):
    """
    Get the Bloch Hamiltonian for a spin chain.
    Args:
        h (np.array): Real-space hamiltonian with shape (a, a, n , n) 
        T (np.array): List of lattice translation vectors.
        k (np.array): List of points in k space.
    """
    diff = T[:, np.newaxis, :] - T[np.newaxis, :, :]
    phase = np.exp(1.j * np.einsum("kp,ijp->kij", k, diff))
    return np.einsum("kij,abij->kab", phase, h) / h.shape[-1]

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

def bloch_wavefunctions(T, k, basis=None):
    """
    Get the Bloch wavefunctions for a spin chain.
    Args:
        T (np.array): List of lattice translation vectors.
        k (np.array): List of points in k space.
    """
    if basis is None:
        basis = np.eye(len(T))
    assert T.shape[0] == basis.shape[0]
    phase = np.exp(1.j * np.einsum("ij,lj->il", k, T))
    # same number of k points as basis vectors...
    chi = np.einsum("ij,aj->ia", phase, basis) # basis is indexed by columns
    return chi

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

if __name__ == '__main__':
    N = 100
    H = setup_hamiltonian(N, J=1).reshape((1, 1, N, N))
    R = np.arange(N).reshape((N, 1))
    b = 2*np.pi*np.linalg.inv([[1.]])
    k = get_kpts((N,), b)
    Hk = get_bloch_hamiltonian(H, R, k)
    e = np.linalg.eigvalsh(Hk)
    chi = bloch_wavefunctions(R, k)

    plt.plot(k.ravel(), np.real(e))
    plt.show()

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

    Hk_ = np.zeros((Nb, Norb, Norb), dtype=complex)

    b_ = 2*np.pi*np.linalg.inv([[m]])
    T_ = (np.arange(Nb) * m).reshape((Nb, 1))
    k_ = get_kpts((Nb,), b_)
    # change this before it gives me ulceritis
    H = H.reshape((N, N))

    # Take all m orbitals. We have Nb of m orbitals each, and they'rea ll linearly independent,
    # so it's just a change of basis.
    for kix in range(len(k_)):
        for tix in range(len(T_)):
            for tix_ in range(len(T_)):
                for jix in range(Norb):
                    for jix_ in range(Norb):
                        phase = np.exp(1.j * k_[kix] * (T_[tix] - T_[tix_]))
                        Hk_[kix, jix, jix_] += phase * np.einsum("i,ij,j->",\
                                phip[tix, :, jix], H, phip[tix_, :, jix_]) / Nb
                        # This can probably be shorter; i.e., we should probably be
                        # doing everything in the individual block space instead of the
                        # larger translation space if possible...

    e = np.linalg.eigvalsh(Hk_)
    plt.plot(k_.ravel(), np.real(e))

    e = np.linalg.eigvalsh(Hk)
    plt.plot(k.ravel(), np.real(e.ravel()))
    plt.show()
        
