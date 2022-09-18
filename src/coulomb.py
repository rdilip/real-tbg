""" TBG coulomb interaction functions """
import numpy as np
from numpy.typing import ArrayLike
from utils import ncells_to_mn
from typing import Tuple
import scipy
from scipy.spatial.distance import cdist

def gate_screened_coulomb(coords, nmax=100, epsilon=1, d=1):
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    diff_norm2 = np.einsum("ija,ija->ij")
    V = np.zeros(coords.shape[0], coords.shape[0])
    for n in range(-nmax, nmax+1):
        V += ((-1)**n) / np.sqrt(diff_norm2 + (n*d)**2)
    # TODO add a constant prefactor
    return V

def yukawa(r1: ArrayLike, 
        r2: ArrayLike, 
        alpha: float=None) -> ArrayLike:
    if alpha is None:
        alpha = 1. / (r1[:, 0].max() - r1[:, 0].min())

    rnorm = cdist(r1, r2)
    if np.abs(rnorm[0,0]) < 1.e-12:
        # diagonals are all the same
        rnorm[np.diag_indices_from(rnorm)] = np.inf
    return np.exp(-alpha*rnorm) / rnorm

def yukawa_periodic(coords: ArrayLike,
        A: ArrayLike,
        Ncells: Tuple[int, int],
        alpha: float=None) -> ArrayLike:
    """
    Args:
        coords (np.array): Coordinates, array of shape (Ncoords, 3)
        A (np.array): Array with lattice vectors. A[i] is ith lattice vector.
        alpha (float): Inverse length scale of Yukawa potential
        Ncells (tuple): Number of cells to include in sum.
    """
    mn = ncells_to_mn(Ncells)
    lattice_vectors = np.tensordot(mn, A, [1, 0])
    Norb = len(coords)
    coul = np.zeros((Norb, Norb), dtype=complex)
    lattice_vectors = np.column_stack([lattice_vectors, np.zeros(len(lattice_vectors))])

    for T in lattice_vectors:
        tmp = yukawa(coords, coords+T, alpha=alpha)
        coul += tmp
    return coul


