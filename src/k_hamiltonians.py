""" Generates Hamiltonians with translational symmetry baked in. """

import numpy as np
from numpy.typing import ArrayLike

from geometry import TBGGeom, get_tbg_unit_cell
from tb import _tbg_tb_hamiltonian, _tbg_tb_hamiltonian_batched

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

def tbg_k_hamiltonian(kpts: ArrayLike,
                      m: int=31,
                      d: float=1.,
                      t: float=1.,
                      batched=False) -> ArrayLike:
    """
    Setup the k point Hamiltonian for a tbg lattice. Assumes constant interlayer 
    spacing -- not strictly correct. This isn't very efficient since we compute all the
    differences, not just the nearest neighbor ones.
    Args:
        kpts (ArrayLike): k points
        m (int): Integer characterizing rotation angle (m=31 is 1.05 degrees).

    """
    geom = TBGGeom(d=d, m=m)
    mn = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],\
            [1, -1], [-1, 1], [-1, -1], [1,1]])
    Ts = np.tensordot(geom.a, mn, axes=[0,1]).T
    uc = get_tbg_unit_cell(geom, z=True)
    all_pts = np.vstack([uc[pt] for pt in uc])
    Hk = np.zeros((len(kpts), len(all_pts), len(all_pts)), dtype=complex)

    Ts = np.hstack([Ts, np.zeros((Ts.shape[0], 1))])
    if batched:
        all_translated_pts = all_pts[np.newaxis, :, :] + Ts[:, np.newaxis, :]
        HR = _tbg_tb_hamiltonian_batched(all_pts, all_translated_pts, d, geom.h)
    else:
        HR = np.array([_tbg_tb_hamiltonian(all_pts, all_pts+T, d, geom.h) for T in Ts])
    phases = np.exp(1.j * np.einsum("ki,ti->kt", kpts, Ts[:, :2]))
    Hk = np.tensordot(phases, HR, axes=[1,0])

    return Hk / len(Ts)

