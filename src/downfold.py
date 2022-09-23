import numpy as np
from scipy.sparse import coo_matrix, bmat
from hamiltonians import tbg_neighbor_cell_hamiltonian

def unit_cell_downfold(geom, M):
    """ Implements downfolding using Slater Koster parameters. 
    geom (TBGGeom): Geometry object for TBG. See geometry.py
    M (int): Number of vectors around Fermi level to take.
    """
    HR, Ts = tbg_neighbor_cell_hamiltonian(geom.m, geom.d, geom.t)
    H = np.sum(HR, axis=0) / len(Ts)
    e, V = np.linalg.eigvalsh(H)
    idx = np.argpartition(np.abs(e), M)
    mo_coeff = V[:, idx].T
    return mo_coeff

def greens_function_downfold(geom, M):
    coords = geom.coords
    HR, Ts = tbg_neighbor_cell_hamiltonian(geom.m, geom.d, geom.t)
    NT = len(Ts)
    assert np.allclose(Ts[0], [0., 0.])
    Ts = np.column_stack([Ts, np.zeros(NT)])

    block_form = [[0.]*NT for i in range(NT)] 
    # This can be MUCH more efficient -- but in practice it doesn't take that
    # long, and it's useful to check that things are as expected, e.g., 
    # recomputed blocks are Hermitian. Optimize later if needed...
    for i in range(NT):
        for j in range(NT):
            h = _tbg_tb_hamiltonian(coords+Ts[i], geom+Ts[j], geom.d)
            h[np.abs(h) < 1.e-12] = 0.
            block_form[i][j] = coo_matrix(h)

    H_supercell = bmat(block_form, format="csr")
    w, v = scipy.sparse.linalg.eigsh(H_supercell, k=M, which="SM")
    dm = np.einsum("ai,bi->ab", v, v.conj())
    w_uc, v_uc = np.linalg.eigh(dm[:len(coords), :len(coords)])
    mo_coeff = v_uc[:, -M:]
    return mo_coeff
    



