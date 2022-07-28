import numpy as np
from numpy.typing import ArrayLike
from typing import List, Tuple

def partition_indices_by_value(lst: ArrayLike) -> List[int]:
    """ Partitions a list into indices, where the value in each group of indices is the same.
    Args:
        lst (list): List of values.
    Returns:
        list: List of lists of indices.
    """
    ixs = np.argsort(lst)
    eq_indices = [[ixs[0]]]
    for i in range(1, len(lst)):
        prev_val = lst[eq_indices[-1][-1]]
        if np.isclose(prev_val, lst[ixs[i]]):
            eq_indices[-1].append(ixs[i])
        else:
            eq_indices.append([ixs[i]])
    return eq_indices

def check_pt_in_lst(pt: ArrayLike, lst: ArrayLike) -> bool:
    """ Checks if a point is in a list of points.
    Args:
        pt (list): Point.
        lst (list): List of points.
    Returns:
        bool: True if point is in list, False otherwise.
    """
    if len(lst.shape) == 1:
        return np.any(np.isclose(pt, lst))

    d = len(pt)
    mask = np.isclose(pt[0], lst[:, 0])
    for i in range(1, d):
        mask = mask & np.isclose(pt[i], lst[:, i])
    in_lst = np.any(mask)
    ix = np.where(mask)
    return in_lst, ix

def check_lst_in_lst(lst1: ArrayLike, lst2: ArrayLike) -> bool:
    """ Checks if lst2 is a subset of lst1.
    Args:
        lst1 (list): List of points.
        lst2 (list): List of lists of points.
    Returns:
        bool: True if lst2 is in lst1, False otherwise.
    """
    raise ValueError("This function has a serious bug -- does not count frequencies.")
    assert lst1.shape[-1] == lst2.shape[-1]
    for pt in lst2:
        if not check_pt_in_lst(pt, lst1):
            return False
    return True

def check_lsts_equal(lst1: ArrayLike, lst2: ArrayLike) -> bool:
    return check_lst_in_lst(lst1, lst2) and check_lst_in_lst(lst2, lst1)

def get_cum_dist_along_path(path: ArrayLike) -> ArrayLike:
    """ Returns the cumulative distance along a path.
    Args:
        path (np.array): List of points along path.
    Returns:
        np.array: Cumulative distance along path.
    """
    cum_dist = np.zeros(len(path))
    for i in range(len(path)-1):
        cum_dist[i+1] = cum_dist[i] + np.linalg.norm(path[i+1] - path[i])
    return cum_dist

def simdiag(Ak: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    """ Simultaneously diagonalizes matrices. Note -- this function works fine
    for our purposes, but at can be re-written to a more general diagonalization 
    routine.
    Usage:
        >>> data, V, index_partitions = simdiag([H, Px, Py])
        >>> k = np.vstack([data[1], data[2]])
        >>> e = data[0]

    Args:
        Ak (np.ndarray): List of matrices to diagonalize. Should mutually commute.
        Note: We don't actually check commutativity.
    Returns:
        data (np.ndarray): Eigenvalues with size (len(Ak), N). Each entry contains 
            the eigenvalues of each matrix.
        V (np.ndarray): Final eigenvectors. Since the matrices commute we have
            the same eigenvectors for all the matrices. V[:, i] is the ith
            eigenvector.
        index_partitions (list): List of lists of indices, corresponding to
            the set of degenerate eigenvalues at each stage of diagonalization.
            Mostly for debugging.
    """

    evals, V = np.linalg.eigh(Ak[0])
    V = np.array(V, dtype=complex)
    data = np.zeros((len(Ak), len(evals)))
    data[0] = np.array(evals)

    if len(Ak) == 1:
        return data, V
    
    m = 1

    partition = partition_indices_by_value(evals)
    for index_group in partition:
        Vdegen = V[:, index_group]
        A1_degen = np.einsum("im,ij,jn->mn", Vdegen.conj(), Ak[1], Vdegen)
        a_new, V_new = np.linalg.eigh(A1_degen)

        Vdegen_orig_basis = np.tensordot(Vdegen, V_new, axes=(1, 0))
        V[:, index_group] = Vdegen_orig_basis
        a_new = a_new.real
        data[1, index_group] = a_new

        if len(Ak) == 2:
            continue

        partition2 = partition_indices_by_value(a_new)

        for index_group2 in partition2:
            Vdegen2_orig_basis = Vdegen_orig_basis[:, index_group2]
            A2_degen = np.einsum("im,ij,jn->mn", Vdegen2_orig_basis.conj(), Ak[2], Vdegen2_orig_basis)
            a_new2, V_new2 = np.linalg.eigh(A2_degen)
            Vdegen2_orig_basis_new = np.tensordot(Vdegen2_orig_basis, V_new2, axes=(1, 0))
            orig_indices = [index_group[i] for i in index_group2]

            data[2, orig_indices] = a_new2.real
            V[:, orig_indices] = Vdegen2_orig_basis_new
    return data, V
