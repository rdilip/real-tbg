""" Plotting utility functions """
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple, Callable
import matplotlib.patches as patches
import warnings

def make_patch_function(B: ArrayLike,
                        Ncells: Tuple[int],
                        cmap,
                        vmin: float=0., 
                        vmax: float=1.):
    """
    Returns a patch function for plotting 2d band structures. Sample usage:

    >>> cmap = matplotlib.cm.get_cmap('viridis')
    >>> make_patch = make_patch_function(b, N, cmap, vals)
    >>> fig = plt.figure()

    >>> ax = fig.add_subplot(111, aspect='equal')
    >>> 
    >>> for i in range(N):
            p = make_patch(ks[i], es[i])
            ax.add_patch(p)
    >>> plt.show()

    Args:
        b (np.array): The reciprocal lattice vectors
        Ncells (tuple): The number of rows and columns in the grid
        cmap (): matplotlib.colors.Colormap object. 
        vals (np.array): List of all the values in the array, used to normalize
            the color map.
    Returns:
        Callable: A function that takes a point and returns a patch object.
    """
    assert len(Ncells) == 2
    ps = B / np.array(Ncells) / 2.
    mn = np.array([[1,1],[1,-1],[-1,-1],[-1,1]])
    corners = np.tensordot(ps, mn, [1, 1]).T
    
    if np.isclose(vmin, vmax):
        warnings.warn("vmin and vmax are equal, plotting all the same color")
        norm = lambda x: 0
    else:
        norm = lambda val: (val - vmin) / (vmax - vmin)
    
    def make_patch(center_pt, val):
        shifted_corners = center_pt + corners
        return patches.Polygon(shifted_corners, facecolor=cmap(norm(val)), edgecolor='black')
    return make_patch

def get_shifted_brillouin_zone_path(B: ArrayLike, Ncells: Tuple[int]):
    """
    Returns a list of points that represent the shifted Brillouin zone.
    Args:
        B (np.array): The reciprocal lattice vectors
        Ncells (tuple): The number of rows and columns in the grid
    Returns:
        list: A list of points that represent the shifted Brillouin zone.
    """
    b1, b2 = B.T
    bz = np.array([[0.,0], b1, b1+b2, b2, [0.,0.]])
    n1, n2 = Ncells
    bz += (b1 / n1) / 2. + (b2 / n2 / 2.) # move to corner of cell
    bz -= b1 * np.ceil(n1 / 2) / n1 + b2 * np.ceil(n2 / 2) / n2
    return bz
