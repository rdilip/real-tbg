""" Visualization functions for twisted bilayer graphene.

"""

import numpy as np
import matplotlib.pyplot as plt


def plot_unit_cell(a, d=None, c='C3'):
    """ Args:
        a: Lattice vectors, a[i] is the ith lattice vector.
        d: displacement vector
        c: color
    """
    plt.gca().set_aspect('equal')
    pts = np.array([[0., 0.], a[0], a[0]+a[1], a[1], [0.,0.]])
    if d is not None:
        pts += d
    plt.plot(pts.T[0], pts.T[1], c=c)

def plot_lattice(pts, nearest_neighbors, c='C0', alpha=.6):
    for pt in pts:
        for nn in nearest_neighbors:
            end = pt + nn
            plt.plot([pt[0], end[0]], [pt[1], end[1]], c=c, alpha=alpha)
