""" Computes geometry of twisted bilayer graphene system. Standard use:

    >>> from geometry import *
    >>> from visualize import *
    >>> geom = Geom(d=1.42, m=31, r=1) # d in Angstroms
    >>> pts = get_tbg_unit_cell(geom)
    >>> list(pts.keys())
            ['A1', 'B1', 'A2', 'B2']
    >>> plot_unit_cell(geom.aM)
    >>> plot_lattice(pts['A1'], geom.deltas)
    >>> plot_lattice(pts['A2'], geom.deltas)
"""

from dataclasses import dataclass, field
import numpy as np

def read_geometry(fname=None):
    """ Reads geometry from yml file. If no file passed, reads and
    saves standard geometry.
    """
    pass 

def rotate(th):
    return np.array([[np.cos(th), np.sin(th)],[-np.sin(th), np.cos(th)]])

@dataclass
class Geom:
    d: int
    m: int
    r: int
    theta: float = field(init=False)
    N: int = field(init=False)
    L: float = field(init=False)
    deltas: np.array = field(init=False)
    a: np.array = field(init=False)
    b: np.array = field(init=False)
    bM: np.array = field(init=False)
    aM: np.array = field(init=False)
    unit_cell: np.array = field(init=False)

    def __post_init__(self):
        d, m, r = self.d, self.m, self.r
        self.N = 4*(3*m*m +3*m*r+r*r)
        self.theta = np.arccos((3*m*m + 3*m*r + r*r/2) / (3*m*m + 3*m*r + r*r))
        self.L = d * np.sqrt(3) / (2 * np.sin(self.theta/2.))
        nn = d * np.array([0, 1])
        self.deltas = np.array([nn, rotate(2*np.pi/3) @ nn, rotate(-2*np.pi/3) @ nn])

        self.a = np.sqrt(3) * d * np.array([[1., 0.],[0.5, np.sqrt(3)/2.],])
        self.b = 2*np.pi*np.linalg.inv(self.a) # columns are vectors
        R = rotate(self.theta / 2.)
        self.bM = R @ self.b - R.T @ self.b
        self.aM = 2*np.pi*np.linalg.inv(self.bM) # rows are vectors
        self.unit_cell = np.array([[0., 0.], self.aM[0], self.aM[0]+self.aM[1], self.aM[1]])


def get_points_in_unit_cell(pts, a):
    """ Returns all points within the unit cell defined by the vectors
    a[0] and a[1]
    Args:
        pts: Points to be checked.
        a: Unit cell vectors.
    Returns:
        pts_in_unit_cell: Points within unit cell.
        mask: Boolean array of same shape as pts. True if point is in unit cell.
    """
    eps = -(a[0] + a[1]) # necessary to avoid double counting at the boundary
    eps = (1.e-9) * (eps / np.linalg.norm(eps))
    decomp = np.tensordot(np.linalg.inv(a.T), pts + eps, [1,1]).T
    mask = (decomp[:, 0] > 0) & (decomp[:, 0] < 1) & (decomp[:, 1] > 0) & (decomp[:, 1] < 1)
    return pts[mask], mask

def get_tbg_unit_cell(geom):
    """ Returns coordinates of all points in tbg unit cell.
    Args:
        geom: Geometry object.
    Returns:
        pts: Dict of points in unit cell, indexed by sublattice and layer (A1, B1, A2, B2).
    """
    R = rotate(geom.theta / 2.)
    mn_top = np.array(np.linalg.inv(R @ geom.a.T) @\
            geom.unit_cell.T, dtype=int)
    mn_bot = np.array(np.linalg.inv(R.T @ geom.a.T) @\
            geom.unit_cell.T, dtype=int)
    pts = {}
    j = 0
    for mnrange in [mn_top, mn_bot]:
        mn = np.array(np.meshgrid(np.arange(np.min(mnrange[0])-1,np.max(mnrange[0])+1),
                          np.arange(np.min(mnrange[1])-1,np.max(mnrange[1])+1)
                         )
            ).reshape(2,-1)
        sl_pts = np.tensordot(geom.a, mn, [0,0]).T

        sl_pts_rot = (rotate((-1)**j * geom.theta/2.) @ sl_pts.T).T
        deltas_rot = (rotate((-1)**j * geom.theta/2.) @ geom.deltas.T).T

        unit_cell_A, _ = get_points_in_unit_cell(sl_pts_rot, geom.aM)
        unit_cell_B, _ = get_points_in_unit_cell(sl_pts_rot + deltas_rot[0], geom.aM)

        pts[f"A{j+1}"] = unit_cell_A
        pts[f"B{j+1}"] = unit_cell_B
        j += 1
        
    return pts
