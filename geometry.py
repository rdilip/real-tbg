""" Dataclasses for geometric information in monolayer and bilayer graphene systems. Standard use:

    >>> from geometry import *
    >>> from visualize import *
    >>> geom = Geom(d=1.42, m=31, r=1) # d in Angstroms
    >>> pts = get_tbg_unit_cell(geom)
    >>> list(pts.keys())
            ['A1', 'B1', 'A2', 'B2']
    >>> plot_unit_cell(geom.a)
    >>> plot_lattice(pts['A1'], geom.deltas)
    >>> plot_lattice(pts['A2'], geom.deltas)
"""

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple, Dict
import warnings

def rotate(th):
    return np.array([[np.cos(th), np.sin(th)],[-np.sin(th), np.cos(th)]])

@dataclass
class MLGGeom:
    """ Geometry of monolayer graphene system.
    """
    d: float = 1.42 # distance between graphene sheets
    a: ArrayLike = field(init=False)
    b: ArrayLike = field(init=False)
    unit_cell: ArrayLike = field(init=False)
    brillouin_zone: ArrayLike = field(init=False)
    deltas: ArrayLike = field(init=False)

    def __post_init__(self):
        d = self.d 
        self.a = np.sqrt(3) * d * np.array([[1., 0.],[0.5, np.sqrt(3)/2.],])
        self.b = 2*np.pi*np.linalg.inv(self.a) # columns are vectors.
        self.unit_cell = np.array([[0., 0.], self.a[0], self.a[0]+self.a[1], self.a[1]])
        self.brillouin_zone = np.array([[0., 0.], self.b.T[0], self.b.T[0]+self.b.T[1], self.b.T[1]])
        delta_1 = d * np.array([0., 1.])
        R = rotate(2*np.pi/3.)
        self.deltas = np.array([delta_1, R@delta_1, R@R@delta_1])
        self.high_sym_pts()

    def high_sym_pts(self):
        self.gamma = np.array([0., 0.])
        self.m = 0.5 * (self.b[:, 0] + self.b[:, 1])
        self.k1 = (self.b[:, 0] + 2*self.b[:, 1]) / 3.
        self.k2 = -self.k1

    def get_dirac_path(self, Npts: int, dirac_pt, axis=0):
        """ Returns a list of points along the dirac path. Crosses 1/2 of the unit cell.
        Args:
            Npts (int): Number of points.
            axis (int): Axis along which to take points.
        Returns:
            list: List of points.
        """
        ticks = np.arange(-Npts//2+1, Npts//2+1) / Npts
        vector = self.b.T[axis]
        path = dirac_pt + ticks[:, np.newaxis] * vector / 2
        return path

@dataclass
class TBGGeom:
    d: float
    m: int
    theta: float = field(init=False)
    N: int = field(init=False)
    L: float = field(init=False)
    h: float = field(init=False)
    deltas: np.array = field(init=False)
    a: np.array = field(init=False)
    b: np.array = field(init=False)
    a_mlg: np.array = field(init=False)
    b_mlg: np.array = field(init=False)
    unit_cell: np.array = field(init=False)
    brillouin_zone: np.array = field(init=False)

    def __post_init__(self):
        d, m = self.d, self.m
        self.r = r = 1
        self.N = 4*(3*m*m +3*m*r+r*r)
        self.theta = np.arccos((3*m*m + 3*m*r + r*r/2) / (3*m*m + 3*m*r + r*r))
        self.L = d * np.sqrt(3) / (2 * np.sin(self.theta/2.))
        nn = d * np.array([0, 1])
        self.deltas = np.array([nn, rotate(2*np.pi/3) @ nn, rotate(-2*np.pi/3) @ nn])

        self.a_mlg = np.sqrt(3) * d * np.array([[1., 0.],[0.5, np.sqrt(3)/2.],])
        self.b_mlg = 2*np.pi*np.linalg.inv(self.a_mlg) # columns are vectors
        self.h = self.d * .335 / .142 # guesstimate, no corrugation
        R = rotate(self.theta / 2.)
        self.b = R @ self.b_mlg - R.T @ self.b_mlg
        self.a = 2*np.pi*np.linalg.inv(self.b) # rows are vectors
        self.unit_cell = np.array([[0., 0.], self.a[0], self.a[0]+self.a[1], self.a[1]])
        self.brillouin_zone = np.array([[0., 0.], self.b.T[0], self.b.T[0]+self.b.T[1], self.b.T[1]])

        self.high_sym_pts()

    def high_sym_pts(self):
        self.Gamma = np.array([0., 0.])
        self.M = 0.5 * (self.b[:, 0] + self.b[:, 1])
        self.K = (self.b[:, 0] + 2*self.b[:, 1]) / 3.


def get_points_in_unit_cell(pts: ArrayLike, a: ArrayLike) -> Tuple[ArrayLike]:
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

def get_tbg_unit_cell(geom: TBGGeom, z: bool=False) -> Dict:
    """ Returns coordinates of all points in tbg unit cell.
    Args:
        geom: Geometry object.
        z (bool): If true, returns points with vertical diretion specified
    Returns:
        pts: Dict of points in unit cell, indexed by sublattice and layer (A1, B1, A2, B2).
    """
    R = rotate(geom.theta / 2.)
    mn_top = np.array(np.linalg.inv(R @ geom.a_mlg.T) @\
            geom.unit_cell.T, dtype=int)
    mn_bot = np.array(np.linalg.inv(R.T @ geom.a_mlg.T) @\
            geom.unit_cell.T, dtype=int)
    pts = {}
    j = 0
    h = geom.h
    # NOTE: This is a hack -- we take an area slightly larger than the unit cell,
    # because we relate the sublattices using the deltas. Since we use a single
    # delta (deltas_rot[0]), a point outside the unit cell can be important for 
    # generating a different point inside the unit cell.
    # i.e., (point outside) + deltas_rot[0] = point inside
    for mnrange in [mn_top, mn_bot]:
        mn = np.array(np.meshgrid(np.arange(np.min(mnrange[0])-3,np.max(mnrange[0])+3),
                          np.arange(np.min(mnrange[1])-3,np.max(mnrange[1])+3)
                         )
            ).reshape(2,-1)
        sl_pts = np.tensordot(geom.a_mlg, mn, [0,0]).T + 1.e-16

        sl_pts_rot = (rotate((-1)**j * geom.theta/2.) @ sl_pts.T).T
        deltas_rot = (rotate((-1)**j * geom.theta/2.) @ geom.deltas.T).T

        unit_cell_A, _ = get_points_in_unit_cell(sl_pts_rot, geom.a)
        unit_cell_B, _ = get_points_in_unit_cell(sl_pts_rot + deltas_rot[0], geom.a)

        if z:
            if j == 0:
                unit_cell_A = np.hstack([unit_cell_A, np.zeros((unit_cell_A.shape[0], 1))])
                unit_cell_B = np.hstack([unit_cell_B, np.zeros((unit_cell_B.shape[0], 1))])
            else:
                unit_cell_A = np.hstack([unit_cell_A, h * np.ones((unit_cell_A.shape[0], 1))])
                unit_cell_B = np.hstack([unit_cell_B, h * np.ones((unit_cell_B.shape[0], 1))])

        pts[f"A{j+1}"] = unit_cell_A
        pts[f"B{j+1}"] = unit_cell_B
        j += 1
    assert sum([len(pts[k]) for k in pts]) == geom.N
    return pts
