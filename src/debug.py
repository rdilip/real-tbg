import numpy as np
import matplotlib.pyplot as plt

from geometry import get_tbg_unit_cell, Geom

if __name__ == '__main__':
    geom = Geom(d=1, m=4)
    uc = get_tbg_unit_cell(geom)
    mn = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, -1], [-1, 1], [-1, -1], [1,1]])
    Ts = np.tensordot(geom.aM, mn, [0, 1]).T
    plt.figure(figsize=(10,10))
    plt.gca().set_aspect(1.)

    for i in range(len(Ts)):
        c = 'C'+str(i)
        T = Ts[i]
        for k in uc:
            if '1' in k:
                marker = 'o'
            else:
                marker = '+'
            pts = uc[k] + T
            plt.scatter(pts.T[0], pts.T[1], c=c, alpha=.5, marker=marker)

    uc_outline = np.array([[0., 0.], geom.aM[0], geom.aM[0] + geom.aM[1], geom.aM[1], [0, 0]])
    plt.plot(uc_outline.T[0], uc_outline.T[1], c='r')
    plt.show()
