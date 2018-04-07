'''
ANP row sensitivity calculations
'''

from copy import deepcopy
from enum import Enum
from pyanp.limitmatrix import calculus

class P0Type(Enum):
    STANDARD=1
    SMART=2
    ORIG_WT=3


def _p_to_scalar(mat, p, p0=0.5):
    if p < 0.5:
        return (False, p/0.5)
    else:
        return (True, (1-p)/0.5)


def _smart_p0():
    pass


def _calc_p0(orig_wt, p0mode):
    if p0mode == P0Type.ORIG_WT:
        return orig_wt
    elif p0mode == P0Type.STANDARD:
        return 0.5
    else:
        return _smart_p0()

def row_adjust(mat, row, p, cluster_nodes=None, inplace=False, p0mode=None):
    n = len(mat)
    if not inplace:
        mat = deepcopy(mat)
    if cluster_nodes is None:
        cluster_nodes = range(n)
    if row not in cluster_nodes:
        raise ValueError("Row was not in cluster_nodes, cannot adjust like that")
    for c in range(n):
        # First normalize column across cluster_nodes
        total = sum(abs(mat[cluster_nodes, c]))
        if total != 0:
            mat[cluster_nodes, c] /= total
        #Now we can continue
        orig = mat[row,c]
        if (orig != 0) and (orig != 1):
            p0 = _calc_p0(orig_wt = orig, p0mode = p0mode)
            scale_up, scalar = _p_to_scalar(mat, p, p0)
            if not scale_up:
                mat[row,c]*=scalar
                for r in cluster_nodes:
                    if r != row:
                        mat[r,c] *= (1-mat[row,c])/(1-orig)
            else:
                mat[row,c] = 1 - scalar*(1-mat[row,c])
                for r in cluster_nodes:
                    if r != row:
                        mat[r, c] *= scalar
        #Now we normalize back to original sum
        if total != 0:
            mat[cluster_nodes,c] *= total

    if not inplace:
        return mat


class InfluenceParams:
    '''
    A class to pass standard parameters to the influence analysis things
    '''
    def __init__(self, limit_fx=calculus, p0=P0Type.STANDARD):
        self.limit_fx = calculus
        self.p0 = p0


def influence_marginal(mat, row, lr_or_avg=None, delta=1e-6, influence_params=None):
    '''

    :param mat: A matrix to do marginal influence on
    :param row: The index of the row
    :param lr_or_avg: A string, which when upper cased, if it starts with L, it means left, starts with R means right
    otherwise it means average of L and R.
    :param delta: The delta x to use for the derivative calculation
    :param influence_params: The standard paramaters that any influence calculation uses.
    :return: An np.array of size [N] where N is the size of the matrix mat, of the influnece scores.
    '''
    pass