'''
ANP row sensitivity calculations
'''

import pandas as pd
from copy import deepcopy
from pyanp.general import linear_interpolate
from enum import Enum
from pyanp.limitmatrix import calculus, priority_from_limit
import matplotlib.pyplot as plt

#class P0Type(Enum):
#    STANDARD=1
#    SMART=2
#    ORIG_WT=3


def _p_to_scalar(mat, p, p0=0.5):
    if p < p0:
        return (False, p/p0)
    else:
        return (True, (1-p)/(1-p0))



def calcp0(mat, row, cluster_nodes, orig, p0mode):
    if isinstance(p0mode, float):
        # p0 has been directly sent
        p0 = p0mode
    elif isinstance(p0mode, int):
        # This is smart mode
        cont_alt = p0mode
        left_deriv = influence_marginal(mat, row, influence_nodes=None, cluster_nodes=cluster_nodes,
                                        left_or_right=-1, p0mode=0.5)
        right_deriv = influence_marginal(mat, row, influence_nodes=None, cluster_nodes=cluster_nodes,
                                        left_or_right=+1, p0mode=0.5)
        lval = left_deriv[cont_alt]
        rval = right_deriv[cont_alt]
        p0 = lval / (lval + rval)
    else:
        # Use the original weight
        p0 = orig
    return p0

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
            # Get our p0
            p0 = calcp0(mat, row, cluster_nodes, orig, p0mode)
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



def p0mode_name(p0mode_value):
    '''
    Tells what kind of p0 the p0mode value is
    :param p0mode_value:
    :return:
    '''
    if isinstance(p0mode_value, int):
        # This is smart p0
        return 'smart'
    elif isinstance(p0mode_value, float):
        # Directvalue
        return 'direct'
    else:
        return 'original weight'

def p0mode_is_smart(p0mode_value):
    '''
    Tells what kind of p0 the p0mode value is
    :param p0mode_value:
    :return:
    '''
    return isinstance(p0mode_value, int)

def p0mode_is_smart(p0mode_value):
    '''
    Tells what kind of p0 the p0mode value is
    :param p0mode_value:
    :return:
    '''
    return isinstance(p0mode_value, float)

def influence_marginal(mat, row, influence_nodes=None, cluster_nodes=None, left_or_right=None, delta=1e-6,
                       p0mode=None, limit_matrix_calc=calculus):
    '''

    '''
    n = len(mat)
    if influence_nodes is None:
        influence_nodes = [i for i in range(n) if i != row]
    orig_lim = calculus(mat)
    orig_pri = priority_from_limit(orig_lim)[influence_nodes]
    orig_pri /= sum(abs(orig_pri))
    if not p0mode_is_smart(p0mode):
        raise ValueError("p0mode must be a direct p0 value for marginal influence")
    else:
        p0 = p0mode
    if left_or_right <= 0:
        #Calculate left deriv
        new_mat = row_adjust(mat, row, p0-delta, cluster_nodes, p0mode=p0mode)
        lim = limit_matrix_calc(new_mat)
        pri = priority_from_limit(lim)[influence_nodes]
        pri /= sum(abs(pri))
        left_deriv = (pri - orig_pri) / -delta
        if left_or_right < 0:
            #Only want left deriv
            rval = pd.Series(data = left_deriv, index=influence_nodes)
            return rval
    if left_or_right >= 0:
        # Calculate left deriv
        new_mat = row_adjust(mat, row, p0 + delta, cluster_nodes, p0mode=p0)
        lim = limit_matrix_calc(new_mat)
        pri = priority_from_limit(lim)[influence_nodes]
        pri /= sum(abs(pri))
        right_deriv = (pri - orig_pri) / delta
        if left_or_right > 0:
            # Only want right deriv
            rval = pd.Series(data = right_deriv, index=influence_nodes)
            return rval
    #If we make it here, we want avg
    rval = pd.Series(data=(left_deriv + right_deriv)/2, index=influence_nodes)
    return rval

def influence_table(mat, row, cluster_nodes=None, alt_indices=None, p0mode=None, limit_matrix_calc=calculus, graph=True):
    xs = [i / 50 for i in range(1, 50)]
    n = len(mat)
    if alt_indices is None:
        alt_indices = [i for i in range(n) if i != row]
    df = pd.DataFrame()
    p0s = pd.Series()
    df['x']=xs
    for alt in alt_indices:
        ys = []
        for x in xs:
            if isinstance(p0mode, int):
                # This means p0mode is smart, and we should do it smart wrt the alt
                p0mode = alt
            new_mat = row_adjust(mat, row, x, cluster_nodes=cluster_nodes, p0mode=p0mode)
            new_lmt = limit_matrix_calc(new_mat)
            new_pri = priority_from_limit(new_lmt)
            new_pri[row] = 0
            new_pri /= sum(new_pri)
            y = new_pri[alt]
            ys.append(y)
        label = "Alt " + str(alt)
        p0 = calcp0(mat, row, cluster_nodes, mat[row, alt], p0mode)
        x = p0
        y = linear_interpolate(xs, ys, x)
        if graph:
            plt.plot(xs, ys, label=label)
            plt.scatter(x, y, label=label+" p0")
        else:
            df[label]=ys
            p0s[label]=(x, y)
    if graph:
        plt.legend()
        plt.show()
    else:
        return df, p0s

def influence_table_plot(df, p0s):
    '''

    :param df: The 1st returned component from influence_table(graph=False): a dataframe
    :param p0s: The 2nd returned component from influence_table(graph=False): a Series of (x,y)'s
    :return:
    '''
    xs = df[df.columns[0]]
    for col, p0 in zip(df.columns[1:], p0s):
        plt.plot(xs, df[col], label=str(col))
        plt.scatter(p0[0], p0[1], label=str(col)+" p0")
    plt.legend()
    plt.show()
