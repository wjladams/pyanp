'''
ANP row sensitivity calculations
'''

import pandas as pd
from copy import deepcopy
from pyanp.general import linear_interpolate
from enum import Enum
from pyanp.limitmatrix import calculus, priority_from_limit
from scipy.stats import rankdata
import numpy as np
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
                                        left_or_right=-1)
        right_deriv = influence_marginal(mat, row, influence_nodes=None, cluster_nodes=cluster_nodes,
                                        left_or_right=+1)
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

def p0mode_is_direct(p0mode_value):
    '''
    Tells what kind of p0 the p0mode value is
    :param p0mode_value:
    :return:
    '''
    return isinstance(p0mode_value, float)

def influence_marginal(mat, row, influence_nodes=None, cluster_nodes=None, left_or_right=None, delta=1e-6,
                       p0mode=0.5, limit_matrix_calc=calculus):
    '''

    '''
    n = len(mat)
    if influence_nodes is None:
        influence_nodes = [i for i in range(n) if i != row]
    orig_lim = calculus(mat)
    orig_pri = priority_from_limit(orig_lim)[influence_nodes]
    orig_pri /= sum(abs(orig_pri))
    if not p0mode_is_direct(p0mode):
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

def influence_table(mat, row, p0s=None, cluster_nodes=None, influence_nodes=None, p0mode=None, limit_matrix_calc=calculus, graph=True):
    if p0s is None:
        xs = [i / 50 for i in range(1, 50)]
    else:
        xs = p0s
    n = len(mat)
    if influence_nodes is None:
        influence_nodes = [i for i in range(n) if i != row]
    df = pd.DataFrame()
    p0s = pd.Series()
    df['x']=xs
    for alt in influence_nodes:
        ys = []
        if isinstance(p0mode, int):
            # This means p0mode is smart, and we should do it smart wrt the alt
            p0mode = alt
        for x in xs:
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

def influence_limit(mat, row, cluster_nodes=None, influence_nodes=None, delta=1e-6, p0mode=0.5, limit_matrix_calc=calculus, graph=True):
    if not p0mode_is_direct(p0mode):
        raise ValueError("p0mode must be a direct p0 value for limit influence")
    n = len(mat)
    if influence_nodes is None:
        influence_nodes = [i for i in range(n) if i != row]
    df = pd.DataFrame()
    limits = pd.Series()
    p0 = 1 - delta
    p0s = pd.Series()
    for alt in influence_nodes:
        if isinstance(p0mode, int):
            # This means p0mode is smart, and we should do it smart wrt the alt
            p0mode = alt
        new_mat = row_adjust(mat, row, p0, cluster_nodes=cluster_nodes, p0mode=p0mode)
        new_lmt = limit_matrix_calc(new_mat)
        new_pri = priority_from_limit(new_lmt)
        row_pri = new_pri[row]
        new_pri[row] = 0
        new_pri /= sum(new_pri)
        new_pri[row]=row_pri
        y = new_pri[alt]
        label = "Alt " + str(alt)
        limits[label]=y
        p0 = calcp0(mat, row, cluster_nodes, mat[row, alt], p0mode)
        p0s[label]=p0
    return limits, p0s

def row_adjust_priority(mat, row, p, cluster_nodes=None, inplace=False, p0mode=None, limit_matrix_calc=calculus,
                        normalize_to_orig=False):
    if normalize_to_orig:
        old_lmt = limit_matrix_calc(mat)
        old_pri = priority_from_limit(old_lmt)
        old_val = old_pri[row]
        old_sum = sum(old_pri) - old_val
    else:
        old_sum = 1
    new_mat = row_adjust(mat, row, p, cluster_nodes=cluster_nodes, p0mode=p0mode)
    new_lmt = limit_matrix_calc(new_mat)
    new_pri = priority_from_limit(new_lmt)
    row_pri = new_pri[row]
    new_pri[row] = 0
    new_pri *= old_sum / sum(new_pri)
    new_pri[row]=row_pri
    return new_pri

def influence_fixed(mat, row, cluster_nodes=None, influence_nodes=None, delta=0.25, p0mode=0.5,
                    limit_matrix_calc=calculus):
    if not p0mode_is_direct(p0mode):
        raise ValueError("p0mode must be a direct p0 value for fixed distance influence")
    if influence_nodes is None:
        n = len(mat)
        influence_nodes = [i for i in range(n) if i != row]
    p0 = p0mode + delta
    old_pri = row_adjust_priority(mat, row, 0.5, cluster_nodes, p0mode=0.5, limit_matrix_calc=limit_matrix_calc)
    new_pri = row_adjust_priority(mat, row,p0, cluster_nodes, p0mode=p0mode, limit_matrix_calc=limit_matrix_calc)
    diff = new_pri - old_pri
    rval = pd.Series(data=diff[influence_nodes], index=influence_nodes)
    return rval

def rank_change(vec1, vec2, places_to_rank, rank_change_places=None, round_to_decimal=5):
    if (rank_change_places is None):
        rank_change_places = places_to_rank
    elif isinstance(rank_change_places, int):
        rank_change_places = [rank_change_places]
    vec1 = np.round(vec1, round_to_decimal)
    vec2 = np.round(vec2, round_to_decimal)
    if len(rank_change_places) == 1:
        rk1 = rankdata(vec1)[rank_change_places[0]]
        rk2 = rankdata(vec2)[rank_change_places[0]]
        return rk1 != rk2
    else:
        vec1 = vec1[rank_change_places]
        vec2 = vec2[rank_change_places]
        rk1 = rankdata(vec1)
        rk2 = rankdata(vec2)
        return any(rk1 != rk2)

def influence_rank(mat, row, cluster_nodes=None, influence_nodes=None,
                   limit_matrix_calc=calculus, rank_change_nodes=None, error=1e-5, upper_lower_both=0,
                   round_to_decimal=5, return_full_info=False):
    n = len(mat)
    if influence_nodes is None:
        influence_nodes = [i for i in range(n) if i != row]
    if rank_change_nodes is None:
        rank_change_nodes = influence_nodes
    #Start with upper rank change
    if upper_lower_both >= 0:
        #Initial bounds
        lower = 0.5
        upper = 0.99999
        p0mode = 0.5
        lower_pri = row_adjust_priority(mat, row, lower, cluster_nodes, p0mode=p0mode, limit_matrix_calc=limit_matrix_calc)
        upper_pri = row_adjust_priority(mat, row, upper, cluster_nodes, p0mode=p0mode, limit_matrix_calc=limit_matrix_calc)
        if not rank_change(lower_pri, upper_pri, influence_nodes, rank_change_nodes, round_to_decimal):
            # There is no rank change to start with
            upper_rank_chg = 1
        else:
            while (upper - lower) > error:
                mid = (upper + lower)/2
                mid_pri = row_adjust_priority(mat, row, mid, cluster_nodes, p0mode=p0mode,
                                              limit_matrix_calc=limit_matrix_calc)
                if rank_change(lower_pri, mid_pri, influence_nodes, rank_change_nodes, round_to_decimal):
                    upper = mid
                    upper_pri = mid_pri
                elif rank_change(mid_pri, upper_pri, influence_nodes, rank_change_nodes, round_to_decimal):
                    lower = mid
                    lower_pri = mid_pri
                else:
                    # This should not happen
                    raise ValueError("Please report this error: rank influence impossible state")
            upper_rank_chg = mid
        upper_rank_value = (1 - upper_rank_chg) / (1 - 0.5)
    if upper_lower_both <= 0:
        #Initial bounds
        lower = 0.00001
        upper = 0.5
        p0mode = 0.5
        lower_pri = row_adjust_priority(mat, row, lower, cluster_nodes, p0mode=p0mode, limit_matrix_calc=limit_matrix_calc)
        upper_pri = row_adjust_priority(mat, row, upper, cluster_nodes, p0mode=p0mode, limit_matrix_calc=limit_matrix_calc)
        if not rank_change(lower_pri, upper_pri, influence_nodes, rank_change_nodes, round_to_decimal):
            # There is no rank change to start with
            lower_rank_chg = 0.0
        else:
            while (upper - lower) > error:
                mid = (upper + lower)/2
                mid_pri = row_adjust_priority(mat, row, mid, cluster_nodes, p0mode=p0mode,
                                              limit_matrix_calc=limit_matrix_calc)
                if rank_change(lower_pri, mid_pri, influence_nodes, rank_change_nodes, round_to_decimal):
                    upper = mid
                    upper_pri = mid_pri
                elif rank_change(mid_pri, upper_pri, influence_nodes, rank_change_nodes, round_to_decimal):
                    lower = mid
                    lower_pri = mid_pri
                else:
                    # This should not happen
                    raise ValueError("Please report this error: rank influence impossible state")
            lower_rank_chg = mid
        lower_rank_value = (lower_rank_chg)/(0.5 - 0)
    if upper_lower_both < 0:
        if return_full_info:
            return lower_rank_value, lower_rank_chg
        else:
            return lower_rank_value
    elif upper_lower_both > 0:
        if return_full_info:
            return upper_rank_value, upper_rank_chg
        else:
            return upper_rank_value
    else:
        rval = max(lower_rank_value, upper_rank_value)
        if return_full_info:
            return rval, lower_rank_value, lower_rank_chg, upper_rank_value, upper_rank_chg
        else:
            return rval