'''
All ANP row sensitivity calculations are in this module.
'''

import pandas as pd
from copy import deepcopy
from pyanp.general import linear_interpolate, get_matrix
from pyanp.limitmatrix import calculus, priority_from_limit, priority
from scipy.stats import rankdata
import numpy as np
import matplotlib.pyplot as plt


def _p_to_scalar(mat, p, p0=0.5):
    '''
    Simple internal function that turns a p parameter for anp row sensitivity
    into the scalar value to multiply the row by.

    :param mat: The matrix that this p will be applied to.  As of this writing,
        it is not used in the calculation.

    :param p: The p-value to convert.  Should be a number between 0 and 1.

    :param p0: The fixed value, should be a number between 0 and 1 EXCLUSIVE

    :return: A list of two items.  The first is a boolean telling whether
        we scaled Up (True) or Down (False) and the second item is the scalar
        to multiply the row by.
    '''
    if p < p0:
        return (False, p/p0)
    else:
        return (True, (1-p)/(1-p0))



def calcp0(mat, row, cluster_nodes, orig, p0mode):
    '''
    Calculates the p0, or resting, value for the row sensitivity

    :param mat: The matrix to do row sensitivity on

    :param row: The row to do row sensitivity on

    :param cluster_nodes: The indices of the other nodes in the cluster that
        `row` is in.  Used for `inluence_marginal()` if p0mode is an integer
        meaning smart mode for alt=p0mode :param orig: The original weight,
        used if p0mode is not an integer (meaning smart) or a float
        (meaning a direct p0 value).

    :param p0mode:  This controls the calculation and has 3 cases:
        Case 1: if it is a float, you are directly setting the p0 value to whatever p0mode is.
        Case 2: if it is an integer, this is the smart p0 mode, and it treats p0mode as
        the index of the alternative/node to make continuous.
        Case 3: otherwise we assume you want original weights to be the p0 value, and
        return the parameter `orig`

    :return: The p0 or resting value, see the `p0mode` parameter for more information
    '''
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
    '''
    Performs an actual row adjust on the matrix, either inplace or returns an adjusted copy, leaving the
    original unchanged.

    :param mat: The scaled supermatrix to perform anp row sensitivity on.

    :param row: The row index to perform the anp row sensitivity on

    :param p: The p value to adjust to

    :param cluster_nodes: The other nodes in the parameter row's cluster (including row itself),
        so we can scale by cluster.  If None we do not scale by cluster.

    :param inplace: Should we change the matrix mat, or should we create a new
        one, adjust it, and return it?

    :param p0mode:  See calcp0() function

    :return: The adjusted matrix if inplace=False, and otherwise returns nothing
        and changes the matrix mat.
    '''
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



def p0mode_name(p0mode_value)->str:
    '''
    Tells what kind of p0 the p0mode value is

    :param p0mode_value: The p0mode value to get a name of

    :return: A string/human readable bit of information about the p0mode.
    '''
    if isinstance(p0mode_value, int):
        # This is smart p0
        return 'smart'
    elif isinstance(p0mode_value, float):
        # Directvalue
        return 'direct'
    else:
        return 'original weight'

def p0mode_is_smart(p0mode_value)->bool:
    '''
    Is the p0mode value a "smart value".  See calcp0 for more info on p0mode values.

    :param p0mode_value:  If it is an int, this p0mode value represents doing
        smart p0 making the node of that index (i.e.p0mode's value) smooth

    :return: True | False
    '''
    return isinstance(p0mode_value, int)

def p0mode_is_direct(p0mode_value):
    '''
    Is the p0mode value a "directly set value".  See calcp0 for more info on p0mode values.

    :param p0mode_value:  If it is an float, this p0mode value is the direct
        value to use for p0.

    :return: True | False
    '''
    return isinstance(p0mode_value, float)

def influence_marginal(mat, row, influence_nodes=None, cluster_nodes=None, left_or_right=None, delta=1e-6,
                       p0mode=0.5, limit_matrix_calc=calculus):
    '''
    Calculates the marginal influence

    :param mat: The scaled supermatrix to calculate marginal influence on

    :param row: The index of the row to perform the marginal influence on

    :param influence_nodes: The nodes to calculate the marginal influence of the row upon, if None then it assumes
        all nodes except row.

    :param cluster_nodes: The other nodes in the parameter row's cluster (including row itself),
        so we can scale by cluster.  If None we do not scale by cluster.

    :param left_or_right: An integer telling whether we should do left-hand side derivative, right-hand side
        derivative or average them.  If left_or_right < 0, then we do LHS deriv.  If left_or_right > 0, we do RHS deriv.
        Finally, if left_or_right == 0, we average LHS and RHS.

    :param delta: The delta_x to use for the derivative calculation.

    :param p0mode:  This controls the calculation and has 3 cases:
        Case 1: if it is a float, you are directly setting the p0 value to whatever p0mode is.
        Case 2: if it is an integer, this is the smart p0 mode, and it treats p0mode as
        the index of the alternative/node to make continuous.
        Case 3: otherwise we assume you want original weights to be the p0 value, and
        return the parameter `orig`

    :param limit_matrix_calc: A function which takes a single input, the matrix to take the limit of.

    :return: A pandas.Series whose indices are influence_nodes and whose values are the marginal influence
        scores of those nodes with respect to the given row.
    '''
    n = len(mat)
    if influence_nodes is None:
        #influence_nodes = [i for i in range(n) if i != row]
        influence_nodes = list(range(n))
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

def influence_table(mat, row, pvals=None, cluster_nodes=None, influence_nodes=None,
                    p0mode=None, limit_matrix_calc=calculus, graph=True, return_p0vals=False):
    '''
    Calculates the direct influence score, i.e. it calculates anp row sensitivity for each of pvals values and
    stores the new scores of the influence_nodes.

    :param mat: The scaled supermatrix to perform the calculation on

    :param row: The row to use for anp row sensitivity

    :param pvals: The values to set p to, this should be a list (or list like) object of values before 0 and 1.

    :param cluster_nodes: If you wish to normalize by cluster, this should be the indices of the nodes that are
        in row's cluster (including row itself).

    :param influence_nodes: The indices of the nodes to calculate the influence of, with respect to row. If None
        it calculates the influence of all nodes other than row.

    :param p0mode:  This controls the calculation and has 3 cases:
        Case 1: if it is a float, you are directly setting the p0 value to whatever p0mode is.
        Case 2: if it is an integer, this is the smart p0 mode, and it treats p0mode as
        the index of the alternative/node to make continuous.
        Case 3: otherwise we assume you want original weights to be the p0 value, and
        return the parameter `orig`

    :param limit_matrix_calc: A function which takes a single input, the matrix to take the limit of.

    :param graph: If True, we return a matplotlib graph, otherwise we return pandas.DataFrame, p0vals

    :param return_p0vals: If true and not doing graphing, we return a tuple of the dataframe of
        the results, and the 2nd item as Series whose index is the names of the nodes, and whose values
        are the (x,y) position of the resting p0 value

    :return: If graph=True, we return nothing, but create a matplotlib object and call plt.show().  Otherwise
        if return_p0vals is True
        we return a pair of items.  The first is the dataframe of results, whose indices are "Node 1", "Node 2", ...
        which corresponds to influence_nodes (and the indices after "Node " are the influence_node indices)
        and has 2 columns, 'x' is the pvals and 'y' is the resulting influence
        score (i.e. changed priority).  The second element is a pd.Series whose indices is the same as the dataframe
        and whose values are pairs of items (x,y) where x is the p0 value for the given alternative and the y is the
        influence score of that alternative at that p-value.

        If return_p0vals is False we return the first dataframe item only.
    '''
    if pvals is None:
        xs = [i / 50 for i in range(1, 50)]
    else:
        xs = pvals
    n = len(mat)
    if influence_nodes is None:
        influence_nodes = [i for i in range(n) if i != row]
    df = pd.DataFrame()
    pvals = pd.Series()
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
        label = "Node " + str(alt)
        p0 = calcp0(mat, row, cluster_nodes, mat[row, alt], p0mode)
        x = p0
        y = linear_interpolate(xs, ys, x)
        if graph:
            plt.plot(xs, ys, label=label)
            plt.scatter(x, y, label=label+" p0")
        else:
            df[label]=ys
            pvals[label]=(x, y)
    if graph:
        plt.legend()
        plt.show()
    else:
        if return_p0vals:
            return df, pvals
        else:
            return df

def influence_table_plot(df, p0s):
    '''
    Graphs the return value of influence_table(graph=False), useful if you want to have both the graph done
    and also the table of values.
    :param df: The 1st returned component from influence_table(graph=False): a dataframe
    :param p0s: The 2nd returned component from influence_table(graph=False): a Series of (x,y)'s
    :return: Nothing, but does call plt.show() to make the matplotlib graph visible.
    '''
    xs = df[df.columns[0]]
    for col, p0 in zip(df.columns[1:], p0s):
        plt.plot(xs, df[col], label=str(col))
        plt.scatter(p0[0], p0[1], label=str(col)+" p0")
    plt.legend()
    plt.show()

def influence_limit(mat, row, cluster_nodes=None, influence_nodes=None, delta=1e-6, p0mode=0.5, limit_matrix_calc=calculus):
    '''
    Calculates the limit influence score of the influence_nodes with respect to row.

    :param mat: The scaled supermatrix to perform the calculation on

    :param row: The row to use for anp row sensitivity

    :param cluster_nodes: If you wish to normalize by cluster, this should be the indices of the nodes that are
        in row's cluster (including row itself).
    :param influence_nodes: The indices of the nodes to calculate the influence of, with respect to row. If None
        it calculates the influence of all nodes other than row.

    :param delta: We use 1-delta for the p-value to plugin to approximate the limit as p -> 1

    :param p0mode:  This controls the calculation and has 3 cases:
        Case 1: if it is a float, you are directly setting the p0 value to whatever p0mode is.
        Case 2: if it is an integer, this is the smart p0 mode, and it treats p0mode as
        the index of the alternative/node to make continuous.
        Case 3: otherwise we assume you want original weights to be the p0 value, and
        return the parameter `orig`

    :param limit_matrix_calc: A function which takes a single input, the matrix to take the limit of.

    :return: A tuple of 2 items, the first is a pandas.Series whose indices are 'Node 1', 'Node 2'
        (and the indices after "Node " are the influence_node indices)
        and whose values are the limit value.  The second element of the returned tuple is a pandas.Series
        with the same indices and whose values are the p0 values we used for that alternative.
    '''
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
        label = "Node " + str(alt)
        limits[label]=y
        p0 = calcp0(mat, row, cluster_nodes, mat[row, alt], p0mode)
        p0s[label]=p0
    return limits, p0s

def row_adjust_priority(mat, row, p, cluster_nodes=None, p0mode=None, limit_matrix_calc=calculus,
                        normalize_to_orig=True):
    '''
    Adjusts a row of matrix and recalculates the priorities of all the nodes.

    :param mat: The scaled supermatrix to perform the calculation on

    :param row: The row to use for anp row sensitivity

    :param cluster_nodes: If you wish to normalize by cluster, this should be the indices of the nodes that are
        in row's cluster (including row itself).

    :param p0mode:  This controls the calculation and has 3 cases:
        Case 1: if it is a float, you are directly setting the p0 value to whatever p0mode is.
        Case 2: if it is an integer, this is the smart p0 mode, and it treats p0mode as
        the index of the alternative/node to make continuous.
        Case 3: otherwise we assume you want original weights to be the p0 value, and
        return the parameter `orig`

    :param limit_matrix_calc: A function which takes a single input, the matrix to take the limit of.

    :param normalize_to_orig: If True we normalize the returning priority score so that the [row] index of it has
        the same value as the original and the other values are rescaled.  Otherwise we simply normalize the priority
        vector directly.
    '''
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
    '''
    Calculates fixed influence, i.e. we do row sensitivity and calculate the difference

    :param mat: The scaled supermatrix to perform the calculation on

    :param row: The row to use for anp row sensitivity

    :param cluster_nodes: If you wish to normalize by cluster, this should be the indices of the nodes that are
        in row's cluster (including row itself).

    :param influence_nodes: The indices of the nodes to calculate the influence of, with respect to row. If None
        it calculates the influence of all nodes other than row.

    :param delta: How much to change from p0 for the fixed influence

    :param p0mode:  This controls the calculation and has 3 cases:
        Case 1: if it is a float, you are directly setting the p0 value to whatever p0mode is.
        Case 2: if it is an integer, this is the smart p0 mode, and it treats p0mode as
        the index of the alternative/node to make continuous.
        Case 3: otherwise we assume you want original weights to be the p0 value, and
        return the parameter `orig`

    :param limit_matrix_calc:

    :return: A pandas.Series whose index is influence_nodes and whose values are the influence scores of those nodes
        with respect to the row.
    '''
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
    '''
    A calculation that rounds 2 vectors to round_to_decimal places and then looks to see if
    the ranking of rounded vec1 is different from rounded vec2.

    :param vec1: A list or list-like object to check rank changing

    :param vec2: Another list or list-like object to check rank changing

    :param places_to_rank: Indices to rank.

    :param rank_change_places: Of the indices we are ranking, which ones are we checking for a change (if None we
        check all indices for rank change).

    :param round_to_decimal: The number of decimal places to round to, before checking for rank changes

    :return: True if a rank change happen and False otherwise
    '''
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
    '''
    Calculates rank influence scores.

    :param mat: The scaled supermatrix to perform the calculation on

    :param row: The row to use for anp row sensitivity

    :param cluster_nodes: If you wish to normalize by cluster, this should be the indices of the nodes that are
        in row's cluster (including row itself).

    :param influence_nodes: The indices of the nodes to calculate the influence of, with respect to row. If None
        it calculates the influence of all nodes other than row.

    :param limit_matrix_calc: A function with one parameter, that calculates the limit matrix.

    :param rank_change_nodes:  The nodes to look for rank change at

    :param error: While we narrow down our search for the p-value that causes a rank change, how close do we want
        the values between a change happening and not, to be.

    :param upper_lower_both:  Do we want to:
        Case 1: look for rank change only by changing p > 0.5=p0, if so upper_lower_both > 0.
        Case 2: look for rank change only by changing p < 0.5=p0, if so upper_lower_both < 0.
        Case 3: look for rank change by changing p>0.5 and p<0.5, if so upper_lower_both = 0.

    :param round_to_decimal:  How many decimals should we round the score to for ranking purposes

    :param return_full_info: If True returns more info, see the return section for more details

    :return: A list of one or more numbers, controlled by return_full_info and upper_lower_both:

        * If upper_lower_both < 0: our p-val search will only be for pval < 0.5

          * If return_full_info is True:
            We return pval_where_rank_chg_happens, score_of_that_pval
          * Otherwise:
            We return score_of_that_pval
        * If upper_lower_both > 0: our pval search will only be for pval > 0.5

          * If return_full_info is True:
            We return pval_where_rank_chg_happens, score_of_that_pval
          * Otherwise:
            We return score_of_that_pval
        * If upper_lower_both = 0: we check both lower and upper

          * If return_full_info is True:
            We return max_of_upper_lower_scores, lower_rank_value, lower_rank_chg_score, upper_rank_value, upper_rank_chg_score
          * Otherwise we return:
            max_of_upper_lower_scores
    '''
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