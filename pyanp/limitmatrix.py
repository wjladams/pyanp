'''
Contains all limit matrix calculations
@author Dr. Bill Adams
'''

import numpy as np
from copy import deepcopy


def _mat_pow2(mat, power):
    n = int(np.ceil(np.log2(power)))
    last = deepcopy(mat)
    count = 0
    nextm = deepcopy(mat)
    for i in range(n):
        np.matmul(last, last, nextm)
        tmp = last
        last = nextm
        nextm = tmp
    return last

def normalize(mat):
    div = mat.sum(axis=0)
    for i in range(len(div)):
        if div[i] == 0:
            div[i] = 1
    return mat/div

def hiearhcy_formula(mat):
    size = len(mat)
    big = _mat_pow2(mat, size+1)
    if np.count_nonzero(big) != 0:
        # Not a hiearchy, return None to indicate this
        return None
    summ = deepcopy(mat)
    thispow = deepcopy(mat)
    nextpow = deepcopy(mat)
    for i in range(size-2):
        np.matmul(thispow, mat, nextpow)
        np.add(summ, nextpow, summ)
        tmp = thispow
        thispow = nextpow
        nextpow = tmp
    rval = normalize(summ)
    return rval

def calculus(mat, error=1e-10, max_iters=1000, use_hierarchy_formula=True, col_scale_type=None):
    '''
    Calculates the 'Calculus Type' limit matrix from superdecisions
    :param mat:
    :param error:
    :param max_iters:
    :param use_hierarchy_formula:
    :param col_scale_type: A string if 'all' it scales mat1 by max(mat1) and similarly for mat2
    otherwise, it scales by column
    :return:
    '''
    size = len(mat)
    diff = 0.0
    start_pow = size+10
    start = _mat_pow2(mat, start_pow)
    tmp1 = deepcopy(mat)
    tmp2 = deepcopy(mat)
    tmp3 = deepcopy(mat)
    # Now we need matrices to store the intermediate results
    pows = [start]
    for i in range(size - 1):
        # Add next one
        pows.append(np.matmul(mat, pows[-1]))
        diff = normalize_cols_dist(pows[-1], pows[-2], tmp1, tmp2, tmp3, col_scale_type)
        # print(diff)
        if diff < error:
            # Already converged, done
            return pows[-1]
    # print(pows)
    for count in range(max_iters):
        nextp = pows[0]
        np.matmul(pows[-1], mat, nextp)
        # print(pows)
        for i in range(len(pows) - 1):
            pows[i] = pows[i + 1]
        pows[-1] = nextp
        # print(pows)
        # Check convergence
        for i in range(len(pows) - 1):
            diff = normalize_cols_dist(pows[i], nextp, tmp1, tmp2, tmp3, col_scale_type)
            # print(pows[i])
            # print(nextp)
            # print(diff)
            if diff < error:
                return nextp / nextp.sum(axis=0)


def normalize_cols_dist(mat1, mat2, tmp1, tmp2, tmp3, col_scale_type=None):
    '''

    :param mat1:
    :param mat2:
    :param tmp1:
    :param tmp2:
    :param tmp3:
    :param col_scale_type: A string if 'all' it scales mat1 by max(mat1) and similarly for mat2
    otherwise, it scales by column
    :return:
    '''
    if col_scale_type == "all":
        div1 = max(mat1)
        div2 = max(mat2)
    else:
        div1 = mat1.max(axis=0)
        div2 = mat2.max(axis=0)
        for i in range(len(div1)):
            if div1[i]==0:
                div1[i]=1
            if div2[i] == 0:
                div2[i]=1
    np.divide(mat1, div1, tmp1)
    #I think this is wrong, I'm just doing it the way I think it should
    #np.divide(mat2, div2.max(axis=0), tmp2)
    np.divide(mat2, div2, tmp2)
    np.subtract(tmp1, tmp2, tmp3)
    np.absolute(tmp3, tmp3)
    return np.max(tmp3)

def zero_cols(full_mat, non_zero=False):
    '''
    Returns the columns that are zero
    :param mat:
    :return:
    '''
    size = len(full_mat)
    rval = []
    rval_other = []
    for col in range(size):
        is_zero = True
        for row in range(size):
            if full_mat[row, col] != 0:
                is_zero = False
                break
        if is_zero:
            rval.append(col)
        else:
            rval_other.append(col)
    if non_zero:
        return rval_other
    else:
        return rval

def hierarchy_nodes(mat):
    '''
    Returns the indices of the nodes that are hierarchy ones.  The others are not hierachy
    :param mat:
    :return:
    '''
    size = len(mat)
    start_pow = size
    full_mat = _mat_pow2(mat, start_pow)
    return zero_cols(full_mat)

def two_two_breakdown(mat, upper_right_indices):
    '''

    :param mat: The matrix to split into
    A | B
    C | D
    form, where A is the "upper_right_indcies" and D is the opposite
    :param upper_righ_indices:
    :return: A list of [A, B, C, D] of those matrices
    '''
    total_n = len(mat)
    lower_left_indices = [i for i in range(total_n) if i not in upper_right_indices]
    upper_n = len(upper_right_indices)
    lower_n = total_n - upper_n
    A = np.zeros([upper_n, upper_n])
    B = np.zeros([upper_n, lower_n])
    C = np.zeros([lower_n, upper_n])
    D = np.zeros([lower_n, lower_n])
    for i in range(upper_n):
        row = upper_right_indices[i]
        for j in range(upper_n):
            col = upper_right_indices[j]
            A[i,j]=mat[row,col]
        for j in range(lower_n):
            col = lower_left_indices[j]
            B[i,j]=mat[row, col]
    for i in range(lower_n):
        row = lower_left_indices[i]
        for j in range(upper_n):
            col = upper_right_indices[j]
            C[i,j]=mat[row,col]
        for j in range(lower_n):
            col = lower_left_indices[j]
            D[i,j]=mat[row, col]
    return (A, B, C, D)

def limit_sinks(mat, straight_normalizer=True):
    n = len(mat)
    nonsinks = zero_cols(mat, non_zero=True)
    sinks = zero_cols(mat, non_zero=False)
    if len(nonsinks) == n:
        # There are no sinks, return calculus type instead
        return calculus(mat)
    # Okay we made it here, we need to get the A and B portions
    (B, z1, A, z2) = two_two_breakdown(mat, nonsinks)
    # Make sure z1 and z2 are zero
    limitB = calculus(B)
    if not straight_normalizer:
        limitB = normalize(limitB)
    axblimit = np.matmul(A, limitB)
    rval = np.zeros([n, n])
    for i in range(len(nonsinks)):
        orig_row = nonsinks[i]
        for j in range(len(nonsinks)):
            orig_col = nonsinks[j]
            rval[orig_row, orig_col] = limitB[i, j]
    for i in range(len(sinks)):
        orig_row = sinks[i]
        for j in range(len(nonsinks)):
            orig_col=nonsinks[j]
            rval[orig_row, orig_col] = axblimit[i, j]
    if straight_normalizer:
        rval = normalize(rval)
    return rval


def limit_newhierarchy(mat, with_limit=False, error=1e-10, col_scale_type = None, max_count = 1000):
    '''
    Performs the new hiearchy limit matrix calculation
    :param mat:
    :param with_limit:
    :return:
    '''
    n = len(mat)
    hier_nodes = hierarchy_nodes(mat)
    net_nodes = [i for i in range(n) if i not in hier_nodes]
    (B, z1, A, C) = two_two_breakdown(mat, net_nodes)
    if len(net_nodes) == n:
        return calculus(mat)
    elif len(hier_nodes) == n:
        return hiearhcy_formula(mat)
    limitB = calculus(B)
    limitC = calculus(C)
    lowerLeftCorner = np.matmul(A, limitB) + np.matmul(B, C)
    lowerLeftCorner = normalize(lowerLeftCorner)
    if with_limit:
        laststep = lowerLeftCorner
        diff = 1
        tmp1 = deepcopy(mat)
        tmp2 = deepcopy(mat)
        tmp3 = deepcopy(mat)
        count = 0
        while (diff > error) and (count < max_count):
            nextstep = np.matmul(A, laststep) + np.matmul(B, C)
            diff = normalize_cols_dist(laststep, nextstep, tmp1, tmp2, tmp3, col_scale_type=col_scale_type)
            laststep = nextstep
            count+=1
        lowerLeftCorner = nextstep
    rval = np.zeros([n, n])
    for i in range(len(net_nodes)):
        orig_row = net_nodes[i]
        for j in range(len(net_nodes)):
            orig_col = net_nodes[j]
            rval[orig_row, orig_col] = limitB[i, j]
    for i in range(len(hier_nodes)):
        orig_row = hier_nodes[i]
        for j in range(len(net_nodes)):
            orig_col=net_nodes[j]
            rval[orig_row, orig_col] = lowerLeftCorner[i, j]
        for j in range(len(hier_nodes)):
            orig_col=hier_nodes[j]
            rval[orig_row, orig_col] = C[i, j]
    rval = normalize(rval)
    return rval