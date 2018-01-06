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

def calculus(mat, error=1e-10, max_iters=1000, use_hierarchy_formula=True):
    '''
    Calculates the 'Calculus Type' limit matrix from superdecisions
    :param mat:
    :param error:
    :param max_iters:
    :param use_hierarchy_formula:
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
        diff = normalize_cols_dist(pows[-1], pows[-2], tmp1, tmp2, tmp3)
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
            diff = normalize_cols_dist(pows[i], nextp, tmp1, tmp2, tmp3)
            # print(pows[i])
            # print(nextp)
            # print(diff)
            if diff < error:
                return nextp / nextp.sum(axis=0)


def normalize_cols_dist(mat1, mat2, tmp1, tmp2, tmp3):
    div1 = mat1.max(axis=0)
    div2 = mat2.max(axis=0)
    for i in range(len(div1)):
        if div1[i]==0:
            div1[i]=1
        if div2[i] == 0:
            div2[i]=1
    np.divide(mat1, div1, tmp1)
    np.divide(mat2, div2.max(axis=0), tmp2)
    np.subtract(tmp1, tmp2, tmp3)
    np.absolute(tmp3, tmp3)
    return np.max(tmp3)