'''
Contains all limit matrix calculations

.. moduleauthor: Dr. Bill Adams
'''

import numpy as np
import pandas as pd
from copy import deepcopy
from pyanp.general import get_matrix

def _mat_pow2(mat, power):
    '''
    Calculates :math:`mat^N` where :math:`N \geq power` and N is a power of 2.
    It does this by squaring mat, and squaring that, etc, until it reaches
    the desired level.  It takes at most floor(log_2(power))+1 matrix
    multiplications to do this, which is much preferred for large powers.

    :param mat: The numpy array to raise to a power.

    :param power: The power to be greater than or equal to

    :return: The resulting power of the matrix
    '''
    last = deepcopy(mat)
    nextm = deepcopy(mat)
    count=1
    while count <= power:
        np.matmul(last, last, nextm)
        tmp = last
        last = nextm
        nextm = tmp
        count *= 2
    return last

def normalize(mat, inplace=False):
    '''
    Makes the columns of a matrix add to 1 (unless the column summed to zero, in which case it is left unchanged)
    Does this by dividing each column by the sum of that column.

    :param mat: The matrix to normalize

    :param inplace: If true normalizes the matrix sent in, otherwise it leaves that matrix alone, and returns a
        normalized copy

    :return: If inplace=False, it returns the normalized matrix, leaving the param mat unchanged.  Otherwise it
        returns nothing and normalizes the param mat.
    '''
    div = mat.sum(axis=0)
    for i in range(len(div)):
        if div[i] == 0:
            div[i] = 1.0
    if not inplace:
        return mat/div
    else:
        thetype = mat.dtype
        #Let's check that the matrix can do float arith
        old = mat[0,0]
        mat[0,0] = 1./3.
        if mat[0,0] == 0:
            # It is an integer type matrix, which causes a fail
            raise ValueError("Matrix cannot be integer type for inplace normalization.")
        #Reset
        mat[0,0]=old
        np.divide(mat, div, out=mat)

def hiearhcy_formula(mat):
    '''
    Uses the hierarchy formula to calculate the limit matrix.  This is essentially the normalization of
    the sum of higher powers of mat.

    :param mat: A square nump.array that you wish to find the limit matrix of, using the hiearchy formula.

    :return: The limit matrix, unless the matrix was not a hiearchy.  If the matrix was not a
        hierarchy we return None
    '''
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

    :param mat: The scaled supermatrix to calculate the limit matrix of

    :param error: The maximum error to allow between iterations

    :param max_iters: The maximum number of iterations before we give up, after we calculate the start power

    :param use_hierarchy_formula: If True and the matrix is for a hierarchy we use that formula instead.

    :param col_scale_type: A string if 'all' it scales mat1 by max(mat1) and similarly for mat2
        otherwise, it scales by column

    :return: The calculats limit matrix as a numpy array.
    '''
    size = len(mat)
    diff = 0.0
    start_pow = size * size +10
    start = _mat_pow2(mat, start_pow)
    if use_hierarchy_formula and (np.max(abs(start))==0):
        # This matrix is for a hiearchy, use that formula
        # But we need to check that it really is a hierarchy and not something
        # that disappears because of round off error
        start_pow = size
        start = _mat_pow2(mat, start_pow)
        if np.max(abs(start)) == 0:
            # It truly was a hieratchy, do that
            return hiearhcy_formula(mat)
    # Temporary storage matrices
    tmp1 = deepcopy(mat)
    tmp2 = deepcopy(mat)
    tmp3 = deepcopy(mat)
    # Now we need matrices to store the intermediate results
    pows = [start]
    for i in range(size - 1):
        # Add next one
        pows.append(np.matmul(mat, pows[-1]))
        diff = normalize_cols_dist(pows[-1], pows[-2], tmp1, tmp2, tmp3, col_scale_type)
        if diff < error:
            # Already converged, done
            mysum = pows[-1].sum(axis=0)
            for i in range(len(mysum)):
                if mysum[i]==0:
                    mysum[i]=1
            return pows[-1] / mysum
    for count in range(max_iters):
        nextp = pows[0]
        np.matmul(pows[-1], mat, nextp)
        for i in range(len(pows) - 1):
            pows[i] = pows[i + 1]
        pows[-1] = nextp
        # Check convergence
        for i in range(len(pows) - 1):
            diff = normalize_cols_dist(pows[i], nextp, tmp1, tmp2, tmp3, col_scale_type)
            if diff < error:
                mysum = nextp.sum(axis=0)
                for i in range(len(mysum)):
                    if mysum[i] == 0:
                        mysum[i] = 1
                return nextp / mysum
    # If we make it here, we never converged
    raise ValueError("Did not converge within "+str(max_iters)+" iterations")

def normalize_cols_dist(mat1, mat2, tmp1=None, tmp2=None, tmp3=None, col_scale_type=None):
    '''
    Calculates the distance between matrices mat1 and mat2 after they have
    been column normalized.  tmp1, tmp2, tmp3
    are temporary matrices to store the normalized versions of things.  This
    code could be called many times in a limit matrix calculation, and allocating
    and freeing those temporary storage bits could take a lot of cycles.  This
    way you allocate them once at the top level loop of the limit matrix calculation
    and they are reused again and again.

    If you do not wish to avail yourself of this savings, simply leave them as None's
    and the algorithm will allocate as appropriate

    :param mat1: First matrix to compare

    :param mat2: The other matrix to compare

    :param tmp1: A temporary storage matrix of same size as mat1 and mat2.  If None, it will be allocated inside the fx.

    :param tmp2: A temporary storage matrix of same size as mat1 and mat2.  If None, it will be allocated inside the fx.

    :param tmp3: A temporary storage matrix of same size as mat1 and mat2.  If None, it will be allocated inside the fx.

    :param col_scale_type: A string if 'all' it scales mat1 by max(mat1) and similarly for mat2
        otherwise, it scales by column

    :return: The maximum difference between the column normalized versions of mat1 and mat2
    '''
    tmp1 = tmp1 if tmp1 is not None else deepcopy(mat1)
    tmp2 = tmp2 if tmp2 is not None else deepcopy(mat1)
    tmp3 = tmp3 if tmp3 is not None else deepcopy(mat1)
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
    Returns the list of indices of columns that are zero or non_zero
    depending on the parameter non_zero

    :param mat: The matrix to search over

    :param non_zero: If False, returns the indices of columns that are zero, otherwise
        returns indices of columns that a not zero.

    :return: A list of indices of columns of the type determined by the parameter non_zero.
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

    :param mat: A supermatrix (scaled or non-scaled, both work).

    :return: List of indices that are the nodes which are hierarhical.
    '''
    size = len(mat)
    start_pow = size
    full_mat = _mat_pow2(mat, start_pow)
    return zero_cols(full_mat)

def two_two_breakdown(mat, upper_right_indices):
    '''
    Given the indices for the upper right portion of a matrix, break the matrix
    down into a 2x2 matrix with the submatrices in each piece.  Useful for
    limit matrix calculations that split the problem up into the hierarchical and
    network components and do separate calculations and then bring them together.
    :param mat: The matrix to split into

    == ==
    A   B
    C   D
    == ==

    form, where A is the "upper_right_indcies" and D is the opposite

    :param upper_right_indices: List of indices of the upper right positions.

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
    '''
    Performs the limit with sinks calculation.  We break up the matrix
    into sinks and nonsinks, and use those pieces.

    :param mat: The matrix to do the limit sinks calculation on.

    :param straight_normalizer: If False we normalize at each step, if True
        we normalize at the end.
    :return: The resulting numpy array result.
    '''
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

    :param mat: The matrix to perform the calculation on.

    :param with_limit: Do we include the final limit step?

    :return: The resulting numpy array
    '''
    n = len(mat)
    hier_nodes = hierarchy_nodes(mat)
    net_nodes = [i for i in range(n) if i not in hier_nodes]
    (B, z1, A, C) = two_two_breakdown(mat, net_nodes)
    if len(net_nodes) == n:
        print("Our network nodes are :"+str(net_nodes))
        return calculus(mat)
    elif len(hier_nodes) == n:
        return hiearhcy_formula(mat)
    limitB = calculus(B)
    limitC = calculus(C)
    lowerLeftCorner = np.matmul(A, limitB) + np.matmul(C, A)
    lowerLeftCorner = normalize(lowerLeftCorner)
    if with_limit:
        laststep = lowerLeftCorner
        diff = 1
        tmp1 = deepcopy(mat)
        tmp2 = deepcopy(mat)
        tmp3 = deepcopy(mat)
        count = 0
        while (diff > error) and (count < max_count):
            nextstep = np.matmul(A, limitB) + np.matmul(limitC, A)
            # diff = normalize_cols_dist(laststep, nextstep, tmp1, tmp2, tmp3, col_scale_type=col_scale_type)
            diff = normalize_cols_dist(laststep, nextstep, None, None, None, col_scale_type=col_scale_type)
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

def priority_from_limit(limit_matrix):
    '''
    Calculates the priority from a limit matrix, i.e. sums columns and divides by the number of
    columns.

    :param limit_matrix: The matrix to extract the priority from

    :return: 1d numpy array of the priority
    '''
    rval = limit_matrix.sum(axis=1)
    adiv = sum(rval)
    if adiv != 0:
        rval /= adiv
    return rval

def priority(matrix, limit_calc=calculus):
    '''
    Calculates the limit matrix and extracts priority from it.  Really just
    a convenience function.

    :param matrix: The scaled supermatrix to calculate the priority for

    :param limit_calc: The limit matrix calculation to use

    :return: The priority as a series
    '''
    lmat = limit_calc(matrix)
    return priority_from_limit(lmat)