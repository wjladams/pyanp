'''
Created on Dec 2, 2017

@author: Dr. Bill Adams
'''
import numpy as np

def geom_avg(vals):
    """
    Compute the geometric average of a list of values.
    
    The values need not be a list, but simply anything with a len() and []
    """
    rval=1.0
    count = 0
    for val in vals:
        val = vals[count]
        if val != 0:
            rval *= val
            count+=1
    if count != 0:
        rval = pow(rval, 1.0/count)
    return(rval)

def geom_avg_mat(mat, coeffs = None):
    '''
    Computes the geometric average of the columns of a matrix.  Returns
    an np.array of dimension [nRowsOfMat], i.e. a vector.  
    
    :param mat: Must be an numpy.array of shape [nRows, nCols]
    :param coeffs:  If not None, it is a list like object with nColsOfMat elements.
    We multiply column 0 of mat by coeffs[0], column 1 of mat by coeffs[1], etc
    and then do the geometric average of the columns.  Essentially this weights the
    columns.
    '''
    """
    """
    size = mat.shape[0]
    rval = np.ones([size])
    for row in range(size):
        if np.any(coeffs):
            theRow = mat[row,:] * np.array(coeffs)
        else:
            theRow = mat[row,:]
        rval[row] = geom_avg(theRow)
    return(rval)

def pri_expeigen(mat, error = 1e-10):
    """
    Calculates priorities using Bill's method
    """
    size = mat.shape[0]
    vec = np.ones([size])
    diff = 1
    count=0
    while diff >= error and count < 100:
        nextv = geom_avg_mat(mat, vec)
        #nextv = nextv/max(nextv)
        diff = max(abs(nextv - vec))
        vec = nextv
        count+=1
    return(vec/sum(vec))

def pri_llsm(mat):
    '''
    Calculates the priorities using the geometric mean method
    :param mat: An numpy.array of dimension [size,size]
    '''
    rval = geom_avg_mat(mat)
    rval = rval / sum(rval)
    return(rval)

# LLSM is the same geometric average, so let's make a synonym
pri_geomavg = pri_llsm

def harker_fix(mat):
    """
    Performs Harkers fix on the numpy matrix mat.  It returns a copy with the fix.
    The function does not change the matrix mat.
    :param mat: A numpy array
    :return: A copy of mat with Harker's fix applied to it
    """
    nrows = mat.shape[0]
    ncols = mat.shape[1]
    rval = mat.copy()
    for row in range(nrows):
        val = 1
        for col in range(ncols):
            if col != row and mat[row,col]==0:
                val+=1
        rval[row,row]=val
    return(rval)

def pri_eigen(mat, error = 1e-10, use_harker = False):
    '''
    Calculates the largest eigen vector of a matrix
    
    :param mat: A square numpy array.
    :return: A numpy vector that is the normalized (sum to 1) largest eigenvector.
    '''
    if use_harker:
        mat = harker_fix(mat)
    size = mat.shape[0]
    vec = np.ones([size])
    diff = 1
    while diff > error:
        nextv = np.matmul(mat, vec)
        nextv = nextv/sum(nextv)
        diff = max(abs(nextv - vec))
        vec = nextv
    return(vec)

def ratio_mat(pv):
    size = len(pv)
    rval = np.identity(n=size)
    for row in range(size):
        for col in range(size):
            rval[row,col]=pv[row]/pv[col]
    return rval

def utmrowlist_to_npmatrix(list_of_votes):
    N = len(list_of_votes)
    # Find dims
    n_float = (1 + np.sqrt(1+8*N))/2
    # This should be an integer for it to work
    n = int(n_float)
    if (np.abs(n-n_float) > 1e-10):
        raise ValueError("The list is the wrong size for a utm")
    rval = np.identity(n)
    pos = 0
    for row in range(n):
        for col in range(row+1, n):
            val = list_of_votes[pos]
            rval[row, col] = val
            if val != 0:
                rval[col, row] = 1/val
            pos += 1
    return rval