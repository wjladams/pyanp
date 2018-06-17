'''
All pairwise matrix to priority vector calculations

@author: Dr. Bill Adams
'''
import numpy as np
from pyanp.general import get_matrix

######################################################
#### Priority Vector Calculations                 ####
######################################################

def geom_avg(vals)->float:
    """
    Compute the geometric average of a list of values.
    
    :param vals: A list-like of numbers

    :return: The geometric average of the values.
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

def geom_avg_mat(mat, coeffs = None)->np.ndarray:
    '''
    Computes the geometric average of the columns of a matrix.
    
    :param mat: Must be an numpy.array of shape [nRows, nCols]

    :param coeffs:  If not None, it is a list like object with nColsOfMat elements.
        We multiply column 0 of mat by coeffs[0], column 1 of mat by coeffs[1], etc
        and then do the geometric average of the columns.  Essentially this weights the
        columns.

    :return:  An np.array of dimension [nRowsOfMat], i.e. a vector. that is the
        weighted geometric average of the columns of the matrix mat.
    '''
    size = mat.shape[0]
    rval = np.ones([size])
    # Normalize the coeffs if they exist
    if np.any(coeffs):
        coeffs = size*np.array(coeffs)/sum(coeffs)
    for row in range(size):
        if np.any(coeffs):
            theRow = mat[row,:] ** np.array(coeffs)
        else:
            theRow = mat[row,:]
        rval[row] = geom_avg(theRow)
    return(rval)

def pri_expeigen(mat, error = 1e-10):
    """
    Calculates priorities using exponential (aka multiplicative) eigenvector

    :param mat: An numpy.array of shape [size, size] of pairwise comparisions.

    :param error=1e-10: The convergence error term

    :return numpy.array: The resulting exponential eigenvector as a numpy.array
        of shape [size]
    """
    size = mat.shape[0]
    vec = np.ones([size])
    diff = 1
    count=0
    while diff >= error and count < 100:
        nextv = geom_avg_mat(mat, vec)
        nextv = nextv/max(nextv)
        diff = max(abs(nextv - vec))
        vec = nextv
        count+=1
    return(vec/sum(vec))

def pri_llsm(mat):
    '''
    Calculates the priorities using the geometric mean method, aka Log Least
        Squares Method (LLSM).

    :param mat: An numpy.array of dimension [size,size]

    :return numpy.array: The resulting llsm priority vector as a numpy.array of
        shape [size]
    '''
    rval = geom_avg_mat(mat)
    rval = rval / sum(rval)
    return(rval)

# LLSM is the same geometric average, so let's make a synonym
pri_geomavg = pri_llsm

def harker_fix(mat:np.ndarray)->np.ndarray:
    """
    Performs Harker's fix on the numpy matrix mat.  It returns a copy with the fix.
    The function does not change the matrix mat.

    :param mat: A square numpy.

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


def pri_eigen(mat:np.ndarray, error:float = 1e-10, use_harker:bool = False,
              return_eigenval:bool=False):
    '''
    Calculates the largest eigen vector of a matrix.
    
    :param mat: A square numpy array.

    :param use_harker=False: Should we apply Harker's fix before computing?

    :param return_eigenval=False: If True it returns only the eigenvalue, otherwise only returns the eigenvector.

    :return numpy.array: The largest eigenvector that is the normalized (sum to 1) largest eigenvector as a numpy.array
        of shape [size] if return_eigenval=False, otherwise returns the eigenvalue as a number.
    '''
    if mat is None or mat.shape==(0,0):
        # Eigen vector of the empty matrix is []
        return np.array([])
    if use_harker:
        mat = harker_fix(mat)
    size = mat.shape[0]
    #Create our return value
    vec = np.ones([size])
    diff = 1
    while diff > error:
        nextv = np.matmul(mat, vec)
        nextv = nextv/sum(nextv)
        diff = max(abs(nextv - vec))
        vec = nextv
    if return_eigenval:
        nextv = np.matmul(mat, vec)
        return sum(nextv)
    else:
        return(vec)


def inconsistency_divisor(mat_or_size)->float:
    '''
    Calculates the inconsistency divisor for a matrix, or the size of a matrix.
    The inconsistency divisor is what you divide (eigenvalue - size) by to get the inconsistency.

    :param mat_or_size: Either a pairwise matrix, or simply the size of the pairwise
        matrix (which is what determines the inconsistency divisor).

    :return: The inconsistency divisor
    '''
    size = size_array_like(mat_or_size)
    t=size - 1
    if size<=0:
        return 1
    elif size==1:
        return 1
    elif size==2:
        return 1
    elif size==3:
        return .52 * t
    elif size==4:
        return .89 * t
    elif size==5:
        return 1.12 * t
    elif size==6:
        return 1.25 * t
    elif size==7:
        return 1.35 * t
    elif size==8:
        return 1.40 * t
    elif size==9:
        return 1.45 * t
    elif size==10:
        return 1.49 * t
    elif size==11:
        return 1.51 * t
    elif size==12:
        return 1.54 * t
    elif size==13:
        return 1.56 * t
    elif size==14:
        return 1.57 * t
    elif size==15:
        return 1.58 * t
    else:
        return 1.98 * (1 - (size - 1) / (size * (size - 1) / 2))

def incon_std(mat:np.ndarray, error:float = 1e-10, use_harker:bool = True)->float:
    '''
    Calculates the inconsistency of a pairwise matrix using the standard
    Saaty AHP/ANP theoretic formula.

    :param mat: A numpy.array of shape [size,size] of pairwise comparisons.

    :param error: The error to use for the pri_eigen calculation

    :param use_harker: Should we apply Harker's fix before the calculation?

    :return: The inconsistency.
    '''
    size = mat.shape[0]
    largest_eigen_val = pri_eigen(mat, error, use_harker, return_eigenval=True)
    return (largest_eigen_val-size)/inconsistency_divisor(mat)


#########################################################
### Priority Error Calculations                  ########
#########################################################
def prerr_euclidratio(pwmat, privec):
    '''
    Calculates the euclidean distance error between the pairwise matrix and the
    ratio matrix of a priority vector.

    This calculates using the following formula

    .. math::
        \\sqrt{ \\sum_{i,j}  \\left(\\frac{pwmat[i, j] - privec[i]}{privec[j]} \\right)^2}

    :param pwmat: A numpy.array of shape [size, size] of pairwise comparisons.

    :param privec: A numpy.array of share [size] of the priority vector to compare
        this pairwise matrix to.

    :return: The error/distance between the ratio matrix and the matrix.
    '''
    rval = 0
    diffsum = 0
    count = 0
    size = pwmat.shape[0]
    for i in range(0, size):
        for j in range(0, size):
            if (i != j) and (pwmat[i, j] != 0):
                diffsum += (pwmat[i, j] - privec[i] / privec[j]) ** 2
                count += 1
    if count == 0:
        return 0
    else:
        return diffsum ** (1.0 / 2)


def prerr_ratio_avg(pwmat, privec):
    '''
    Calculates priority error using the arithmetic average of ratio distance of
    pwmat from the ratio matrix of privec

    It averages:

    .. math::
        ratio\_greater\_1(pwmat[i, j], (privec[i]/privec[j])) - 1

    where

    .. math::
        ratio\_greater\_1(a,b) = \\begin{cases}
            1 & \\hbox{ if  a or b = 0 } \\\\
            max(a/b, b/a) & \\hbox{otherwise}
        \\end{cases}.

    :param pwmat: A numpy.array of shape [size, size] of pairwise comparisons
    :param privec: A numpy.array of shape [size] of priortiy vector
    :return: The ratio average priority vector
    '''
    diffsum = 0
    count = 0
    size = pwmat.shape[0]
    rmat = ratio_mat(privec)
    for i in range(0, size):
        for j in range(0, size):
            if (pwmat[i, j] >= 1) and (i != j):
                ratio = ratio_greater_1(pwmat[i, j], rmat[i, j])
                score = ratio - 1
                diffsum += score
                count += 1
                # print("ratio={} diffprod={}".format(ratio, diffprod))
    if count == 0:
        return 0
    else:
        return diffsum * (1.0 / count)

def ratio_greater_1(a, b):
    '''
    The ratio of a to b (or b to a) that is larger than or equal to 1.

    :param a: A numerical value for the ratio calculation

    :param b: Another numerical value

    :return: 1 if a or b is 0, otherwise max(a/b, b/a)
    '''
    if (a == 0) or (b == 0):
        return 1
    else:
        return max(a/b, b/a)

def prerr_ratio_prod(pwmat, privec):
    '''
    Calculates priority error using the geometric average of ratios of pwmat
    and the ratio matrix of privec, the formula is:

    .. math::
        \\sqrt[n(n-1)/2]{\\prod_{i=1, j=i+1} ratio\_greater\_1(pwmat[i, j], privec[i]/privec[j])}

    :param pwmat: A numpy.array of shape [size, size] of pairwise comparisons

    :param privec: A numpy.array of shape [size] of priortiy vector

    :return: The calculated error
    '''
    diffprod = 1
    count = 0
    size = pwmat.shape[0]
    for i in range(0, size):
        for j in range(0, size):
            if (pwmat[i, j] >= 1) and (i != j):
                ratio = ratio_greater_1(pwmat[i, j], privec[i]/privec[j])
                diffprod *= ratio
                count += 1
                #print("ratio={} diffprod={}".format(ratio, diffprod))
    #print(diffprod)
    if count == 0:
        return 0
    else:
        return diffprod ** (1.0 / count)

######################################################
### Creating a pairwise matrix from simpler data #####
######################################################

def ratio_mat(pv)->np.ndarray:
    '''
    Returns the ratio matrix of a vector

    :param pv: An array-like object with len(pv)=size

    :return: A numpy.array of shape [size, size] of the ratios
    '''
    size = len(pv)
    rval = np.identity(n=size)
    for row in range(size):
        for col in range(size):
            if pv[col] != 0:
                rval[row, col] = pv[row] / pv[col]
    return rval

def utmrowlist_to_npmatrix(list_of_votes)->np.ndarray:
    '''
    Convert a list of values to a pairwise matrix, assuming the list is the upper triangular part only

    :param list_of_votes: An array like, the first elements are the top row of the UTM part of the matrix.
        Then it goes to the second row, etc.

    :return: A numpy.array of the full pairwise comparison matrix.
    '''
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


######################################################
### Helper functions  ################################
######################################################
def size_array_like(mat_or_size):
    '''
    Returns the size of an array like or integer

    :param mat_or_size: Either an integer (specifying the size) or an array-like or numpy.array of shape [size, size].
        If array-like list of lists, we only use len(mat_or_size), we do not check that the array-like is actually
        square.

    :return: The parameter if it was an integer, len(mat_or_size) if param is a list, or mat_or_size.shape[0] if
        mat_or_size is a numpy.ndarray
    '''
    if isinstance(mat_or_size, (int)):
        return mat_or_size
    elif isinstance(mat_or_size, (np.ndarray)):
        return mat_or_size.shape[0]
    elif isinstance(mat_or_size, (tuple, list)):
        return len(mat_or_size)
    else:
        raise ValueError("Unable to get size")

